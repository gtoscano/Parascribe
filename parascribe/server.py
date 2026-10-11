#!/usr/bin/env python3
"""
Parascribe transcription and session speaker-recognition API.

V1 batch: WhisperX transcription -> alignment -> pyannote diarization.
V2 sentences/streams: Parakeet and/or Cohere -> session-local WeSpeaker matching.

Jobs run on the configured background workers. Results are
written to /data/out/<job_id>/ so you can also retrieve transcripts as files.

Endpoints:
  GET  /health
  GET  /v1/models                      -> which whisper model / device / config
  POST /v1/transcribe                  -> multipart upload, returns {job_id}
  GET  /v1/jobs                        -> list jobs
  GET  /v1/jobs/{job_id}               -> status + result (when done)
  GET  /v1/jobs/{job_id}/transcript.txt
  GET  /v1/jobs/{job_id}/transcript.srt
  GET  /v1/jobs/{job_id}/result.json
  GET  /v1/jobs/{job_id}/timings.json
  POST /v2/sessions                   -> create a conversation/session
  GET  /v2/sessions/{session_id}      -> session transcript
  POST /v2/sessions/{session_id}/sentences
  WS   /v2/sessions/{session_id}/stream
  DELETE /v2/sessions/{session_id}/voiceprints
  POST /v2/sessions/{session_id}/end
"""
import asyncio
import json
import os
import shutil
import subprocess
import time
import traceback
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import (
    FastAPI, File, Form, HTTPException, UploadFile, WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from sentence_asr import DEFAULT_MODEL as DEFAULT_SENTENCE_ASR_MODEL
from sentence_asr import ParakeetSentenceTranscriber
from sentence_asr import (
    ASR_MODES, COHERE_MODEL, CohereSentenceTranscriber,
    run_sentence_models, validate_asr_request,
)
from session_store import SessionEnded, SessionNotFound, SessionStore
from voiceprints import DEFAULT_MODEL as DEFAULT_VOICEPRINT_MODEL
from voiceprints import VoiceprintExtractor

# ----------------------------------------------------------------------------
# Config (override via environment in the compose file)
# ----------------------------------------------------------------------------
# Replica identity: a non-negative integer, unique per replica in a cluster.
# It is prefixed onto every job_id ("<id>-<token>") so an nginx gateway can
# route GET /v1/jobs/<job_id> back to the replica that owns the (per-process)
# job registry. Defaults to 0 for a standalone server.
REPLICA_ID = os.environ.get("REPLICA_ID", "0")

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
DEVICE = os.environ.get("DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16")  # try "int8" if VRAM tight
DEFAULT_LANGUAGE = os.environ.get("DEFAULT_LANGUAGE", "en")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
SESSION_ASR_MODEL = os.environ.get(
    "SESSION_ASR_MODEL", DEFAULT_SENTENCE_ASR_MODEL)
SESSION_ASR_MODEL_DIR = os.environ.get("SESSION_ASR_MODEL_DIR") or None
SESSION_ASR_MODE = os.environ.get("SESSION_ASR_MODE", "parakeet")
if SESSION_ASR_MODE not in ASR_MODES:
    raise ValueError(f"Invalid SESSION_ASR_MODE: {SESSION_ASR_MODE}")
COHERE_MODEL_PATH = os.environ.get("COHERE_MODEL_PATH", "")
COHERE_BACKEND = os.environ.get("COHERE_BACKEND", "cpu")
SESSION_ASR_QUANTIZATION = os.environ.get("SESSION_ASR_QUANTIZATION", "int8")
SESSION_ASR_THREADS = int(os.environ.get(
    "SESSION_ASR_THREADS", str(max(1, (os.cpu_count() or 4) // 2))))
SESSION_ASR_MIN_CONFIDENCE = float(
    os.environ.get("SESSION_ASR_MIN_CONFIDENCE", "0"))
VOICEPRINT_MODEL = os.environ.get("VOICEPRINT_MODEL", DEFAULT_VOICEPRINT_MODEL)
VOICEPRINT_SIMILARITY_THRESHOLD = float(
    os.environ.get("VOICEPRINT_SIMILARITY_THRESHOLD", "0.50"))
VOICEPRINT_UPDATE_THRESHOLD = float(
    os.environ.get("VOICEPRINT_UPDATE_THRESHOLD", "0.70"))
STREAM_MAX_UTTERANCE_SECONDS = float(
    os.environ.get("STREAM_MAX_UTTERANCE_SECONDS", "60"))
SENTENCE_MAX_UPLOAD_BYTES = int(
    os.environ.get("SENTENCE_MAX_UPLOAD_BYTES", str(32 * 1024 * 1024)))

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
IN_DIR = DATA_DIR / "in"
OUT_DIR = DATA_DIR / "out"
for d in (IN_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)
SESSION_STORE = SessionStore(DATA_DIR / "sessions.sqlite3")

# ----------------------------------------------------------------------------
# PyTorch >= 2.6 defaults torch.load(weights_only=True), which breaks loading
# the pyannote VAD/diarization and wav2vec align checkpoints (they contain
# pickled config objects). These models are downloaded from HF with the user's
# own token, so loading them with weights_only=False is safe here.
# ----------------------------------------------------------------------------
import torch  # noqa: E402

_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    # Force (not setdefault) — pyannote via lightning passes weights_only=True
    # explicitly, so a default would be ignored.
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat

# ----------------------------------------------------------------------------
# Per-worker model sets. Each background worker owns an isolated ModelSet
# (asr / align / diarize), loaded lazily on its first job and reused after.
#
# Isolation is what makes concurrency safe: pyannote 3.x diarization is NOT
# thread-safe and must never be shared across threads. Because every worker
# runs single-threaded over its OWN models, no cross-worker lock is needed and
# GPU jobs can truly overlap (bounded by WORKERS).
# ----------------------------------------------------------------------------
import whisperx  # noqa: E402


def _load_diarize_pipeline():
    """DiarizationPipeline import path has moved across whisperx versions."""
    try:
        from whisperx.diarize import DiarizationPipeline  # newer
    except Exception:
        from whisperx import DiarizationPipeline  # older
    return DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)


class ModelSet:
    """One private set of GPU models, owned by a single worker thread."""

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self._asr = None
        self._align: Dict[str, Any] = {}
        self._diarize = None
        self._sentence_asr = None
        self._cohere = None
        self._voiceprint = None

    def get_asr(self):
        if self._asr is None:
            self._asr = whisperx.load_model(
                WHISPER_MODEL, DEVICE, compute_type=COMPUTE_TYPE,
                language=DEFAULT_LANGUAGE if DEFAULT_LANGUAGE else None,
            )
        return self._asr

    def get_align(self, language_code: str):
        if language_code not in self._align:
            model_a, metadata = whisperx.load_align_model(
                language_code=language_code, device=DEVICE)
            self._align[language_code] = (model_a, metadata)
        return self._align[language_code]

    def get_diarize(self):
        if self._diarize is None:
            self._diarize = _load_diarize_pipeline()
        return self._diarize

    def get_voiceprint(self):
        if self._voiceprint is None:
            self._voiceprint = VoiceprintExtractor(
                model_name=VOICEPRINT_MODEL, device=DEVICE, token=HF_TOKEN)
        return self._voiceprint

    def get_sentence_asr(self):
        if self._sentence_asr is None:
            self._sentence_asr = ParakeetSentenceTranscriber(
                model_name=SESSION_ASR_MODEL,
                model_dir=SESSION_ASR_MODEL_DIR,
                quantization=SESSION_ASR_QUANTIZATION,
                threads=SESSION_ASR_THREADS,
                min_confidence=SESSION_ASR_MIN_CONFIDENCE,
            )
        return self._sentence_asr

    def get_cohere(self):
        if self._cohere is None:
            self._cohere = CohereSentenceTranscriber(
                model_path=COHERE_MODEL_PATH, backend=COHERE_BACKEND,
                threads=SESSION_ASR_THREADS)
        return self._cohere


# ----------------------------------------------------------------------------
# Core transcription routine (runs in the worker thread)
# ----------------------------------------------------------------------------
def _relabel_speakers(segments):
    """Map raw pyannote labels (SPEAKER_00, ...) to friendly 'Speaker N',
    numbered by order of first appearance."""
    order: Dict[str, str] = {}
    for seg in segments:
        spk = seg.get("speaker")
        if spk and spk not in order:
            order[spk] = f"Speaker {len(order) + 1}"
    for seg in segments:
        spk = seg.get("speaker")
        seg["speaker_label"] = order.get(spk, "Speaker ?") if spk else "Speaker ?"
    return order


def _to_txt(segments) -> str:
    """Speaker-labeled plain text, merging consecutive segments per speaker."""
    lines, cur_spk, buf = [], None, []
    for seg in segments:
        spk = seg.get("speaker_label", "Speaker ?")
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if spk != cur_spk:
            if buf:
                lines.append(f"{cur_spk}: {' '.join(buf)}")
            cur_spk, buf = spk, [text]
        else:
            buf.append(text)
    if buf:
        lines.append(f"{cur_spk}: {' '.join(buf)}")
    return "\n\n".join(lines) + "\n"


def _fmt_ts(seconds: float) -> str:
    if seconds is None:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _to_srt(segments) -> str:
    out = []
    for i, seg in enumerate(segments, 1):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        spk = seg.get("speaker_label", "")
        prefix = f"[{spk}] " if spk and spk != "Speaker ?" else ""
        out.append(f"{i}\n{_fmt_ts(seg.get('start'))} --> {_fmt_ts(seg.get('end'))}\n{prefix}{text}\n")
    return "\n".join(out)


def _timed(job: Dict[str, Any], name: str, func, *args, **kwargs):
    """Run ``func`` and retain its wall-clock duration, including on failure."""
    started = time.perf_counter()
    try:
        return func(*args, **kwargs)
    finally:
        job["timings"][name] = round(time.perf_counter() - started, 6)


def run_transcription(job: Dict[str, Any], models: "ModelSet"):
    pipeline_started = time.perf_counter()
    audio_path = job["audio_path"]
    language = job.get("language") or DEFAULT_LANGUAGE
    do_diarize = job.get("diarize", True)
    min_spk = job.get("min_speakers")
    max_spk = job.get("max_speakers")
    num_spk = job.get("num_speakers")

    # No lock: each worker owns `models`, so its GPU work runs concurrently
    # with other workers but never touches another worker's models.
    job["stage"] = "load_audio"
    audio = _timed(job, "load_audio_s", whisperx.load_audio, audio_path)

    job["stage"] = "transcribe"
    asr = _timed(job, "prepare_asr_model_s", models.get_asr)
    result = _timed(
        job, "transcribe_s", asr.transcribe, audio,
        batch_size=BATCH_SIZE, language=language if language else None)
    lang = result.get("language", language or DEFAULT_LANGUAGE)

    job["stage"] = "align"
    model_a, metadata = _timed(
        job, "prepare_align_model_s", models.get_align, lang)
    result = _timed(
        job, "align_s", whisperx.align, result["segments"], model_a,
        metadata, audio, DEVICE, return_char_alignments=False)

    speaker_map = {}
    if do_diarize:
        job["stage"] = "diarize"
        diarize_model = _timed(
            job, "prepare_diarize_model_s", models.get_diarize)
        dia_kwargs = {}
        if num_spk:
            dia_kwargs["num_speakers"] = num_spk
        if min_spk:
            dia_kwargs["min_speakers"] = min_spk
        if max_spk:
            dia_kwargs["max_speakers"] = max_spk
        diarize_segments = _timed(
            job, "diarize_s", diarize_model, audio, **dia_kwargs)

        def assign_speakers():
            assigned = whisperx.assign_word_speakers(diarize_segments, result)
            mapping = _relabel_speakers(assigned["segments"])
            return assigned, mapping

        result, speaker_map = _timed(
            job, "assign_speakers_s", assign_speakers)
    else:
        job["timings"].update({
            "prepare_diarize_model_s": 0.0,
            "diarize_s": 0.0,
            "assign_speakers_s": 0.0,
        })
        for seg in result["segments"]:
            seg["speaker_label"] = "Speaker ?"

    job["stage"] = "format_outputs"

    def format_outputs():
        segments = result["segments"]
        full = {
            "language": lang,
            "num_speakers": len(speaker_map),
            "speaker_map": speaker_map,
            "segments": segments,
            "text": _to_txt(segments),
        }
        return full, _to_srt(segments)

    full, srt = _timed(job, "format_outputs_s", format_outputs)

    out_dir = OUT_DIR / job["job_id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_outputs():
        (out_dir / "transcript.txt").write_text(full["text"])
        (out_dir / "transcript.srt").write_text(srt)

    job["stage"] = "write_outputs"
    _timed(job, "write_outputs_s", write_outputs)

    job["result"] = full
    job["out_dir"] = str(out_dir)
    job["timings"]["pipeline_s"] = round(
        time.perf_counter() - pipeline_started, 6)


def _load_audio_16k(path: str) -> "Any":
    """Decode any ffmpeg-readable upload to mono 16 kHz float32 samples."""
    import numpy as np

    command = [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", path,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
        "-ar", "16000", "-",
    ]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def run_session_utterance(job: Dict[str, Any], models: "ModelSet"):
    """Transcribe with selected engines and identify one speaker within a session."""
    pipeline_started = time.perf_counter()
    audio_path = job["audio_path"]
    language = job.get("language") or None

    job["stage"] = "decode_audio"
    audio = _timed(job, "decode_audio_s", _load_audio_16k, audio_path)
    audio_seconds = len(audio) / 16000.0

    job["stage"] = "transcribe"
    transcription, engine_timings = _timed(
        job, "transcribe_s", run_sentence_models, audio,
        mode=job.get("asr_mode", SESSION_ASR_MODE), language=language,
        get_parakeet=models.get_sentence_asr, get_cohere=models.get_cohere)
    job["timings"].update(engine_timings)
    # Retain both hypotheses even if the primary engine reports no speech.
    job["result"] = {"transcription": transcription}
    primary = transcription["outputs"][transcription["primary"]]
    text, lang = primary["text"], primary["language"]
    if not text:
        raise ValueError("no speech was transcribed from this utterance")

    session = SESSION_STORE.get_session(job["session_id"])
    speaker_id = None
    similarity = None
    enrolled = False
    if session["capture_voiceprints"]:
        job["stage"] = "voiceprint"
        extractor = _timed(
            job, "prepare_voiceprint_model_s", models.get_voiceprint)
        embedding = _timed(
            job, "extract_voiceprint_s", extractor, audio_path,
            audio_seconds=audio_seconds)
        speaker_id, similarity, enrolled = _timed(
            job, "match_voiceprint_s", SESSION_STORE.match_voiceprint,
            job["session_id"], embedding, audio_seconds=audio_seconds,
            similarity_threshold=VOICEPRINT_SIMILARITY_THRESHOLD,
            update_threshold=VOICEPRINT_UPDATE_THRESHOLD,
        )
    else:
        job["timings"].update({
            "prepare_voiceprint_model_s": 0.0,
            "extract_voiceprint_s": 0.0,
            "match_voiceprint_s": 0.0,
        })

    job["stage"] = "store_utterance"
    utterance = _timed(
        job, "store_utterance_s", SESSION_STORE.append_utterance,
        job["session_id"], text=text, language=lang, speaker_id=speaker_id,
        speaker_similarity=similarity, audio_seconds=audio_seconds,
        source=job.get("source", "sentence"),
    )
    job["result"] = {
        **utterance,
        "transcription": transcription,
        "speaker": {
            "id": speaker_id,
            "similarity": similarity,
            "enrolled": enrolled,
            "signature_stored": speaker_id is not None,
        },
    }
    job["timings"]["pipeline_s"] = round(
        time.perf_counter() - pipeline_started, 6)


# ----------------------------------------------------------------------------
# Job queue: WORKERS parallel workers, simple in-memory registry + on-disk
# results. Each worker owns a private ModelSet and runs its blocking job in a
# dedicated thread, so up to WORKERS jobs overlap on the GPU.
# ----------------------------------------------------------------------------
WORKERS = max(1, int(os.environ.get("WORKERS", "2")))

JOBS: Dict[str, Dict[str, Any]] = {}
_queue: "asyncio.Queue[str]" = None  # set on startup
_executor: "ThreadPoolExecutor" = None  # set on startup

app = FastAPI(title="Parascribe API", version="2.0")


@app.get("/v2/test", include_in_schema=False)
async def browser_test_page():
    return FileResponse(Path(__file__).parent / "web" / "index.html")


@app.get("/v2/test/client.js", include_in_schema=False)
async def browser_test_client():
    return FileResponse(Path(__file__).parent / "web" / "client.js",
                        media_type="text/javascript")


@app.get("/v2/test/pcm-worklet.js", include_in_schema=False)
async def browser_test_worklet():
    return FileResponse(Path(__file__).parent / "web" / "pcm-worklet.js",
                        media_type="text/javascript")


@app.get("/v2/test/streaming.js", include_in_schema=False)
async def browser_test_streaming():
    return FileResponse(Path(__file__).parent / "web" / "streaming.js",
                        media_type="text/javascript")


def run_preview(item, models):
    """Ephemeral snapshot decode: no files, session writes, or voice extraction."""
    import numpy as np

    audio = np.frombuffer(item["pcm"], dtype="<i2").astype(np.float32) / 32768.0
    transcription, _ = run_sentence_models(
        audio, mode=item["asr_mode"], language=item["language"],
        get_parakeet=models.get_sentence_asr, get_cohere=models.get_cohere)
    return transcription


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timing_document(job: Dict[str, Any]) -> Dict[str, Any]:
    """Serializable timing record written for successful and failed jobs."""
    timestamps = {
        key: job[key]
        for key in ("received_at", "uploaded_at", "created_at", "queued_at",
                    "started_at", "finished_at")
        if job.get(key)
    }
    return {
        "schema_version": 1,
        "job_id": job["job_id"],
        "filename": job.get("filename"),
        "status": job.get("status"),
        "stage": job.get("stage"),
        "worker": job.get("worker"),
        "replica_id": REPLICA_ID,
        "timestamps": timestamps,
        "durations_s": dict(job.get("timings", {})),
    }


def _persist_job(job: Dict[str, Any]):
    """Persist final result and timing metadata in the job output directory."""
    out_dir = OUT_DIR / job["job_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    timing_document = _timing_document(job)
    (out_dir / "timings.json").write_text(
        json.dumps(timing_document, ensure_ascii=False, indent=2))
    if job.get("result"):
        result = dict(job["result"])
        result["job_id"] = job["job_id"]
        result["timestamps"] = timing_document["timestamps"]
        result["timings"] = timing_document["durations_s"]
        (out_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2))


async def _worker(worker_id: int, models: "ModelSet"):
    loop = asyncio.get_running_loop()
    while True:
        job_id = await _queue.get()
        if isinstance(job_id, dict):
            # Previews share each worker's model ownership; never run a second
            # thread against a final decoder or retain draft audio/results.
            item = job_id
            future = item["future"]
            try:
                if not future.cancelled():
                    result = await loop.run_in_executor(_executor, run_preview, item, models)
                    if not future.done():
                        future.set_result(result)
            except Exception as error:
                if not future.done():
                    future.set_exception(error)
            finally:
                item.pop("pcm", None)
                _queue.task_done()
            continue
        job = JOBS.get(job_id)
        if not job:
            _queue.task_done()
            continue
        worker_started = time.perf_counter()
        job["timings"]["queue_wait_s"] = round(
            worker_started - job["_queued_perf"], 6)
        job["status"] = "running"
        job["worker"] = worker_id
        job["started_at"] = _now()
        try:
            runner = job.get("_runner", run_transcription)
            await loop.run_in_executor(_executor, runner, job, models)
            job["status"] = "done"
            job["stage"] = "done"
        except Exception as e:
            job["status"] = "error"
            job["error"] = f"{e}\n{traceback.format_exc()}"
        finally:
            job["finished_at"] = _now()
            job["timings"]["processing_s"] = round(
                time.perf_counter() - worker_started, 6)
            job["timings"].setdefault(
                "pipeline_s", job["timings"]["processing_s"])
            persistence_started = time.perf_counter()
            try:
                _persist_job(job)
                job["timings"]["persist_metadata_s"] = round(
                    time.perf_counter() - persistence_started, 6)
                job["timings"]["total_server_s"] = round(
                    time.perf_counter() - job["_received_perf"], 6)
                # Refresh both JSON documents with the final persistence and
                # total durations. The first full write is what is measured.
                _persist_job(job)
            except Exception:
                job["timings"]["persist_metadata_s"] = round(
                    time.perf_counter() - persistence_started, 6)
                job["timings"]["total_server_s"] = round(
                    time.perf_counter() - job["_received_perf"], 6)
                persistence_error = traceback.format_exc()
                job["error"] = (job.get("error", "")
                                + "\nFailed to persist job metadata:\n"
                                + persistence_error).strip()
                job["status"] = "error"
            if job.get("_delete_audio"):
                try:
                    Path(job["audio_path"]).unlink(missing_ok=True)
                except Exception:
                    job["error"] = (job.get("error", "")
                                    + "\nFailed to delete temporary audio:\n"
                                    + traceback.format_exc()).strip()
                    job["status"] = "error"
            future = job.get("_future")
            if future is not None and not future.done():
                if job["status"] == "done":
                    future.set_result(job["result"])
                else:
                    future.set_exception(RuntimeError(job.get("error", "job failed")))
            _queue.task_done()


@app.on_event("startup")
async def _startup():
    global _queue, _executor
    _queue = asyncio.Queue()
    # One thread per worker so their blocking run_transcription calls run
    # concurrently rather than contending for a shared executor's threads.
    _executor = ThreadPoolExecutor(max_workers=WORKERS,
                                   thread_name_prefix="parascribe-worker")
    for i in range(WORKERS):
        asyncio.create_task(_worker(i, ModelSet(i)))


@app.get("/health")
async def health():
    return {"status": "ok", "model": WHISPER_MODEL, "device": DEVICE,
            "diarization": bool(HF_TOKEN), "workers": WORKERS,
            "session_asr": SESSION_ASR_MODEL,
            "session_asr_mode": SESSION_ASR_MODE,
            "cohere_configured": bool(COHERE_MODEL_PATH and Path(COHERE_MODEL_PATH).is_file()),
            "voiceprints": True, "replica_id": REPLICA_ID}


@app.get("/v1/models")
async def models():
    return {"whisper_model": WHISPER_MODEL, "device": DEVICE,
            "compute_type": COMPUTE_TYPE, "default_language": DEFAULT_LANGUAGE,
            "diarization_enabled": bool(HF_TOKEN), "workers": WORKERS,
            "session_asr_model": SESSION_ASR_MODEL,
            "session_asr_quantization": SESSION_ASR_QUANTIZATION,
            "session_asr_device": "cpu",
            "session_asr_modes": list(ASR_MODES),
            "session_asr_default_mode": SESSION_ASR_MODE,
            "cohere_model": COHERE_MODEL,
            "cohere_backend": COHERE_BACKEND,
            "voiceprint_model": VOICEPRINT_MODEL,
            "voiceprint_similarity_threshold": VOICEPRINT_SIMILARITY_THRESHOLD,
            "replica_id": REPLICA_ID}


def _public(job: Dict[str, Any]) -> Dict[str, Any]:
    fields = ["job_id", "status", "stage", "filename", "created_at",
              "received_at", "uploaded_at", "queued_at", "started_at",
              "finished_at", "error", "num_speakers", "worker", "session_id",
              "source"]
    out = {k: job.get(k) for k in fields if k in job}
    out["timings"] = dict(job.get("timings", {}))
    if job.get("result") and job.get("kind", "batch") == "batch":
        out["num_speakers"] = job["result"].get("num_speakers")
        out["language"] = job["result"].get("language")
        out["links"] = {
            "text": f"/v1/jobs/{job['job_id']}/transcript.txt",
            "srt": f"/v1/jobs/{job['job_id']}/transcript.srt",
            "json": f"/v1/jobs/{job['job_id']}/result.json",
        }
    elif job.get("result"):
        out["links"] = {"json": f"/v1/jobs/{job['job_id']}/result.json"}
    if job.get("finished_at"):
        out.setdefault("links", {})["timings"] = (
            f"/v1/jobs/{job['job_id']}/timings.json")
    return out


@app.post("/v1/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    diarize: bool = Form(True),
    num_speakers: Optional[int] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
):
    received_perf = time.perf_counter()
    received_at = _now()
    if diarize and not HF_TOKEN:
        raise HTTPException(400, "Diarization requested but HF_TOKEN is not set. "
                                 "Set it in .env and accept the pyannote license.")
    # Prefix the replica id so an nginx gateway can route follow-up
    # GET /v1/jobs/<job_id> requests back to this replica (see REPLICA_ID).
    job_id = f"{REPLICA_ID}-{uuid.uuid4().hex[:12]}"
    suffix = Path(file.filename or "audio").suffix or ".wav"
    audio_path = IN_DIR / f"{job_id}{suffix}"
    upload_save_started = time.perf_counter()
    try:
        with audio_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        upload_save_s = round(time.perf_counter() - upload_save_started, 6)
    uploaded_at = _now()
    queued_perf = time.perf_counter()
    queued_at = _now()

    job = {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "filename": file.filename,
        "audio_path": str(audio_path),
        "language": language,
        "diarize": diarize,
        "num_speakers": num_speakers,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "received_at": received_at,
        "uploaded_at": uploaded_at,
        "created_at": received_at,
        "queued_at": queued_at,
        "timings": {"upload_save_s": upload_save_s},
        "_received_perf": received_perf,
        "_queued_perf": queued_perf,
    }
    JOBS[job_id] = job
    await _queue.put(job_id)
    return JSONResponse(_public(job), status_code=202)


# ----------------------------------------------------------------------------
# V2: session-local sentence transcription and voice recognition
# ----------------------------------------------------------------------------
class CreateSessionRequest(BaseModel):
    capture_voiceprints: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EndSessionRequest(BaseModel):
    delete_voiceprints: bool = False


def _session_or_404(session_id: str, *, utterances: bool = False) -> dict:
    try:
        return SESSION_STORE.get_session(session_id, include_utterances=utterances)
    except SessionNotFound as error:
        raise HTTPException(404, "session not found") from error


def _session_error(error: Exception) -> HTTPException:
    if isinstance(error, SessionNotFound):
        return HTTPException(404, "session not found")
    if isinstance(error, SessionEnded):
        return HTTPException(409, "session has ended")
    return HTTPException(400, str(error))


def _new_session_job(
    session_id: str,
    audio_path: Path,
    *,
    filename: str,
    language: str | None,
    source: str,
    upload_save_s: float,
    asr_mode: str = SESSION_ASR_MODE,
    job_id: str | None = None,
    future=None,
) -> Dict[str, Any]:
    received_perf = time.perf_counter()
    now = _now()
    job_id = job_id or f"{REPLICA_ID}-{uuid.uuid4().hex[:12]}"
    job = {
        "job_id": job_id,
        "kind": "session_utterance",
        "asr_mode": asr_mode,
        "status": "queued",
        "stage": "queued",
        "filename": filename,
        "audio_path": str(audio_path),
        "language": language,
        "session_id": session_id,
        "source": source,
        "received_at": now,
        "uploaded_at": now,
        "created_at": now,
        "queued_at": now,
        "timings": {"upload_save_s": upload_save_s},
        "_received_perf": received_perf,
        "_queued_perf": time.perf_counter(),
        "_runner": run_session_utterance,
        "_delete_audio": True,
    }
    if future is not None:
        job["_future"] = future
    JOBS[job_id] = job
    return job


@app.post("/v2/sessions")
async def create_session(request: CreateSessionRequest):
    return SESSION_STORE.create_session(
        capture_voiceprints=request.capture_voiceprints,
        metadata=request.metadata,
    )


@app.get("/v2/sessions/{session_id}")
async def get_session(session_id: str):
    return _session_or_404(session_id, utterances=True)


@app.post("/v2/sessions/{session_id}/sentences")
async def transcribe_sentence(
    session_id: str,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    asr_mode: str = Form(SESSION_ASR_MODE),
):
    session = _session_or_404(session_id)
    if session["status"] != "active":
        raise HTTPException(409, "session has ended")
    _validate_session_asr(asr_mode, language)

    suffix = Path(file.filename or "utterance").suffix or ".wav"
    job_id = f"{REPLICA_ID}-{uuid.uuid4().hex[:12]}"
    audio_path = IN_DIR / f"{job_id}{suffix}"
    started = time.perf_counter()
    written = 0
    try:
        with audio_path.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > SENTENCE_MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "sentence audio is too large")
                target.write(chunk)
    except Exception:
        audio_path.unlink(missing_ok=True)
        raise
    job = _new_session_job(
        session_id, audio_path, filename=file.filename or "utterance",
        language=language, source="sentence",
        upload_save_s=round(time.perf_counter() - started, 6),
        job_id=job_id,
        asr_mode=asr_mode,
    )
    await _queue.put(job["job_id"])
    return JSONResponse(_public(job), status_code=202)


@app.delete("/v2/sessions/{session_id}/voiceprints")
async def delete_session_voiceprints(session_id: str):
    try:
        deleted = SESSION_STORE.delete_voiceprints(session_id)
    except (SessionNotFound, SessionEnded, ValueError) as error:
        raise _session_error(error) from error
    return {"session_id": session_id, "deleted_voiceprints": deleted,
            "capture_voiceprints": False}


@app.post("/v2/sessions/{session_id}/end")
async def end_session(session_id: str, request: EndSessionRequest):
    try:
        return SESSION_STORE.end_session(
            session_id, delete_voiceprints=request.delete_voiceprints)
    except (SessionNotFound, ValueError) as error:
        raise _session_error(error) from error


def _write_pcm16_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    if len(pcm) % 2:
        raise ValueError("PCM16 audio must contain complete 16-bit samples")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)


def _validate_session_asr(mode: str, language: str | None) -> None:
    try:
        validate_asr_request(mode, language)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    if mode in ("cohere", "compare"):
        if not COHERE_MODEL_PATH or not Path(COHERE_MODEL_PATH).is_file():
            raise HTTPException(503, "Cohere is not configured; set COHERE_MODEL_PATH to a local GGUF file")


@app.websocket("/v2/sessions/{session_id}/stream")
async def stream_session(ws: WebSocket, session_id: str):
    try:
        session = SESSION_STORE.get_session(session_id)
    except SessionNotFound:
        await ws.close(code=4404, reason="session not found")
        return
    if session["status"] != "active":
        await ws.close(code=4409, reason="session has ended")
        return

    await ws.accept()
    pending: asyncio.Queue = asyncio.Queue(maxsize=8)
    send_lock = asyncio.Lock()
    sample_rate = 16000
    audio = bytearray()
    started = False
    asr_mode = SESSION_ASR_MODE
    language = None
    partials = False
    generation = 0
    preview_task = None
    last_preview = 0.0

    async def send(message: dict):
        async with send_lock:
            await ws.send_json(message)

    async def preview(pcm, client_id, version):
        future = asyncio.get_running_loop().create_future()
        try:
            await _queue.put({"pcm": pcm, "asr_mode": asr_mode,
                              "language": language, "future": future})
            transcription = await future
            if version == generation:
                primary = transcription["outputs"][transcription["primary"]]
                await send({"type": "partial", "client_utterance_id": client_id,
                            "text": primary["text"], "transcription": transcription,
                            "speaker_status": "pending", "provisional": True})
        except asyncio.CancelledError:
            future.cancel()
            raise
        except Exception as error:
            if version == generation:
                await send({"type": "partial_error", "client_utterance_id": client_id,
                            "error": str(error)})

    def invalidate_preview():
        nonlocal generation
        generation += 1
        if preview_task and not preview_task.done():
            preview_task.cancel()

    async def process_utterances():
        while True:
            item = await pending.get()
            if item is None:
                pending.task_done()
                return
            pcm, client_utterance_id, item_mode, item_language = item
            job_id = f"{REPLICA_ID}-{uuid.uuid4().hex[:12]}"
            path = IN_DIR / f"{job_id}.wav"
            try:
                write_started = time.perf_counter()
                await asyncio.to_thread(_write_pcm16_wav, path, pcm, sample_rate)
                future = asyncio.get_running_loop().create_future()
                job = _new_session_job(
                    session_id, path, filename=f"{client_utterance_id}.wav",
                    language=item_language, source="realtime", asr_mode=item_mode,
                    upload_save_s=round(time.perf_counter() - write_started, 6),
                    job_id=job_id,
                    future=future,
                )
                await _queue.put(job["job_id"])
                await send({"type": "utterance_queued", "job_id": job["job_id"],
                            "client_utterance_id": client_utterance_id})
                result = await future
                await send({"type": "utterance", "job_id": job["job_id"],
                            "client_utterance_id": client_utterance_id,
                            "result": result})
            except Exception as error:  # keep the stream alive for the next sentence
                path.unlink(missing_ok=True)
                await send({"type": "utterance_error",
                            "client_utterance_id": client_utterance_id,
                            "error": str(error)})
            finally:
                pending.task_done()

    processor = asyncio.create_task(process_utterances())
    await send({"type": "ready", "session": session,
                "audio": {"format": "pcm_s16le", "sample_rate": sample_rate,
                          "channels": 1}})
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                if not started:
                    await send({"type": "error", "error": "send start before audio"})
                    continue
                if len(message["bytes"]) % 2:
                    await send({"type": "error", "error": "PCM16 requires complete samples"})
                    continue
                audio.extend(message["bytes"])
                max_bytes = int(STREAM_MAX_UTTERANCE_SECONDS * sample_rate * 2)
                if len(audio) > max_bytes:
                    audio.clear()
                    invalidate_preview()
                    await send({"type": "error", "error": "utterance is too long"})
                continue

            try:
                command = json.loads(message.get("text") or "{}")
            except json.JSONDecodeError:
                await send({"type": "error", "error": "invalid JSON command"})
                continue
            kind = command.get("type")
            if kind == "start":
                if command.get("format", "pcm_s16le") != "pcm_s16le":
                    await send({"type": "error", "error": "only pcm_s16le is supported"})
                    continue
                requested_rate = int(command.get("sample_rate", 16000))
                if requested_rate != 16000 or int(command.get("channels", 1)) != 1:
                    await send({"type": "error",
                                "error": "send mono PCM16 audio at 16000 Hz"})
                    continue
                requested_mode = command.get("asr_mode", SESSION_ASR_MODE)
                requested_language = command.get("language") or None
                try:
                    _validate_session_asr(requested_mode, requested_language)
                except HTTPException as error:
                    await send({"type": "error", "error": error.detail})
                    continue
                if started:
                    await send({"type": "error", "error": "stream already started"})
                    continue
                asr_mode, language = requested_mode, requested_language
                partials = command.get("partials") is True
                started = True
                audio.clear()
                await send({"type": "started", "asr_mode": asr_mode, "language": language})
            elif kind == "preview":
                # One in-flight preview per connection; discard excess requests
                # rather than building an ever-growing draft decode backlog.
                if not started or not partials:
                    await send({"type": "partial_skipped", "reason": "partials not enabled"})
                    continue
                now = time.monotonic()
                if (len(audio) < 16000 or now - last_preview < 1.0
                        or (preview_task and not preview_task.done()) or not pending.empty()):
                    await send({"type": "partial_skipped", "reason": "busy or insufficient audio"})
                    continue
                last_preview = now
                preview_task = asyncio.create_task(preview(
                    bytes(audio), str(command.get("utterance_id") or "draft"), generation))
            elif kind == "discard":
                audio.clear()
                invalidate_preview()
            elif kind == "commit":
                if not audio:
                    await send({"type": "error", "error": "no audio to commit"})
                    continue
                client_id = str(command.get("utterance_id") or uuid.uuid4().hex)
                invalidate_preview()
                await pending.put((bytes(audio), client_id, asr_mode, language))
                audio.clear()
            elif kind == "delete_voiceprints":
                deleted = await asyncio.to_thread(
                    SESSION_STORE.delete_voiceprints, session_id)
                await send({"type": "voiceprints_deleted", "count": deleted})
            elif kind == "end":
                invalidate_preview()
                if audio:
                    client_id = str(command.get("utterance_id") or uuid.uuid4().hex)
                    await pending.put((bytes(audio), client_id, asr_mode, language))
                    audio.clear()
                await pending.join()
                ended = await asyncio.to_thread(
                    SESSION_STORE.end_session, session_id,
                    delete_voiceprints=bool(command.get("delete_voiceprints", False)))
                await send({"type": "session_ended", "session": ended})
                break
            elif kind == "ping":
                await send({"type": "pong"})
            else:
                await send({"type": "error", "error": f"unknown command: {kind}"})
    except WebSocketDisconnect:
        pass
    finally:
        invalidate_preview()
        if preview_task:
            try:
                await preview_task
            except asyncio.CancelledError:
                pass
        processor.cancel()
        try:
            await processor
        except asyncio.CancelledError:
            pass


@app.get("/v1/jobs")
async def list_jobs():
    return {"jobs": [_public(j) for j in JOBS.values()]}


@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    out = _public(job)
    if job.get("status") == "done":
        out["result"] = job["result"]
    return out


def _read_out(job_id: str, name: str) -> str:
    path = OUT_DIR / job_id / name
    if not path.exists():
        raise HTTPException(404, f"{name} not ready (job not finished?)")
    return path.read_text()


@app.get("/v1/jobs/{job_id}/transcript.txt", response_class=PlainTextResponse)
async def get_txt(job_id: str):
    return _read_out(job_id, "transcript.txt")


@app.get("/v1/jobs/{job_id}/transcript.srt", response_class=PlainTextResponse)
async def get_srt(job_id: str):
    return _read_out(job_id, "transcript.srt")


@app.get("/v1/jobs/{job_id}/result.json")
async def get_json(job_id: str):
    return JSONResponse(json.loads(_read_out(job_id, "result.json")))


@app.get("/v1/jobs/{job_id}/timings.json")
async def get_timings(job_id: str):
    return JSONResponse(json.loads(_read_out(job_id, "timings.json")))
