#!/usr/bin/env python3
"""
WhisperX transcription + speaker-diarization API.

Pipeline per request:  transcribe (Whisper large-v3) -> word-align -> diarize
(pyannote) -> assign words to speakers -> relabel SPEAKER_00 as "Speaker 1".

Jobs run on a single background worker (the GPU does one at a time). Results are
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
"""
import asyncio
import json
import os
import shutil
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

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

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
IN_DIR = DATA_DIR / "in"
OUT_DIR = DATA_DIR / "out"
for d in (IN_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

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


def run_transcription(job: Dict[str, Any], models: "ModelSet"):
    audio_path = job["audio_path"]
    language = job.get("language") or DEFAULT_LANGUAGE
    do_diarize = job.get("diarize", True)
    min_spk = job.get("min_speakers")
    max_spk = job.get("max_speakers")
    num_spk = job.get("num_speakers")

    # No lock: each worker owns `models`, so its GPU work runs concurrently
    # with other workers but never touches another worker's models.
    audio = whisperx.load_audio(audio_path)

    job["stage"] = "transcribe"
    asr = models.get_asr()
    result = asr.transcribe(audio, batch_size=BATCH_SIZE,
                            language=language if language else None)
    lang = result.get("language", language or DEFAULT_LANGUAGE)

    job["stage"] = "align"
    model_a, metadata = models.get_align(lang)
    result = whisperx.align(result["segments"], model_a, metadata, audio,
                            DEVICE, return_char_alignments=False)

    speaker_map = {}
    if do_diarize:
        job["stage"] = "diarize"
        diarize_model = models.get_diarize()
        dia_kwargs = {}
        if num_spk:
            dia_kwargs["num_speakers"] = num_spk
        if min_spk:
            dia_kwargs["min_speakers"] = min_spk
        if max_spk:
            dia_kwargs["max_speakers"] = max_spk
        diarize_segments = diarize_model(audio, **dia_kwargs)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        speaker_map = _relabel_speakers(result["segments"])
    else:
        for seg in result["segments"]:
            seg["speaker_label"] = "Speaker ?"

    segments = result["segments"]
    full = {
        "language": lang,
        "num_speakers": len(speaker_map),
        "speaker_map": speaker_map,
        "segments": segments,
        "text": _to_txt(segments),
    }

    out_dir = OUT_DIR / job["job_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(full, ensure_ascii=False, indent=2))
    (out_dir / "transcript.txt").write_text(full["text"])
    (out_dir / "transcript.srt").write_text(_to_srt(segments))

    job["result"] = full
    job["out_dir"] = str(out_dir)


# ----------------------------------------------------------------------------
# Job queue: WORKERS parallel workers, simple in-memory registry + on-disk
# results. Each worker owns a private ModelSet and runs its blocking job in a
# dedicated thread, so up to WORKERS jobs overlap on the GPU.
# ----------------------------------------------------------------------------
WORKERS = max(1, int(os.environ.get("WORKERS", "2")))

JOBS: Dict[str, Dict[str, Any]] = {}
_queue: "asyncio.Queue[str]" = None  # set on startup
_executor: "ThreadPoolExecutor" = None  # set on startup

app = FastAPI(title="WhisperX API", version="1.0")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _worker(worker_id: int, models: "ModelSet"):
    loop = asyncio.get_running_loop()
    while True:
        job_id = await _queue.get()
        job = JOBS.get(job_id)
        if not job:
            _queue.task_done()
            continue
        job["status"] = "running"
        job["worker"] = worker_id
        job["started_at"] = _now()
        try:
            await loop.run_in_executor(_executor, run_transcription, job, models)
            job["status"] = "done"
            job["stage"] = "done"
        except Exception as e:
            job["status"] = "error"
            job["error"] = f"{e}\n{traceback.format_exc()}"
        finally:
            job["finished_at"] = _now()
            _queue.task_done()


@app.on_event("startup")
async def _startup():
    global _queue, _executor
    _queue = asyncio.Queue()
    # One thread per worker so their blocking run_transcription calls run
    # concurrently rather than contending for a shared executor's threads.
    _executor = ThreadPoolExecutor(max_workers=WORKERS,
                                   thread_name_prefix="whisperx-worker")
    for i in range(WORKERS):
        asyncio.create_task(_worker(i, ModelSet(i)))


@app.get("/health")
async def health():
    return {"status": "ok", "model": WHISPER_MODEL, "device": DEVICE,
            "diarization": bool(HF_TOKEN), "workers": WORKERS,
            "replica_id": REPLICA_ID}


@app.get("/v1/models")
async def models():
    return {"whisper_model": WHISPER_MODEL, "device": DEVICE,
            "compute_type": COMPUTE_TYPE, "default_language": DEFAULT_LANGUAGE,
            "diarization_enabled": bool(HF_TOKEN), "workers": WORKERS,
            "replica_id": REPLICA_ID}


def _public(job: Dict[str, Any]) -> Dict[str, Any]:
    fields = ["job_id", "status", "stage", "filename", "created_at",
              "started_at", "finished_at", "error", "num_speakers", "worker"]
    out = {k: job.get(k) for k in fields if k in job}
    if job.get("result"):
        out["num_speakers"] = job["result"].get("num_speakers")
        out["language"] = job["result"].get("language")
        out["links"] = {
            "text": f"/v1/jobs/{job['job_id']}/transcript.txt",
            "srt": f"/v1/jobs/{job['job_id']}/transcript.srt",
            "json": f"/v1/jobs/{job['job_id']}/result.json",
        }
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
    if diarize and not HF_TOKEN:
        raise HTTPException(400, "Diarization requested but HF_TOKEN is not set. "
                                 "Set it in .env and accept the pyannote license.")
    # Prefix the replica id so an nginx gateway can route follow-up
    # GET /v1/jobs/<job_id> requests back to this replica (see REPLICA_ID).
    job_id = f"{REPLICA_ID}-{uuid.uuid4().hex[:12]}"
    suffix = Path(file.filename or "audio").suffix or ".wav"
    audio_path = IN_DIR / f"{job_id}{suffix}"
    with audio_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

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
        "created_at": _now(),
    }
    JOBS[job_id] = job
    await _queue.put(job_id)
    return JSONResponse(_public(job), status_code=202)


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
