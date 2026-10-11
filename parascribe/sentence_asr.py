"""Low-latency sentence transcription with Parakeet TDT v3 on ONNX Runtime."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np


DEFAULT_MODEL = "nemo-parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16000
MIN_SECONDS = 0.1
ASR_MODES = ("parakeet", "cohere", "compare")
COHERE_MODEL = "cohere-transcribe-03-2026"
COHERE_LANGUAGES = frozenset(
    "en de fr it es pt el nl pl vi zh ar ja ko".split())


def validate_asr_request(mode: str, language: str | None) -> None:
    if mode not in ASR_MODES:
        raise ValueError(f"asr_mode must be one of: {', '.join(ASR_MODES)}")
    if mode in ("cohere", "compare") and language not in COHERE_LANGUAGES:
        raise ValueError(
            "Cohere requires an explicit supported language: "
            + ", ".join(sorted(COHERE_LANGUAGES)))


class CohereSentenceTranscriber:
    """Local Cohere GGUF inference through the runtime used by Voikoro."""

    def __init__(self, *, model_path: str | Path, backend: str = "cpu", threads: int = 4):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Cohere GGUF model not found: {path}")
        import transcribe_cpp

        self.model = transcribe_cpp.Model(path, backend=backend)
        self.session = self.model.session(n_threads=max(1, threads))
        self.name = COHERE_MODEL
        self.quantization = "Q8_0" if "Q8_0" in path.name else "custom"
        self.device = self.model.device.kind

    def transcribe(self, audio: np.ndarray, *, language: str | None = None):
        validate_asr_request("cohere", language)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) < MIN_SECONDS * SAMPLE_RATE:
            return "", language
        return self.session.run(audio, language=language).text.strip(), language


def run_sentence_models(audio, *, mode, language, get_parakeet, get_cohere):
    """Run selected engines on identical samples; retain independently timed outputs.

    Compare mode keeps Parakeet as the session transcript and records Cohere as
    an alternative. Neither engine filters the other's input. Load time is
    separate from inference time; comparison runs sequentially, not concurrently.
    """
    validate_asr_request(mode, language)
    outputs = {}
    timings = {}
    engines = ("parakeet", "cohere") if mode == "compare" else (mode,)
    for engine in engines:
        started = time.perf_counter()
        model = get_parakeet() if engine == "parakeet" else get_cohere()
        timings[f"prepare_{engine}_model_s"] = round(time.perf_counter() - started, 6)
        started = time.perf_counter()
        text, lang = model.transcribe(audio, language=language)
        elapsed = time.perf_counter() - started
        seconds = len(audio) / SAMPLE_RATE
        timings[f"{engine}_transcribe_s"] = round(elapsed, 6)
        outputs[engine] = {
            "text": text, "language": lang, "model": model.name,
            "quantization": model.quantization,
            "device": getattr(model, "device", "cpu"),
            "inference_s": round(elapsed, 6),
            "real_time_factor": round(elapsed / seconds, 6) if seconds else None,
        }
    primary = "parakeet" if mode == "compare" else mode
    return {
        "mode": mode, "primary": primary, "outputs": outputs,
        "audio_seconds": len(audio) / SAMPLE_RATE,
    }, timings


def _word_probabilities(tokens: list[str] | None, logprobs: list[float] | None) -> list[float]:
    return [
        float(np.exp(logprob))
        for token, logprob in zip(tokens or [], logprobs or [], strict=False)
        if any(character.isalnum() for character in token)
    ]


class ParakeetSentenceTranscriber:
    """Parakeet v3 INT8, optimized for short utterances on the CPU.

    The model is multilingual and performs language selection internally.  It
    does not expose the detected language, so the API reports a client-forced
    language when present and ``und`` (undetermined) otherwise.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        model_dir: str | Path | None = None,
        quantization: str = "int8",
        threads: int = 4,
        min_confidence: float = 0.0,
    ):
        import onnx_asr
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, threads)
        options.log_severity_level = 3
        model = onnx_asr.load_model(
            model_name,
            Path(model_dir) if model_dir else None,
            quantization=quantization or None,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.model = model.with_timestamps()
        self.name = model_name
        self.quantization = quantization or "full"
        self.min_confidence = min_confidence

    def transcribe(self, audio: np.ndarray, *, language: str | None = None) -> tuple[str, str]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) < MIN_SECONDS * SAMPLE_RATE:
            return "", language or "und"
        result = self.model.recognize(audio, sample_rate=SAMPLE_RATE)
        text = result.text.strip()
        if self.min_confidence > 0:
            probabilities = _word_probabilities(result.tokens, result.logprobs)
            if not probabilities or float(np.mean(probabilities)) < self.min_confidence:
                text = ""
        return text, language or "und"
