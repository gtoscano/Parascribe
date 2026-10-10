"""Low-latency sentence transcription with Parakeet TDT v3 on ONNX Runtime."""
from __future__ import annotations

from pathlib import Path

import numpy as np


DEFAULT_MODEL = "nemo-parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16000
MIN_SECONDS = 0.1


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
