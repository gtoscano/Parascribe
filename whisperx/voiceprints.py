"""Speaker-embedding extraction for session-local voice recognition."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


DEFAULT_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
MIN_VOICEPRINT_SECONDS = 0.5


class VoiceprintExtractor:
    """Lazily load pyannote's speaker embedding model and normalize its output."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        token: str | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.token = token
        self._inference = None

    def _load(self):
        from pyannote.audio import Inference, Model

        kwargs = {"use_auth_token": self.token} if self.token else {}
        try:
            model = Model.from_pretrained(self.model_name, **kwargs)
        except TypeError:
            # pyannote.audio 4 renamed use_auth_token to token.  Parascribe is
            # currently pinned to 3.x, but accepting both makes upgrades safe.
            kwargs = {"token": self.token} if self.token else {}
            model = Model.from_pretrained(self.model_name, **kwargs)
        if model is None:
            raise RuntimeError(
                f"Could not load speaker embedding model {self.model_name!r}; "
                "check the Hugging Face token and model access"
            )
        model.to(torch.device(self.device))
        self._inference = Inference(model, window="whole")
        return self._inference

    def __call__(self, audio_path: str | Path, *, audio_seconds: float) -> np.ndarray:
        if audio_seconds < MIN_VOICEPRINT_SECONDS:
            raise ValueError(
                f"at least {MIN_VOICEPRINT_SECONDS:.1f}s of speech is required "
                "to create a voiceprint"
            )
        inference = self._inference or self._load()
        embedding = np.asarray(inference(str(audio_path)), dtype=np.float32).reshape(-1)
        if not embedding.size or not np.all(np.isfinite(embedding)):
            raise RuntimeError("speaker embedding model returned an invalid voiceprint")
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-12:
            raise RuntimeError("speaker embedding model returned a zero voiceprint")
        return np.ascontiguousarray(embedding / norm, dtype=np.float32)
