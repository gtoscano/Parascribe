"""Energy-based sentence endpointing for one participant's audio.

A Python port of ``SentenceGate`` in ``parascribe/web/streaming.js`` so the bot
and the browser tester cut sentences the same way.  This is not a neural VAD:
it compares each frame's RMS against a threshold, keeps a short pre-roll so the
first syllable is not clipped, and never submits silence on its own.
"""
from __future__ import annotations

import math
from array import array
from dataclasses import dataclass

SAMPLE_RATE = 16000
PRE_ROLL_SAMPLES = 3200  # 200 ms kept before speech onset
MIN_VOICED_SAMPLES = 4000  # 250 ms of speech needed to commit rather than discard


@dataclass
class Frame:
    """Mono PCM16 little-endian audio and the wall-clock time it began."""
    pcm: bytes
    started: float

    @property
    def samples(self) -> int:
        return len(self.pcm) // 2


def rms(pcm: bytes) -> float:
    samples = array("h")
    samples.frombytes(pcm)
    if not samples:
        return 0.0
    return math.sqrt(sum(value * value for value in samples) / len(samples)) / 32768


class SentenceGate:
    def __init__(self, *, threshold: float = 0.015, pause: float = 0.8,
                 max_seconds: float = 20.0):
        self.threshold, self.pause, self.max_seconds = threshold, pause, max_seconds
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.pre: list[Frame] = []
        self.pre_samples = self.total = self.silence = self.voiced = 0

    def push(self, frame: Frame) -> tuple[list[Frame], str | None]:
        """Return frames to send now and ``"commit"``/``"discard"`` at a boundary."""
        count = frame.samples
        speech = rms(frame.pcm) >= self.threshold
        if not self.active:
            self.pre.append(frame)
            self.pre_samples += count
            while self.pre_samples > PRE_ROLL_SAMPLES + count:
                self.pre_samples -= self.pre.pop(0).samples
            if not speech:
                return [], None
            self.active = True
            frames, self.total = self.pre, self.pre_samples
            self.pre, self.pre_samples = [], 0
        else:
            frames = [frame]
            self.total += count
        self.voiced += count if speech else 0
        self.silence = 0 if speech else self.silence + count
        boundary = None
        if (self.silence >= self.pause * SAMPLE_RATE
                or self.total >= self.max_seconds * SAMPLE_RATE):
            boundary = self.finish()
        return frames, boundary

    def finish(self) -> str | None:
        """End the current utterance now (speaker change, track end, shutdown)."""
        if not self.active:
            self.reset()
            return None
        boundary = "commit" if self.voiced >= MIN_VOICED_SAMPLES else "discard"
        self.reset()
        return boundary
