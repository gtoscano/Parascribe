"""Parascribe v2 connections: one WebSocket per Jitsi audio track.

Jitsi delivers every participant's microphone as a separate track, so each
utterance has exactly one speaker and overlapping speech never mixes.  On
meet.jit.si a received track can be reassigned to another participant (SSRC
rewriting forwards only the most active speakers), so the owner is read for
every chunk and a change of owner ends the current utterance.
"""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Awaitable, Callable

import websockets

from endpointing import SAMPLE_RATE, Frame, SentenceGate

log = logging.getLogger("parascribe.jitsi")

FRAME_SAMPLES = 512  # matches the browser tester's worklet buffer (32 ms)
# Sentences Parascribe may hold per connection before it stops reading.
MAX_OUTSTANDING = 8


def iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


class ParascribeClient:
    """Minimal REST client for the Parascribe v2 session API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    @property
    def ws_base(self) -> str:
        return "ws" + self.base_url[len("http"):] if self.base_url.startswith("http") else self.base_url

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        request = urllib.request.Request(
            self.base_url + path, method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"content-type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    async def health(self) -> dict:
        return await asyncio.to_thread(self._request, "GET", "/health")

    async def create_session(self, *, capture_voiceprints: bool, metadata: dict) -> dict:
        return await asyncio.to_thread(self._request, "POST", "/v2/sessions", {
            "capture_voiceprints": capture_voiceprints, "metadata": metadata})

    async def end_session(self, session_id: str, *, delete_voiceprints: bool = True) -> dict:
        return await asyncio.to_thread(self._request, "POST", f"/v2/sessions/{session_id}/end",
                                       {"delete_voiceprints": delete_voiceprints})


ResultHandler = Callable[[dict], Awaitable[None] | None]


class TrackStream:
    """Endpoint one remote audio track and transcribe it through Parascribe."""

    def __init__(self, key: str, *, ws_url: str, names: dict[str, str],
                 on_result: ResultHandler, asr_mode: str = "parakeet",
                 language: str | None = None, gate_options: dict | None = None):
        self.key, self.ws_url, self.names = key, ws_url, names
        self.on_result, self.asr_mode, self.language = on_result, asr_mode, language
        self.gate = SentenceGate(**(gate_options or {}))
        self.socket = None
        self.reader: asyncio.Task | None = None
        self.outstanding: dict[str, dict] = {}
        self.completed = asyncio.Event()  # set whenever a sentence leaves `outstanding`
        self.owner: str | None = None
        self.utterance: dict | None = None  # {id, owner, started}
        self.buffer = b""
        self.buffer_started = 0.0
        self.lock = asyncio.Lock()

    def label(self, owner: str | None) -> str:
        return self.names.get(owner or "", "") or f"Participant {owner or '?'}"

    async def feed(self, owner: str | None, pcm: bytes, arrived: float) -> None:
        """Accept a chunk of 16 kHz PCM16 that finished arriving at ``arrived``."""
        async with self.lock:
            if owner != self.owner:
                # Different person on this track: close their sentence first.
                await self._boundary(self.gate.finish())
                self.buffer = b""
                self.owner = owner
            if owner is None:
                return  # unassigned track: nobody to attribute this audio to
            # Re-anchor on every chunk: tracks deliver nothing while muted, so a
            # running sample clock would drift across those gaps.
            self.buffer_started = arrived - (len(self.buffer) + len(pcm)) / 2 / SAMPLE_RATE
            self.buffer += pcm
            step = FRAME_SAMPLES * 2
            while len(self.buffer) >= step:
                chunk, self.buffer = self.buffer[:step], self.buffer[step:]
                frame = Frame(chunk, self.buffer_started)
                self.buffer_started += FRAME_SAMPLES / SAMPLE_RATE
                frames, boundary = self.gate.push(frame)
                if frames:
                    await self._send_audio(frames)
                await self._boundary(boundary)

    async def idle(self) -> None:
        """No audio is arriving (e.g. the speaker muted): end any open sentence."""
        async with self.lock:
            await self._boundary(self.gate.finish())

    async def _connect(self) -> None:
        if self.socket is not None:
            return
        self.socket = await websockets.connect(self.ws_url, max_size=None)
        ready = json.loads(await self.socket.recv())
        if ready.get("type") != "ready":
            raise RuntimeError(f"Parascribe stream refused: {ready}")
        start = {"type": "start", "format": "pcm_s16le", "sample_rate": SAMPLE_RATE,
                 "channels": 1, "asr_mode": self.asr_mode}
        if self.language:
            start["language"] = self.language
        await self.socket.send(json.dumps(start))
        started = json.loads(await self.socket.recv())
        if started.get("type") != "started":
            raise RuntimeError(f"Parascribe stream did not start: {started}")
        self.reader = asyncio.create_task(self._read(self.socket))

    async def _send_audio(self, frames: list[Frame]) -> None:
        if self.utterance is None:
            self.utterance = {"id": f"{self.key}-{uuid.uuid4().hex[:8]}",
                              "owner": self.owner, "started": frames[0].started}
        try:
            await self._connect()
            for frame in frames:
                await self.socket.send(frame.pcm)
        except (OSError, websockets.ConnectionClosed, RuntimeError) as error:
            log.warning("track %s: lost audio (%s); will reconnect", self.key, error)
            await self._drop_socket()
            self.utterance = None
            self.gate.reset()

    async def _boundary(self, boundary: str | None) -> None:
        if boundary is None or self.utterance is None or self.socket is None:
            if boundary is not None:
                self.utterance = None
            return
        utterance, self.utterance = self.utterance, None
        try:
            if boundary == "discard":
                await self.socket.send(json.dumps({"type": "discard"}))
                return
            await self._wait_until(lambda: len(self.outstanding) < MAX_OUTSTANDING)
            meta = {"speaker_label": self.label(utterance["owner"]),
                    "participant_id": utterance["owner"],
                    "spoken_at": iso(utterance["started"]), "track": self.key}
            self.outstanding[utterance["id"]] = meta
            await self.socket.send(json.dumps({
                "type": "commit", "utterance_id": utterance["id"],
                "speaker_label": meta["speaker_label"], "spoken_at": meta["spoken_at"]}))
        except websockets.ConnectionClosed as error:
            log.warning("track %s: connection closed at commit (%s)", self.key, error)
            await self._drop_socket()

    async def _read(self, socket) -> None:
        try:
            async for raw in socket:
                message = json.loads(raw)
                kind = message.get("type")
                client_id = message.get("client_utterance_id")
                if kind in ("utterance", "no_speech", "utterance_error"):
                    meta = self.outstanding.pop(client_id, {})
                    if kind == "utterance":
                        result = await self._deliver(message["result"], meta)
                        log.debug("track %s: %s", self.key, result.get("text"))
                    elif kind == "utterance_error":
                        log.warning("track %s: utterance failed: %s",
                                    self.key, message.get("error"))
                    self.completed.set()
                elif kind == "error":
                    log.warning("track %s: Parascribe error: %s", self.key, message.get("error"))
        except websockets.ConnectionClosed:
            pass
        finally:
            if self.socket is socket:
                self._fail_outstanding()

    async def _deliver(self, result: dict, meta: dict) -> dict:
        entry = {**result, "participant_id": meta.get("participant_id"),
                 "track": self.key}
        entry.setdefault("speaker_label", meta.get("speaker_label"))
        entry.setdefault("spoken_at", meta.get("spoken_at"))
        outcome = self.on_result(entry)
        if asyncio.iscoroutine(outcome):
            await outcome
        return entry

    def _fail_outstanding(self) -> None:
        if self.outstanding:
            log.warning("track %s: %d sentence(s) lost with the connection",
                        self.key, len(self.outstanding))
        self.outstanding.clear()
        self.completed.set()

    async def _drop_socket(self) -> None:
        socket, self.socket = self.socket, None
        if socket is not None:
            await socket.close()
        self._fail_outstanding()

    async def close(self, timeout: float = 120.0) -> None:
        """Commit any sentence in progress, wait for results, and disconnect."""
        async with self.lock:
            await self._boundary(self.gate.finish())
        if self.socket is None:
            return
        try:
            await asyncio.wait_for(self._wait_until(lambda: not self.outstanding), timeout)
        except asyncio.TimeoutError:
            log.warning("track %s: gave up waiting for %d sentence(s)",
                        self.key, len(self.outstanding))
        socket, self.socket = self.socket, None
        await socket.close()
        if self.reader:
            await asyncio.gather(self.reader, return_exceptions=True)

    async def _wait_until(self, condition) -> None:
        while not condition():
            self.completed.clear()
            await self.completed.wait()
