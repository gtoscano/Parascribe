"""Parascribe Jitsi transcriber bot.

Joins a Jitsi Meet room as a visible participant, announces itself in the chat,
and transcribes each participant's microphone separately through a Parascribe
v2 session.  The transcript is written live to ``--out`` and kept in the
Parascribe session.

    python bot.py my-room --parascribe http://localhost:8357 --language es

On meet.jit.si the bot waits until a host has signed in and started the
meeting; it cannot start a meeting by itself.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import re
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlparse

from playwright.async_api import async_playwright

from streams import ParascribeClient, TrackStream
from transcript import Transcript

log = logging.getLogger("parascribe.jitsi")
HERE = Path(__file__).resolve().parent

DEFAULT_ANNOUNCEMENT = (
    "Parascribe is transcribing this meeting. Your display name will appear "
    "next to what you say. / Parascribe está transcribiendo esta reunión. Su "
    "nombre aparecerá junto a lo que diga."
)


def parse_room(room: str, server: str) -> tuple[str, str]:
    """Accept a room name or a full meeting link; return (server, room)."""
    if "://" in room:
        link = urlparse(room)
        return f"{link.scheme}://{link.netloc}", link.path.strip("/")
    return server.rstrip("/"), room.strip("/")


def meeting_url(server: str, room: str, name: str) -> str:
    # Jitsi Meet reads JSON-encoded config overrides from the URL fragment.
    overrides = {
        "config.prejoinConfig.enabled": False,
        "config.startWithAudioMuted": True,
        "config.startWithVideoMuted": True,
        "config.disableDeepLinking": True,
        "config.p2p.enabled": False,  # keep all audio on the bridge, one code path
        "config.channelLastN": 0,  # receive no video; audio is unaffected
        "userInfo.displayName": name,
    }
    fragment = "&".join(f"{key}={quote(json.dumps(value))}" for key, value in overrides.items())
    return f"{server}/{quote(room)}#{fragment}"


def capture_source() -> str:
    # Playwright needs the text to start with the function, not comments.
    lines = (HERE / "capture.js").read_text().splitlines()
    return "\n".join(line for line in lines if not line.startswith("//"))


class Bot:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.server, self.room = parse_room(args.room, args.server)
        self.client = ParascribeClient(args.parascribe)
        self.names: dict[str, str] = {}
        self.queues: dict[str, asyncio.Queue] = {}
        self.consumers: list[asyncio.Task] = []
        self.stop = asyncio.Event()
        self.transcript: Transcript | None = None
        self.session_id: str | None = None

    # ---- page events (called synchronously from Playwright's binding) ----
    def on_event(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "participant":
            if event["name"]:
                self.names[event["id"]] = event["name"]
        elif kind == "audio":
            queue = self.queues.get(event["key"])
            if queue is None:
                queue = self.queues[event["key"]] = asyncio.Queue()
                self.consumers.append(asyncio.create_task(self.consume(event["key"], queue)))
            queue.put_nowait((event["owner"], base64.b64decode(event["pcm"]), event["at"]))
        elif kind == "track_ended":
            queue = self.queues.pop(event["key"], None)
            if queue is not None:
                queue.put_nowait(None)
        elif kind == "conference_left":
            log.info("left the conference (%s)", event.get("reason"))
            self.stop.set()

    async def consume(self, key: str, queue: asyncio.Queue) -> None:
        stream = TrackStream(
            key, ws_url=f"{self.client.ws_base}/v2/sessions/{self.session_id}/stream",
            names=self.names, on_result=self.transcript.add, asr_mode=self.args.asr_mode,
            language=self.args.language,
            gate_options={"threshold": self.args.threshold, "pause": self.args.pause,
                          "max_seconds": self.args.max_seconds})
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), self.args.pause)
                except asyncio.TimeoutError:
                    await stream.idle()  # muted tracks stop delivering audio
                    continue
                if item is None:
                    break
                await stream.feed(*item)
        except Exception:
            log.exception("track %s failed", key)
        finally:
            await stream.close()

    # ---- lifecycle ----
    async def start_session(self) -> None:
        health = await self.client.health()
        log.info("Parascribe ready (%s, replica %s)", health.get("session_asr"),
                 health.get("replica_id"))
        session = await self.client.create_session(
            capture_voiceprints=self.args.voiceprints,
            metadata={"source": "jitsi", "server": self.server, "room": self.room})
        self.session_id = session["session_id"]
        started = datetime.now(timezone.utc)
        safe_room = re.sub(r"[^A-Za-z0-9._-]+", "_", self.room) or "room"
        path = Path(self.args.out) / f"{safe_room}-{started:%Y%m%dT%H%M%SZ}.md"
        self.transcript = Transcript(path, room=self.room, session_id=self.session_id,
                                     started=started)
        log.info("Parascribe session %s; transcript %s", self.session_id, path)

    async def attach(self, page) -> None:
        # Playwright tags the callback, so it must be a plain function.
        await page.expose_function("parascribeEmit", lambda event: self.on_event(event))
        page.on("close", lambda _: self.stop.set())

    async def run(self) -> int:
        await self.start_session()
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(signum, self.stop.set)

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                channel="chromium", headless=not self.args.headed, args=[
                    "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                    "--autoplay-policy=no-user-gesture-required"])
            page = None
            try:
                context = await browser.new_context(
                    permissions=["microphone", "camera"], bypass_csp=True)
                page = await context.new_page()
                await self.attach(page)
                await page.goto(meeting_url(self.server, self.room, self.args.name))
                if await self.wait_until_joined(page):
                    await self.transcribe(page)
            finally:
                await self.shutdown(page)
                await browser.close()
        return 0

    async def wait_until_joined(self, page) -> bool:
        deadline = time.monotonic() + self.args.join_timeout
        told = False
        while not self.stop.is_set() and time.monotonic() < deadline:
            state = await page.evaluate("""() => ({
                joined: !!window.APP?.conference?._room?.isJoined?.(),
                waiting: /no moderators have yet arrived|waiting for the host/i
                         .test(document.body?.innerText || "")})""")
            if state["joined"]:
                log.info("joined %s/%s", self.server, self.room)
                return True
            if state["waiting"] and not told:
                log.info("waiting for the meeting host to sign in and start the meeting")
                told = True
            await asyncio.sleep(2)
        if not self.stop.is_set():
            log.error("did not join within %ss", self.args.join_timeout)
        return False

    async def transcribe(self, page) -> None:
        await page.evaluate("() => APP.conference.muteAudio(true)")
        if self.args.announce:
            await page.evaluate("text => APP.conference._room.sendTextMessage(text)",
                                self.args.announce)
        info = await page.evaluate(capture_source(), {"flushMs": 200})
        log.info("capturing audio (%s track(s) at start)", info["tracks"])
        empty_since = None
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(self.stop.wait(), 5)
                break
            except asyncio.TimeoutError:
                pass
            try:
                others = await page.evaluate(
                    "() => APP.conference._room?.isJoined() ? "
                    "APP.conference._room.getParticipants().length : -1")
            except Exception as error:  # page crashed or navigated away
                log.error("meeting page unavailable: %s", error)
                break
            if others < 0:
                log.info("no longer in the meeting")
                break
            empty_since = (empty_since or time.monotonic()) if others == 0 else None
            if empty_since and time.monotonic() - empty_since > self.args.empty_timeout:
                log.info("room empty for %ss; leaving", self.args.empty_timeout)
                break

    async def shutdown(self, page) -> None:
        log.info("finishing: committing open sentences and waiting for results")
        if page is not None and not page.is_closed():
            try:
                await page.evaluate("() => window.parascribeStop?.()")
                await asyncio.sleep(0.5)  # let the final binding calls arrive
            except Exception:
                pass
        for queue in self.queues.values():
            queue.put_nowait(None)
        self.queues.clear()
        if self.consumers:
            await asyncio.gather(*self.consumers, return_exceptions=True)
        if self.session_id and not self.args.keep_session_open:
            try:
                await self.client.end_session(self.session_id, delete_voiceprints=True)
            except Exception as error:
                log.warning("could not end Parascribe session: %s", error)
        if self.transcript:
            self.transcript.write()
            log.info("transcript saved: %s (%d sentences)", self.transcript.path,
                     len(self.transcript.entries))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("room", help="room name or full meeting link")
    parser.add_argument("--server", default="https://meet.jit.si",
                        help="Jitsi Meet server for a bare room name")
    parser.add_argument("--parascribe", default=os.environ.get("PARASCRIBE_URL", "http://localhost:8357"))
    parser.add_argument("--name", default="Parascribe (transcriber)", help="bot display name")
    parser.add_argument("--language", help="e.g. es or en; Parakeet detects it when omitted")
    parser.add_argument("--asr-mode", default="parakeet", choices=["parakeet", "cohere", "compare"])
    parser.add_argument("--announce", default=DEFAULT_ANNOUNCEMENT,
                        help="chat message sent on joining; empty string to skip")
    parser.add_argument("--voiceprints", action="store_true",
                        help="also capture session voiceprints (off: Jitsi names identify speakers)")
    parser.add_argument("--out", default="transcripts", help="directory for transcript files")
    parser.add_argument("--join-timeout", type=float, default=1800,
                        help="seconds to wait for the host to start the meeting")
    parser.add_argument("--empty-timeout", type=float, default=300,
                        help="leave after the room has had no one else for this long")
    parser.add_argument("--threshold", type=float, default=0.015, help="speech RMS threshold")
    parser.add_argument("--pause", type=float, default=0.8, help="seconds of silence ending a sentence")
    parser.add_argument("--max-seconds", type=float, default=20, help="longest single sentence")
    parser.add_argument("--keep-session-open", action="store_true",
                        help="do not end the Parascribe session on exit")
    parser.add_argument("--headed", action="store_true", help="show the browser (debugging)")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("websockets").setLevel(logging.INFO)
    return asyncio.run(Bot(args).run())


if __name__ == "__main__":
    raise SystemExit(main())
