"""End-to-end check against a running Parascribe, with a simulated meeting.

Real Chromium capture, real endpointing, real Parascribe transcription; only
Jitsi itself is simulated.  Two speakers from an Oyez recording each get their
own track, and the second speaker's turn overlaps the first speaker's next turn.

    python tests/fake_meeting_e2e.py --parascribe http://localhost:8357 \
        --wav ../testing_audio/dataset/oyez/2009.08-1107-t01.wav

Creates one Parascribe session (ended at exit, no voiceprints) and writes the
transcript under --out.
"""
import argparse
import asyncio
import base64
import json
import re
import sys
import wave
from pathlib import Path

from playwright.async_api import async_playwright

from helpers import fake_meeting_source, open_fake_meeting
from bot import Bot, build_parser


def clip(wav: wave.Wave_read, start: float, stop: float) -> bytes:
    wav.setpos(int(start * 16000))
    return wav.readframes(int((stop - start) * 16000))


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parascribe", default="http://localhost:8357")
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--out", default="/tmp/parascribe-e2e")
    options = parser.parse_args()

    turns = json.loads(options.wav.with_name(
        options.wav.name.replace(".wav", ".transcript.json")).read_text())["turns"]
    first, second = turns[0]["speaker"], turns[1]["speaker"]
    a1 = turns[0]
    b1 = turns[1]
    a2 = next(t for t in turns[2:] if t["speaker"] == first)
    with wave.open(str(options.wav)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        b_stop = min(b1["stop"], b1["start"] + 25)
        a2_stop = min(a2["stop"], a2["start"] + 10)
        pcm_a = clip(wav, a1["start"], a1["stop"]) + b"\0\0" * int(16000 * 18) \
            + clip(wav, a2["start"], a2_stop)
        pcm_b = clip(wav, b1["start"], b_stop)
    a1_len = a1["stop"] - a1["start"]
    # B starts after A's first turn; A's second turn begins 18 s later, overlapping B.
    duration = 1 + a1_len + 18 + (a2_stop - a2["start"])

    args = build_parser().parse_args([
        "e2e-fake-room", "--parascribe", options.parascribe, "--out", options.out,
        "--language", "en", "--announce", "test announcement"])
    bot = Bot(args)
    await bot.start_session()
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            channel="chromium", args=["--autoplay-policy=no-user-gesture-required"])
        page = await open_fake_meeting(browser)
        await bot.attach(page)
        b64 = lambda pcm: base64.b64encode(pcm).decode()
        await page.evaluate(fake_meeting_source(), {
            "participants": [{"id": "pa", "name": first}, {"id": "pb", "name": second}],
            "tracks": [
                {"pcm": b64(pcm_a), "startAt": 1.0, "owners": [[0, "pa"]]},
                {"pcm": b64(pcm_b), "startAt": 1.0 + a1_len + 0.5, "owners": [[0, "pb"]]},
            ],
            "addAfter": 0.2, "leaveAfter": duration + 2,
        })
        try:
            await bot.transcribe(page)
            chat = await page.evaluate("() => window.fakeChat || []")
        finally:
            await bot.shutdown(page)
            await browser.close()

    print("\n" + bot.transcript.path.read_text())
    by_speaker: dict[str, str] = {}
    for entry in bot.transcript.ordered():
        by_speaker[entry["speaker"]] = by_speaker.get(entry["speaker"], "") + " " + entry["text"]
    reference = {first: a1["text"] + " " + a2["text"], second: b1["text"]}
    ok = chat == ["test announcement"] and set(by_speaker) == {first, second}
    for speaker, text in by_speaker.items():
        heard = words(text)
        # Speaker B was cut at 25 s, A's second turn at 10 s: compare overlap
        # against what each person said, not exact WER.
        precision = len(heard & words(reference.get(speaker, ""))) / max(1, len(heard))
        print(f"{speaker}: {len(heard)} distinct words, {precision:.0%} found in their own turns")
        ok &= precision >= 0.8
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
