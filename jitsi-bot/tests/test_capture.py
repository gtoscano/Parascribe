"""capture.js in real Chromium, inside a simulated Jitsi room (no network)."""
import asyncio
import base64
import unittest
from collections import defaultdict

from playwright.async_api import async_playwright

from bot import capture_source
from endpointing import rms
from helpers import fake_meeting_source, open_fake_meeting, tone


class CaptureTest(unittest.IsolatedAsyncioTestCase):
    async def test_tracks_are_captured_per_owner_at_16khz(self):
        events = []
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                channel="chromium", args=["--autoplay-policy=no-user-gesture-required"])
            try:
                page = await open_fake_meeting(browser)
                await page.expose_function("parascribeEmit", lambda event: events.append(event))
                b64 = lambda pcm: base64.b64encode(pcm).decode()
                await page.evaluate(fake_meeting_source(), {
                    "participants": [{"id": "p1", "name": "Ana"}, {"id": "p2", "name": "Luis"},
                                     {"id": "p3", "name": "Marta"}],
                    "tracks": [
                        {"pcm": b64(tone(1.5)), "startAt": 1.0, "owners": [[0, "p1"]]},
                        # Reassigned from Luis to Marta mid-stream (SSRC rewriting).
                        {"pcm": b64(tone(3.0, frequency=330)), "startAt": 1.0,
                         "owners": [[0, "p2"], [2.5, "p3"]]},
                    ],
                    "addAfter": 0.3, "leaveAfter": 5.0,
                })
                info = await page.evaluate(capture_source(), {"flushMs": 200})
                for _ in range(80):
                    if any(e["type"] == "conference_left" for e in events):
                        break
                    await asyncio.sleep(0.1)
                await page.evaluate("() => window.parascribeStop()")
                await asyncio.sleep(0.3)
            finally:
                await browser.close()

        self.assertEqual(info["tracks"], 0)  # tracks arrived later, via the event
        names = {e["id"]: e["name"] for e in events if e["type"] == "participant"}
        self.assertEqual(names, {"p1": "Ana", "p2": "Luis", "p3": "Marta"})

        audio = defaultdict(bytes)
        owners = defaultdict(list)
        for event in (e for e in events if e["type"] == "audio"):
            audio[(event["key"], event["owner"])] += base64.b64decode(event["pcm"])
            if not owners[event["key"]] or owners[event["key"]][-1] != event["owner"]:
                owners[event["key"]].append(event["owner"])
        self.assertEqual(dict(owners), {"t1": ["p1"], "t2": ["p2", "p3"]})

        def tone_seconds(pcm):  # 10 ms windows that carry the tone
            return sum(rms(pcm[i:i + 320]) > 0.05 for i in range(0, len(pcm), 320)) / 100

        # 16 kHz output from a 44.1/48 kHz browser: tone durations are preserved.
        self.assertAlmostEqual(tone_seconds(audio[("t1", "p1")]), 1.5, delta=0.1)
        self.assertAlmostEqual(tone_seconds(audio[("t2", "p2")])
                               + tone_seconds(audio[("t2", "p3")]), 3.0, delta=0.15)
        self.assertGreater(tone_seconds(audio[("t2", "p2")]), 1.0)
        self.assertGreater(tone_seconds(audio[("t2", "p3")]), 1.0)
        ended = {e["key"] for e in events if e["type"] == "track_ended"}
        self.assertEqual(ended, {"t1", "t2"})


if __name__ == "__main__":
    unittest.main()
