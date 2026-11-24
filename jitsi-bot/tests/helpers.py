import math
import sys
from array import array
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOT_DIR))

FAKE_ORIGIN = "https://fake-meeting.test/"


def tone(seconds: float, amplitude: float = 0.2, frequency: float = 220.0) -> bytes:
    count = int(seconds * 16000)
    return array("h", (int(amplitude * 32767 * math.sin(2 * math.pi * frequency * i / 16000))
                       for i in range(count))).tobytes()


def silence(seconds: float) -> bytes:
    return b"\0\0" * int(seconds * 16000)


async def open_fake_meeting(browser):
    """A secure-origin blank page (AudioWorklet requires a secure context)."""
    context = await browser.new_context(bypass_csp=True)
    page = await context.new_page()
    await page.route(FAKE_ORIGIN + "**", lambda route: route.fulfill(
        status=200, content_type="text/html", body="<!doctype html><title>fake</title>"))
    await page.goto(FAKE_ORIGIN)
    return page


def fake_meeting_source() -> str:
    lines = (BOT_DIR / "tests/fake_meeting.js").read_text().splitlines()
    return "\n".join(line for line in lines if not line.startswith("//"))
