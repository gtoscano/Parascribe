"""Optional real-browser integration smoke test (requires Playwright/Chromium).

Run against a disposable local v2 server configured with both models:
  python tests/browser_smoke.py --url http://localhost:18357 --audio /tmp/speech.wav

Audio must contain clear single-speaker English speech. The microphone is
Chromium's fake device, fed the same WAV; this test never opens a physical mic.
No voice signatures are captured. Sessions created by the test are ended.
"""
import argparse
import json
import re
from pathlib import Path

from playwright.sync_api import sync_playwright, expect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:18357")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={args.audio.resolve()}",
        ])
        context = browser.new_context(permissions=["microphone"])
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url.rstrip("/") + "/v2/test")
        expect(page.locator("#status")).to_contain_text("Server ready", timeout=60000)
        worklet = context.request.get(args.url.rstrip("/") + "/v2/test/pcm-worklet.js").text()
        conversions = page.evaluate("""source => {
          const counts = [];
          for (const rate of [16000, 44100, 48000]) {
            let Capture;
            const messages = [];
            class Base { constructor() { this.port = {postMessage: data => messages.push(data)}; } }
            new Function('AudioWorkletProcessor', 'sampleRate', 'registerProcessor', source)(
              Base, rate, (name, implementation) => { Capture = implementation; });
            const capture = new Capture();
            for (let offset = 0; offset < rate; offset += 128) {
              const length = Math.min(128, rate - offset);
              capture.process([[new Float32Array(length).fill(.5), new Float32Array(length)]]);
            }
            capture.port.onmessage({data: {type: 'flush', id: 'test', pause: true}});
            let samples = 0;
            for (const message of messages.filter(item => item.type === 'audio')) {
              const view = new DataView(message.buffer);
              samples += view.byteLength / 2;
              for (let index = 0; index < view.byteLength; index += 2) {
                if (view.getInt16(index, true) !== 8192) throw Error('Incorrect PCM mix/encoding');
              }
            }
            if (messages.at(-1).type !== 'flushed') throw Error('Flush acknowledgment order');
            counts.push(samples);
          }
          return counts;
        }""", worklet)
        assert conversions == [16000, 16000, 16000], conversions
        # Endpointing must ignore silence, preserve pre-roll, and finalize once.
        boundaries = page.evaluate("""async () => {
          const {SentenceGate} = await import('/v2/test/streaming.js');
          const frame = value => new Int16Array(512).fill(value).buffer;
          const gate = new SentenceGate();
          for (let i = 0; i < 100; i++) {
            if (gate.push(frame(0)).frames.length) throw Error('Silence submitted');
          }
          let sent = 0, commits = 0;
          for (let i = 0; i < 32; i++) sent += gate.push(frame(6000)).frames.length;
          for (let i = 0; i < 40; i++) {
            const result = gate.push(frame(0));
            if (result.boundary === 'commit') commits++;
          }
          if (sent < 32 || commits !== 1) throw Error('Endpointing failed');
          gate.push(frame(6000));
          let discarded = false;
          for (let i = 0; i < 40; i++) discarded ||= gate.push(frame(0)).boundary === 'discard';
          if (!discarded) throw Error('Click noise was not discarded');
          return commits;
        }""")
        assert boundaries == 1
        page.locator("#capture").uncheck()
        page.locator("#create").click()
        expect(page.locator("#sessionInfo")).to_contain_text("active")
        session_id = page.locator("#sessionId").input_value()
        page.locator("#mode").select_option("compare")
        page.locator("#language").select_option("en")
        page.locator("#file").set_input_files(args.audio)
        page.locator("#upload").click()
        page.locator("#finalTab").click()
        expect(page.locator("#results article")).to_have_count(1, timeout=120000)
        expect(page.locator("#results article").first).to_contain_text("cohere")
        expect(page.locator("#results article").first).to_contain_text("parakeet")
        page.locator("#start").click()
        expect(page.locator("#status")).to_contain_text("Microphone on", timeout=20000)
        page.wait_for_timeout(3200)
        page.locator("#commit").click()
        expect(page.locator("#results article")).to_have_count(2, timeout=60000)
        # Stop commits the tail before disconnecting; then the socket drains.
        page.locator("#stop").click()
        expect(page.locator("#activity")).to_contain_text("Microphone off", timeout=10000)
        expect(page.locator("#start")).to_be_enabled(timeout=60000)
        with page.expect_download() as download:
            page.locator("#download").click()
        exported = json.loads(Path(download.value.path()).read_text())
        assert exported["session"]["session_id"] == session_id
        assert len(exported["results"]) >= 2
        if args.screenshot:
            page.screenshot(path=str(args.screenshot), full_page=True)
        page.locator("#attach").click()
        expect(page.locator("#status")).to_contain_text("Session opened")
        assert page.locator("#results article").count() >= 2
        page.locator("#end").click()
        expect(page.locator("#sessionInfo")).to_contain_text("ended")
        # Both automatic delivery options must work without Send sentence.
        for delivery in ("auto", "live"):
            page.locator("#create").click()
            expect(page.locator("#sessionInfo")).to_contain_text("active")
            page.locator("#mode").select_option("parakeet")
            page.locator("#delivery").select_option(delivery)
            page.locator("#pause").select_option("1.8")
            page.locator("#liveTab").click()
            page.locator("#start").click()
            expect(page.locator("#status")).to_contain_text("Microphone on", timeout=20000)
            if delivery == "live":
                expect(page.locator("#draftText")).to_contain_text(re.compile(r".{12,}"), timeout=60000)
                expect(page.locator("#draftSpeaker")).to_have_text("Speaker recognition off")
            expect(page.locator("#liveResults article").first).to_be_visible(timeout=90000)
            page.locator("#stop").click()
            expect(page.locator("#start")).to_be_enabled(timeout=60000)
            page.locator("#end").click()
            expect(page.locator("#sessionInfo")).to_contain_text("ended")
        # Exercise signature-deletion controls without storing a biometric.
        page.locator("#capture").check()
        page.locator("#create").click()
        expect(page.locator("#sessionInfo")).to_contain_text("voice signatures enabled")
        page.locator("#delete").click()
        expect(page.locator("#sessionInfo")).to_contain_text("voice signatures disabled")
        page.locator("#end").click()
        expect(page.locator("#sessionInfo")).to_contain_text("ended")
        assert not errors, errors
        # UI-only named-speaker fixture: no voice embedding or fake identity is
        # written to the server. Verify safe text rendering and local recovery.
        named = context.new_page()
        named.route("**/v2/sessions/ui-speaker-test", lambda route: route.fulfill(json={
            "session_id": "ui-speaker-test", "status": "active", "capture_voiceprints": False,
            "utterances": [{"utterance_id": "ui-1", "sequence": 1, "speaker_id": "speaker_1",
                            "text": "A named speaker", "language": "en", "audio_seconds": 2}],
        }))
        named.goto(args.url.rstrip("/") + "/v2/test")
        named.locator("#sessionId").fill("ui-speaker-test")
        named.locator("#attach").click()
        expect(named.locator("#liveResults .speaker")).to_have_text("speaker_1")
        named.locator("#speakerName").fill("Maria <example>")
        named.locator("#nameSpeaker").click()
        expect(named.locator("#liveResults .speaker")).to_have_text("Maria <example> · speaker_1")
        named.locator("#attach").click()
        expect(named.locator("#liveResults .speaker")).to_have_text("Maria <example> · speaker_1")
        named.close()
        print("Browser upload, comparison, manual/auto/live microphone, endpointing, export, and deletion: OK")
        browser.close()


if __name__ == "__main__":
    main()
