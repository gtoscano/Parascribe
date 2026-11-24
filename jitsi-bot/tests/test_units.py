import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

from helpers import silence, tone
from bot import meeting_url, parse_room
from endpointing import Frame, SentenceGate
from transcript import Transcript


def frames(pcm: bytes, size: int = 512):
    step = size * 2
    return [Frame(pcm[i:i + step], i / 2 / 16000) for i in range(0, len(pcm), step)]


def run_gate(gate, pcm):
    sent, boundaries = 0, []
    for frame in frames(pcm):
        out, boundary = gate.push(frame)
        sent += sum(item.samples for item in out)
        if boundary:
            boundaries.append((boundary, sent))
    return sent, boundaries


class GateTest(unittest.TestCase):
    def test_silence_is_never_sent(self):
        self.assertEqual(run_gate(SentenceGate(), silence(5)), (0, []))

    def test_sentence_commits_after_pause_with_pre_roll(self):
        sent, boundaries = run_gate(SentenceGate(), silence(1) + tone(1.5) + silence(1))
        self.assertEqual([kind for kind, _ in boundaries], ["commit"])
        # Speech plus ~200 ms pre-roll plus the 0.8 s pause that ended it.
        self.assertAlmostEqual(boundaries[0][1] / 16000, 1.5 + 0.2 + 0.8, delta=0.1)

    def test_short_blip_is_discarded(self):
        _, boundaries = run_gate(SentenceGate(), tone(0.1) + silence(1))
        self.assertEqual([kind for kind, _ in boundaries], ["discard"])

    def test_long_speech_is_split_at_max_seconds(self):
        _, boundaries = run_gate(SentenceGate(max_seconds=5), tone(12))
        self.assertEqual([kind for kind, _ in boundaries], ["commit", "commit"])

    def test_finish_ends_an_utterance_in_progress(self):
        gate = SentenceGate()
        for frame in frames(tone(1)):
            gate.push(frame)
        self.assertEqual(gate.finish(), "commit")
        self.assertIsNone(gate.finish())


class RoomTest(unittest.TestCase):
    def test_room_name_or_link(self):
        self.assertEqual(parse_room("Synod-Group-3", "https://meet.jit.si/"),
                         ("https://meet.jit.si", "Synod-Group-3"))
        self.assertEqual(parse_room("https://jitsi.example.org/abc", "https://meet.jit.si"),
                         ("https://jitsi.example.org", "abc"))

    def test_url_fragment_is_json_encoded(self):
        url = meeting_url("https://meet.jit.si", "Grupo 3", "Parascribe (transcriber)")
        self.assertTrue(url.startswith("https://meet.jit.si/Grupo%203#"))
        fragment = dict(item.split("=", 1) for item in url.split("#", 1)[1].split("&"))
        self.assertEqual(json.loads(unquote(fragment["userInfo.displayName"])),
                         "Parascribe (transcriber)")
        self.assertIs(json.loads(unquote(fragment["config.prejoinConfig.enabled"])), False)


class TranscriptTest(unittest.TestCase):
    def test_sorted_by_spoken_time_and_grouped_by_speaker(self):
        with tempfile.TemporaryDirectory() as folder:
            started = datetime(2026, 11, 24, 1, 0, tzinfo=timezone.utc)
            transcript = Transcript(Path(folder) / "room.md", room="room", session_id="s",
                                    started=started, echo=False)
            transcript.add({"text": "Second", "speaker_label": "Luis",
                            "spoken_at": "2026-11-24T01:00:20+00:00"})
            transcript.add({"text": "First", "speaker_label": "Ana",
                            "spoken_at": "2026-11-24T01:00:05+00:00"})
            transcript.add({"text": "Again", "speaker_label": "Ana",
                            "spoken_at": "2026-11-24T01:00:08+00:00"})
            transcript.add({"text": "  ", "speaker_label": "Ana"})
            body = (Path(folder) / "room.md").read_text()
            self.assertLess(body.index("First"), body.index("Again"))
            self.assertLess(body.index("Again"), body.index("Second"))
            self.assertIn("**Ana** · 00:00:05", body)
            self.assertEqual(body.count("**Ana**"), 1)
            data = json.loads((Path(folder) / "room.json").read_text())
            self.assertEqual([u["speaker"] for u in data["utterances"]], ["Ana", "Ana", "Luis"])


if __name__ == "__main__":
    unittest.main()
