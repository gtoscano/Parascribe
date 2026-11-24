"""TrackStream against a fake Parascribe v2 WebSocket endpoint."""
import asyncio
import json
import unittest
from datetime import datetime

import websockets

from helpers import silence, tone
from streams import TrackStream


class FakeParascribe:
    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.connections = 0
        self.commits, self.discards = [], 0

    async def handler(self, socket):
        self.connections += 1
        await socket.send(json.dumps({"type": "ready"}))
        audio = 0
        async for raw in socket:
            if isinstance(raw, bytes):
                audio += len(raw)
                continue
            command = json.loads(raw)
            if command["type"] == "start":
                await socket.send(json.dumps({"type": "started"}))
            elif command["type"] == "discard":
                self.discards += 1
                audio = 0
            elif command["type"] == "commit":
                self.commits.append({**command, "bytes": audio})
                audio = 0
                asyncio.create_task(self.reply(socket, command))

    async def reply(self, socket, command):
        await asyncio.sleep(self.delay)
        await socket.send(json.dumps({"type": "utterance_queued",
                                      "client_utterance_id": command["utterance_id"]}))
        await socket.send(json.dumps({"type": "utterance",
                                      "client_utterance_id": command["utterance_id"],
                                      "result": {"text": f"from {command['speaker_label']}",
                                                 "speaker_label": command["speaker_label"],
                                                 "spoken_at": command["spoken_at"]}}))


class TrackStreamTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fake = FakeParascribe(delay=0.2)
        self.server = await websockets.serve(self.fake.handler, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        self.results = []
        self.names = {"a": "Ana", "b": "Luis"}
        self.stream = TrackStream("t1", ws_url=f"ws://127.0.0.1:{port}", names=self.names,
                                  on_result=self.results.append, language="es")

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()

    async def feed(self, owner, pcm, start):
        # Deliver like the page does: ~200 ms chunks stamped when they arrive.
        step = 6400
        for offset in range(0, len(pcm), step):
            chunk = pcm[offset:offset + step]
            await self.stream.feed(owner, chunk, start + (offset + len(chunk)) / 32000)

    async def test_sentences_are_labelled_timed_and_awaited_on_close(self):
        await self.feed("a", silence(0.5) + tone(1.0) + silence(1.0), 1000.0)
        await self.feed("b", tone(1.0), 1002.5)
        await self.stream.close()

        labels = [c["speaker_label"] for c in self.fake.commits]
        self.assertEqual(labels, ["Ana", "Luis"])
        self.assertEqual([r["text"] for r in self.results], ["from Ana", "from Luis"])
        # Ana's sentence began at 1000.5 s, minus the gate's ~200 ms pre-roll.
        spoken = datetime.fromisoformat(self.fake.commits[0]["spoken_at"]).timestamp()
        self.assertAlmostEqual(spoken, 1000.3, delta=0.05)
        self.assertEqual(self.results[1]["participant_id"], "b")
        self.assertEqual(self.fake.connections, 1)

    async def test_owner_change_splits_an_utterance(self):
        await self.feed("a", tone(1.0), 0.0)
        await self.feed("b", tone(1.0), 1.0)  # same track, reassigned mid-speech
        await self.stream.close()
        self.assertEqual([c["speaker_label"] for c in self.fake.commits], ["Ana", "Luis"])
        self.assertTrue(all(c["bytes"] >= 16000 * 2 * 0.9 for c in self.fake.commits))

    async def test_unowned_audio_and_blips_are_not_transcribed(self):
        await self.feed(None, tone(2.0), 0.0)
        await self.feed("a", tone(0.1) + silence(1.0), 2.0)
        await self.stream.close()
        self.assertEqual(self.fake.commits, [])
        self.assertEqual(self.fake.discards, 1)

    async def test_idle_track_ends_the_open_sentence(self):
        await self.feed("a", tone(1.0), 0.0)  # then the speaker mutes: no more audio
        await self.stream.idle()
        await self.feed("a", tone(1.0), 30.0)
        await self.stream.close()
        self.assertEqual(len(self.fake.commits), 2)
        starts = [datetime.fromisoformat(c["spoken_at"]).timestamp() for c in self.fake.commits]
        self.assertAlmostEqual(starts[1], 30.0, delta=0.1)  # clock re-anchored after the gap

    async def test_label_uses_name_known_at_commit(self):
        self.names.pop("a")
        await self.feed("a", tone(1.0) + silence(1.0), 0.0)
        await self.stream.close()
        self.assertEqual(self.fake.commits[0]["speaker_label"], "Participant a")


if __name__ == "__main__":
    unittest.main()
