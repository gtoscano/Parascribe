import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


_TEMP_DATA = tempfile.TemporaryDirectory()
os.environ["DATA_DIR"] = _TEMP_DATA.name
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parascribe"))

from fastapi.testclient import TestClient  # noqa: E402
import server  # noqa: E402


class SessionApiTest(unittest.TestCase):
    def test_browser_tester_assets_are_served_from_the_api(self):
        with TestClient(server.app) as client:
            page = client.get("/v2/test")
            self.assertEqual(page.status_code, 200)
            self.assertIn("/v2/test/client.js", page.text)
            for name in ("client.js", "pcm-worklet.js", "streaming.js"):
                asset = client.get(f"/v2/test/{name}")
                self.assertEqual(asset.status_code, 200)
                self.assertIn("javascript", asset.headers["content-type"])

    def test_live_preview_is_ephemeral_and_final_is_stored_once(self):
        import numpy as np
        from test_sentence_asr import FakeTranscriber

        decoder = FakeTranscriber("parakeet-test", "Words as I speak")
        with patch.object(server.ModelSet, "get_sentence_asr", return_value=decoder), \
                patch.object(server.ModelSet, "get_voiceprint") as voiceprint, \
                patch.object(server, "_load_audio_16k", return_value=np.zeros(16000)), \
                TestClient(server.app) as client:
            session_id = client.post("/v2/sessions", json={"capture_voiceprints": True}).json()["session_id"]
            jobs_before = set(server.JOBS)
            with client.websocket_connect(f"/v2/sessions/{session_id}/stream") as socket:
                socket.receive_json()
                socket.send_json({"type": "start", "partials": True})
                socket.receive_json()
                socket.send_bytes(b"\0\0" * 16000)
                socket.send_json({"type": "preview", "utterance_id": "one"})
                draft = socket.receive_json()
                self.assertEqual(draft["type"], "partial")
                self.assertEqual(draft["client_utterance_id"], "one")
                self.assertTrue(draft["provisional"])
                self.assertEqual(draft["speaker_status"], "pending")
                self.assertEqual(set(server.JOBS), jobs_before)
                self.assertEqual(client.get(f"/v2/sessions/{session_id}").json()["utterance_count"], 0)
                voiceprint.assert_not_called()
                # Disable capture before finalization: the preview above must
                # not have extracted a signature even while capture was on.
                client.delete(f"/v2/sessions/{session_id}/voiceprints")
                socket.send_json({"type": "commit", "utterance_id": "one"})
                self.assertEqual(socket.receive_json()["type"], "utterance_queued")
                self.assertEqual(socket.receive_json()["type"], "utterance")
                self.assertEqual(client.get(f"/v2/sessions/{session_id}").json()["utterance_count"], 1)
                socket.send_json({"type": "end", "delete_voiceprints": True})
                self.assertEqual(socket.receive_json()["type"], "session_ended")

    def test_discard_and_preview_opt_in(self):
        with TestClient(server.app) as client:
            session_id = client.post("/v2/sessions", json={}).json()["session_id"]
            with client.websocket_connect(f"/v2/sessions/{session_id}/stream") as socket:
                socket.receive_json()
                socket.send_json({"type": "start"})
                socket.receive_json()
                socket.send_bytes(b"\0\0" * 16000)
                socket.send_json({"type": "preview"})
                self.assertEqual(socket.receive_json()["type"], "partial_skipped")
                socket.send_json({"type": "discard"})
                socket.send_json({"type": "commit"})
                self.assertEqual(socket.receive_json()["error"], "no audio to commit")
                socket.send_json({"type": "end", "delete_voiceprints": True})
                self.assertEqual(socket.receive_json()["type"], "session_ended")

    def test_commit_invalidates_in_flight_partial(self):
        import threading
        import numpy as np
        from test_sentence_asr import FakeTranscriber

        entered, release = threading.Event(), threading.Event()
        def slow_preview(item, models):
            entered.set()
            release.wait(timeout=10)
            return {"primary": "parakeet", "outputs": {"parakeet": {"text": "stale"}}}

        with patch.object(server, "run_preview", side_effect=slow_preview), \
                patch.object(server.ModelSet, "get_sentence_asr", return_value=FakeTranscriber("test", "Final")), \
                patch.object(server, "_load_audio_16k", return_value=np.zeros(16000)), \
                TestClient(server.app) as client:
            session_id = client.post("/v2/sessions", json={"capture_voiceprints": False}).json()["session_id"]
            try:
                with client.websocket_connect(f"/v2/sessions/{session_id}/stream") as socket:
                    socket.receive_json()
                    socket.send_json({"type": "start", "partials": True})
                    socket.receive_json()
                    socket.send_bytes(b"\0\0" * 16000)
                    socket.send_json({"type": "preview", "utterance_id": "old"})
                    self.assertTrue(entered.wait(timeout=5))
                    socket.send_json({"type": "commit", "utterance_id": "old"})
                    self.assertEqual(socket.receive_json()["type"], "utterance_queued")
                    release.set()
                    self.assertEqual(socket.receive_json()["type"], "utterance")
                    socket.send_json({"type": "end"})
                    self.assertEqual(socket.receive_json()["type"], "session_ended")
            finally:
                release.set()

    def test_sentence_rejects_invalid_mode_and_missing_cohere_language(self):
        with TestClient(server.app) as client:
            session_id = client.post("/v2/sessions", json={}).json()["session_id"]
            for fields in ({"asr_mode": "unknown"}, {"asr_mode": "compare"}):
                response = client.post(
                    f"/v2/sessions/{session_id}/sentences", data=fields,
                    files={"file": ("test.wav", b"unused", "audio/wav")})
                self.assertEqual(response.status_code, 400)

    def test_unconfigured_cohere_is_rejected_before_queueing(self):
        with TestClient(server.app) as client, patch.object(server, "COHERE_MODEL_PATH", ""):
            session_id = client.post("/v2/sessions", json={}).json()["session_id"]
            response = client.post(
                f"/v2/sessions/{session_id}/sentences",
                data={"asr_mode": "cohere", "language": "en"},
                files={"file": ("test.wav", b"unused", "audio/wav")})
            self.assertEqual(response.status_code, 503)

    def test_websocket_comparison_routes_language_and_persists_both_outputs(self):
        import numpy as np
        from test_sentence_asr import FakeTranscriber

        parakeet = FakeTranscriber("parakeet-test", "Draft words")
        cohere = FakeTranscriber("cohere-test", "Final words")
        with tempfile.NamedTemporaryFile(suffix=".gguf") as model_file, \
                patch.object(server, "COHERE_MODEL_PATH", model_file.name), \
                patch.object(server.ModelSet, "get_sentence_asr", return_value=parakeet), \
                patch.object(server.ModelSet, "get_cohere", return_value=cohere), \
                patch.object(server, "_load_audio_16k", return_value=np.zeros(16000)), \
                TestClient(server.app) as client:
            session_id = client.post(
                "/v2/sessions", json={"capture_voiceprints": False}).json()["session_id"]
            with client.websocket_connect(f"/v2/sessions/{session_id}/stream") as socket:
                socket.receive_json()
                socket.send_json({"type": "start", "asr_mode": "compare", "language": "es"})
                self.assertEqual(socket.receive_json()["asr_mode"], "compare")
                socket.send_bytes(b"\0\0" * 16000)
                socket.send_json({"type": "commit", "utterance_id": "both"})
                queued = socket.receive_json()
                completed = socket.receive_json()
                self.assertEqual(completed["type"], "utterance")
                outputs = completed["result"]["transcription"]["outputs"]
                self.assertEqual(outputs["cohere"]["text"], "Final words")
                self.assertEqual(outputs["parakeet"]["language"], "es")
                job_id = queued["job_id"]
                persisted = client.get(f"/v1/jobs/{job_id}/result.json").json()
                self.assertEqual(persisted["transcription"]["outputs"], outputs)
                self.assertIn("cohere_transcribe_s", persisted["timings"])
                self.assertFalse(Path(server.JOBS[job_id]["audio_path"]).exists())
                socket.send_json({"type": "end"})
                socket.receive_json()

    def test_session_lifecycle_and_voiceprint_deletion(self):
        with TestClient(server.app) as client:
            created = client.post(
                "/v2/sessions",
                json={"capture_voiceprints": True, "metadata": {"case": "api-test"}},
            )
            self.assertEqual(created.status_code, 200)
            session_id = created.json()["session_id"]

            fetched = client.get(f"/v2/sessions/{session_id}")
            self.assertEqual(fetched.status_code, 200)
            self.assertEqual(fetched.json()["metadata"], {"case": "api-test"})

            deleted = client.delete(f"/v2/sessions/{session_id}/voiceprints")
            self.assertEqual(deleted.status_code, 200)
            self.assertFalse(deleted.json()["capture_voiceprints"])

            ended = client.post(
                f"/v2/sessions/{session_id}/end",
                json={"delete_voiceprints": True},
            )
            self.assertEqual(ended.status_code, 200)
            self.assertEqual(ended.json()["status"], "ended")

    def test_websocket_control_protocol_without_audio(self):
        with TestClient(server.app) as client:
            session_id = client.post("/v2/sessions", json={}).json()["session_id"]
            with client.websocket_connect(
                f"/v2/sessions/{session_id}/stream"
            ) as socket:
                self.assertEqual(socket.receive_json()["type"], "ready")
                socket.send_json({
                    "type": "start", "format": "pcm_s16le",
                    "sample_rate": 16000, "channels": 1,
                })
                self.assertEqual(socket.receive_json()["type"], "started")
                socket.send_json({"type": "ping"})
                self.assertEqual(socket.receive_json(), {"type": "pong"})
                socket.send_json({"type": "end", "delete_voiceprints": True})
                ended = socket.receive_json()
                self.assertEqual(ended["type"], "session_ended")
                self.assertEqual(ended["session"]["status"], "ended")


if __name__ == "__main__":
    unittest.main()
