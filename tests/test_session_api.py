import os
import sys
import tempfile
import unittest
from pathlib import Path


_TEMP_DATA = tempfile.TemporaryDirectory()
os.environ["DATA_DIR"] = _TEMP_DATA.name
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whisperx"))

from fastapi.testclient import TestClient  # noqa: E402
import server  # noqa: E402


class SessionApiTest(unittest.TestCase):
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
