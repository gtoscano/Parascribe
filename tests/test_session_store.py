import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parascribe"))
from session_store import SessionEnded, SessionStore


class SessionStoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = SessionStore(Path(self.temp.name) / "sessions.sqlite3")

    def tearDown(self):
        self.temp.cleanup()

    def test_voiceprints_are_scoped_to_one_session(self):
        first = self.store.create_session()
        second = self.store.create_session()
        voice = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        a = self.store.match_voiceprint(
            first["session_id"], voice, audio_seconds=2.0,
            similarity_threshold=0.5, update_threshold=0.7,
        )
        b = self.store.match_voiceprint(
            second["session_id"], voice, audio_seconds=2.0,
            similarity_threshold=0.5, update_threshold=0.7,
        )

        self.assertEqual(a, ("speaker_1", None, True))
        self.assertEqual(b, ("speaker_1", None, True))
        self.assertEqual(self.store.get_session(first["session_id"])["speaker_count"], 1)
        self.assertEqual(self.store.get_session(second["session_id"])["speaker_count"], 1)

    def test_match_updates_only_a_confident_centroid(self):
        session_id = self.store.create_session()["session_id"]
        self.store.match_voiceprint(
            session_id, np.array([1.0, 0.0]), audio_seconds=1.0,
            similarity_threshold=0.5, update_threshold=0.8,
        )

        speaker, similarity, enrolled = self.store.match_voiceprint(
            session_id, np.array([0.9, 0.1]), audio_seconds=1.5,
            similarity_threshold=0.5, update_threshold=0.8,
        )

        self.assertEqual(speaker, "speaker_1")
        self.assertGreater(similarity, 0.99)
        self.assertFalse(enrolled)

    def test_a_dissimilar_voice_enrolls_a_new_session_speaker(self):
        session_id = self.store.create_session()["session_id"]
        self.store.match_voiceprint(
            session_id, np.array([1.0, 0.0]), audio_seconds=1.0,
            similarity_threshold=0.5, update_threshold=0.8,
        )
        speaker, similarity, enrolled = self.store.match_voiceprint(
            session_id, np.array([0.0, 1.0]), audio_seconds=1.0,
            similarity_threshold=0.5, update_threshold=0.8,
        )

        self.assertEqual(speaker, "speaker_2")
        self.assertIsNone(similarity)
        self.assertTrue(enrolled)

    def test_deletion_keeps_transcript_but_prevents_new_voiceprints(self):
        session_id = self.store.create_session()["session_id"]
        self.store.match_voiceprint(
            session_id, np.array([1.0, 0.0]), audio_seconds=1.0,
            similarity_threshold=0.5, update_threshold=0.8,
        )
        self.store.append_utterance(
            session_id, text="Hello.", language="en", speaker_id="speaker_1",
            speaker_similarity=None, audio_seconds=1.0, source="sentence",
        )

        self.assertEqual(self.store.delete_voiceprints(session_id), 1)
        speaker, similarity, enrolled = self.store.match_voiceprint(
            session_id, np.array([1.0, 0.0]), audio_seconds=1.0,
            similarity_threshold=0.5, update_threshold=0.8,
        )
        record = self.store.get_session(session_id, include_utterances=True)

        self.assertEqual((speaker, similarity, enrolled), (None, None, False))
        self.assertEqual(record["speaker_count"], 0)
        self.assertEqual(record["utterances"][0]["speaker_id"], "speaker_1")
        self.assertFalse(record["capture_voiceprints"])

    def test_ended_session_rejects_new_utterances(self):
        session_id = self.store.create_session()["session_id"]
        self.store.end_session(session_id, delete_voiceprints=True)
        with self.assertRaises(SessionEnded):
            self.store.append_utterance(
                session_id, text="Too late", language="en", speaker_id=None,
                speaker_similarity=None, audio_seconds=1.0, source="sentence",
            )

    def test_concurrent_matches_do_not_duplicate_a_speaker(self):
        session_id = self.store.create_session()["session_id"]

        def match(_):
            return self.store.match_voiceprint(
                session_id, np.array([1.0, 0.0]), audio_seconds=1.0,
                similarity_threshold=0.5, update_threshold=0.7,
            )[0]

        with ThreadPoolExecutor(max_workers=8) as pool:
            speakers = list(pool.map(match, range(16)))

        self.assertEqual(set(speakers), {"speaker_1"})
        self.assertEqual(self.store.get_session(session_id)["speaker_count"], 1)


if __name__ == "__main__":
    unittest.main()
