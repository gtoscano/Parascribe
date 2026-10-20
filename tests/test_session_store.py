import sqlite3
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

        for match in (a, b):
            self.assertEqual(
                (match["speaker_id"], match["similarity"], match["enrolled"]),
                ("speaker_1", None, True))
        self.assertEqual(self.store.get_session(first["session_id"])["speaker_count"], 1)
        self.assertEqual(self.store.get_session(second["session_id"])["speaker_count"], 1)

    def test_match_updates_only_a_confident_centroid(self):
        session_id = self.store.create_session()["session_id"]
        self.store.match_voiceprint(
            session_id, np.array([1.0, 0.0]), audio_seconds=1.0,
            similarity_threshold=0.5, update_threshold=0.8,
        )

        match = self.store.match_voiceprint(
            session_id, np.array([0.9, 0.1]), audio_seconds=1.5,
            similarity_threshold=0.5, update_threshold=0.8,
        )
        speaker, similarity, enrolled = (
            match["speaker_id"], match["similarity"], match["enrolled"])

        self.assertEqual(speaker, "speaker_1")
        self.assertGreater(similarity, 0.99)
        self.assertFalse(enrolled)

    def test_a_dissimilar_voice_enrolls_a_new_session_speaker(self):
        session_id = self.store.create_session()["session_id"]
        self.store.match_voiceprint(
            session_id, np.array([1.0, 0.0]), audio_seconds=1.0,
            similarity_threshold=0.5, update_threshold=0.8,
        )
        match = self.store.match_voiceprint(
            session_id, np.array([0.0, 1.0]), audio_seconds=1.0,
            similarity_threshold=0.5, update_threshold=0.8,
        )
        speaker, similarity, enrolled = (
            match["speaker_id"], match["similarity"], match["enrolled"])

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
        match = self.store.match_voiceprint(
            session_id, np.array([1.0, 0.0]), audio_seconds=1.0,
            similarity_threshold=0.5, update_threshold=0.8,
        )
        speaker, similarity, enrolled = (
            match["speaker_id"], match["similarity"], match["enrolled"])
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
            )["speaker_id"]

        with ThreadPoolExecutor(max_workers=8) as pool:
            speakers = list(pool.map(match, range(16)))

        self.assertEqual(set(speakers), {"speaker_1"})
        self.assertEqual(self.store.get_session(session_id)["speaker_count"], 1)

    def _utterance(self, session_id, text):
        return self.store.append_utterance(
            session_id, text=text, language="en", speaker_id=None,
            speaker_similarity=None, audio_seconds=2.0, source="sentence",
        )["utterance_id"]

    def _confirmed_match(self, session_id, utterance_id, voice):
        return self.store.match_voiceprint(
            session_id, np.array(voice), audio_seconds=2.0,
            similarity_threshold=0.5, update_threshold=0.5,
            utterance_id=utterance_id, confirm_new_speakers=True,
        )

    def test_unmatched_voice_waits_for_confirmation_then_relabels(self):
        session_id = self.store.create_session()["session_id"]
        first = self._utterance(session_id, "Hello.")
        pending = self._confirmed_match(session_id, first, [1.0, 0.0])
        self.assertEqual((pending["speaker_id"], pending["pending"]), (None, True))
        self.assertEqual(self.store.get_session(session_id)["speaker_count"], 0)

        second = self._utterance(session_id, "Hello again.")
        confirmed = self._confirmed_match(session_id, second, [0.9, 0.1])
        record = self.store.get_session(session_id, include_utterances=True)

        self.assertEqual(confirmed["speaker_id"], "speaker_1")
        self.assertTrue(confirmed["enrolled"])
        self.assertEqual([item["utterance_id"] for item in confirmed["relabeled"]], [first])
        self.assertEqual([u["speaker_id"] for u in record["utterances"]],
                         ["speaker_1", "speaker_1"])
        self.assertEqual(record["speaker_count"], 1)

    def test_unconfirmed_different_voices_stay_pending(self):
        session_id = self.store.create_session()["session_id"]
        self._confirmed_match(session_id, self._utterance(session_id, "A"), [1.0, 0.0])
        match = self._confirmed_match(session_id, self._utterance(session_id, "B"), [0.0, 1.0])

        self.assertTrue(match["pending"])
        self.assertEqual(self.store.get_session(session_id)["speaker_count"], 0)

    def test_voiceprint_deletion_also_deletes_pending_voices(self):
        session_id = self.store.create_session()["session_id"]
        self._confirmed_match(session_id, self._utterance(session_id, "A"), [1.0, 0.0])
        self.store.delete_voiceprints(session_id)
        with sqlite3.connect(self.store.path) as connection:
            remaining = connection.execute(
                "SELECT COUNT(*) FROM pending_voiceprints").fetchone()[0]
        self.assertEqual(remaining, 0)


if __name__ == "__main__":
    unittest.main()
