import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parascribe"))
from sentence_asr import run_sentence_models, validate_asr_request


class FakeTranscriber:
    quantization = "test"
    device = "cpu"

    def __init__(self, name, text):
        self.name = name
        self.text = text
        self.calls = []

    def transcribe(self, audio, *, language=None):
        self.calls.append((audio, language))
        return self.text, language or "und"


class SentenceAsrTest(unittest.TestCase):
    def setUp(self):
        self.audio = np.zeros(16000, dtype=np.float32)
        self.parakeet = FakeTranscriber("parakeet-test", "Parakeet draft")
        self.cohere = FakeTranscriber("cohere-test", "Cohere final")

    def run_models(self, mode, language=None):
        return run_sentence_models(
            self.audio, mode=mode, language=language,
            get_parakeet=lambda: self.parakeet, get_cohere=lambda: self.cohere)

    def test_parakeet_does_not_load_cohere(self):
        result, timings = self.run_models("parakeet")
        self.assertEqual(result["primary"], "parakeet")
        self.assertEqual(set(result["outputs"]), {"parakeet"})
        self.assertFalse(self.cohere.calls)
        self.assertNotIn("cohere_transcribe_s", timings)

    def test_cohere_does_not_load_parakeet(self):
        result, _ = self.run_models("cohere", "es")
        self.assertEqual(result["primary"], "cohere")
        self.assertEqual(result["outputs"]["cohere"]["language"], "es")
        self.assertFalse(self.parakeet.calls)

    def test_compare_keeps_both_outputs_on_identical_audio(self):
        result, timings = self.run_models("compare", "en")
        self.assertEqual(result["primary"], "parakeet")
        self.assertEqual(result["outputs"]["cohere"]["text"], "Cohere final")
        self.assertEqual(result["outputs"]["parakeet"]["text"], "Parakeet draft")
        self.assertIs(self.parakeet.calls[0][0], self.cohere.calls[0][0])
        self.assertEqual(result["audio_seconds"], 1)
        for name in ("cohere", "parakeet"):
            self.assertGreaterEqual(timings[f"{name}_transcribe_s"], 0)
            self.assertIn("real_time_factor", result["outputs"][name])

    def test_compare_does_not_filter_cohere_using_parakeet_output(self):
        self.parakeet.text = ""
        result, _ = self.run_models("compare", "es")
        self.assertEqual(result["outputs"]["cohere"]["text"], "Cohere final")
        self.assertEqual(len(self.cohere.calls), 1)

    def test_language_and_mode_validation(self):
        for mode, language in (("invalid", "en"), ("cohere", None),
                               ("compare", "und"), ("cohere", "ru")):
            with self.subTest(mode=mode, language=language):
                with self.assertRaises(ValueError):
                    validate_asr_request(mode, language)


if __name__ == "__main__":
    unittest.main()
