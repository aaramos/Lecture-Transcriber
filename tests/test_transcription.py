import os
import unittest
from pathlib import Path
from unittest import mock

from lecture_processor.config import TranscriptionEngine
from lecture_processor.errors import DependencyMissingError, ProcessingError
from lecture_processor.transcription import (
    FasterWhisperTranscriber,
    NullTranscriber,
    _whisper_cpp_segment_times,
    build_transcriber,
    configure_whisper_cpp_runtime_env,
)


class FakeWhisperCppSegment:
    t0 = 125
    t1 = 260


class FakeFasterWhisperSegment:
    start = 1.0
    end = 3.0
    text = " hello lecture "


class TranscriptionTests(unittest.TestCase):
    def test_whisper_cpp_segment_times_convert_centiseconds(self):
        self.assertEqual(_whisper_cpp_segment_times(FakeWhisperCppSegment()), (1.25, 2.6))

    def test_explicit_whisper_cpp_reports_missing_dependency(self):
        with mock.patch(
            "lecture_processor.transcription.importlib.import_module",
            side_effect=ImportError("missing"),
        ):
            with self.assertRaises(DependencyMissingError) as context:
                build_transcriber(TranscriptionEngine.WHISPER_CPP, "tiny")

        self.assertIn("pywhispercpp", str(context.exception))

    def test_explicit_whisper_cpp_reports_initialization_failure(self):
        with mock.patch(
            "lecture_processor.transcription.WhisperCppTranscriber",
            side_effect=RuntimeError("model failed"),
        ):
            with self.assertRaises(ProcessingError) as context:
                build_transcriber(TranscriptionEngine.WHISPER_CPP, "large-v3")

        self.assertIn("pywhispercpp", str(context.exception))

    def test_auto_falls_back_when_preferred_whisper_cpp_cannot_initialize(self):
        with mock.patch(
            "lecture_processor.transcription.WhisperCppTranscriber",
            side_effect=RuntimeError("model failed"),
        ), mock.patch(
            "lecture_processor.transcription.FasterWhisperTranscriber",
            return_value=NullTranscriber(),
        ):
            transcriber = build_transcriber(
                TranscriptionEngine.AUTO,
                "large-v3",
                prefer_whisper_cpp=True,
            )

        self.assertIsInstance(transcriber._transcriber, NullTranscriber)

    def test_faster_whisper_uses_safer_english_transcription_options(self):
        captured = {}

        class FakeWhisperModel:
            def __init__(self, model_name):
                captured["model_name"] = model_name

            def transcribe(self, media_path, **options):
                captured["media_path"] = media_path
                captured["options"] = options
                return iter([FakeFasterWhisperSegment()]), object()

        fake_module = type("FakeFasterWhisperModule", (), {"WhisperModel": FakeWhisperModel})
        with mock.patch("lecture_processor.transcription.importlib.import_module", return_value=fake_module):
            result = FasterWhisperTranscriber("large-v3").transcribe(Path("clean.wav"))

        self.assertEqual(result.text, "hello lecture")
        self.assertEqual(captured["model_name"], "large-v3")
        self.assertEqual(captured["media_path"], "clean.wav")
        self.assertEqual(captured["options"]["language"], "en")
        self.assertFalse(captured["options"]["condition_on_previous_text"])
        self.assertTrue(captured["options"]["vad_filter"])
        self.assertEqual(captured["options"]["temperature"], [0.0, 0.2, 0.4])
        self.assertEqual(captured["options"]["no_repeat_ngram_size"], 5)

    def test_whisper_cpp_runtime_defaults_disable_unstable_metal_decoder(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            configure_whisper_cpp_runtime_env()

            self.assertEqual(os.environ["PYWHISPERCPP_USE_GPU"], "0")
            self.assertEqual(os.environ["PYWHISPERCPP_FLASH_ATTN"], "0")

    def test_whisper_cpp_runtime_preserves_explicit_overrides(self):
        with mock.patch.dict(
            os.environ,
            {"PYWHISPERCPP_USE_GPU": "1", "PYWHISPERCPP_FLASH_ATTN": "1"},
            clear=True,
        ):
            configure_whisper_cpp_runtime_env()

            self.assertEqual(os.environ["PYWHISPERCPP_USE_GPU"], "1")
            self.assertEqual(os.environ["PYWHISPERCPP_FLASH_ATTN"], "1")


if __name__ == "__main__":
    unittest.main()
