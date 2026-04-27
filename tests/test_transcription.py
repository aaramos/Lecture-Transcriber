import unittest
from unittest import mock

from lecture_processor.config import TranscriptionEngine
from lecture_processor.errors import DependencyMissingError, ProcessingError
from lecture_processor.transcription import (
    NullTranscriber,
    _whisper_cpp_segment_times,
    build_transcriber,
)


class FakeWhisperCppSegment:
    t0 = 125
    t1 = 260


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


if __name__ == "__main__":
    unittest.main()
