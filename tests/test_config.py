import tempfile
import unittest
from pathlib import Path

from lecture_processor.config import AudioQuality, BatchConfig, RecordingSpeed, TranscriptionQuality
from lecture_processor.errors import LectureProcessorError


class BatchConfigTests(unittest.TestCase):
    def test_requires_confirmation_for_2x_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = BatchConfig(
                input_dir=folder,
                output_dir=folder / "out",
                recording_speed=RecordingSpeed.DOUBLE,
                confirm_normalization=False,
            )

            with self.assertRaises(LectureProcessorError) as context:
                config.validate()

            self.assertIn("--confirm-normalization", str(context.exception))

    def test_validates_concurrent_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = BatchConfig(
                input_dir=folder,
                output_dir=folder / "out",
                concurrent_files=9,
            )

            with self.assertRaises(LectureProcessorError):
                config.validate()

    def test_defaults_match_app_recommendations(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = BatchConfig(input_dir=folder, output_dir=folder / "out")

            self.assertIs(config.audio_quality, AudioQuality.HIGH)
            self.assertIs(config.transcription_quality, TranscriptionQuality.ACCURATE)
            self.assertEqual(config.whisper_model, "medium.en")


if __name__ == "__main__":
    unittest.main()
