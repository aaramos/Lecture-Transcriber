import tempfile
import unittest
from pathlib import Path

from lecture_processor.config import (
    AIModelProvider,
    AIProviderName,
    AudioEnhancementMode,
    AudioQuality,
    BatchConfig,
    RecordingSpeed,
    TranscriptionQuality,
)
from lecture_processor.profiles import QUALITY_PROFILE_ID
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

    def test_validates_ai_concurrency_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = BatchConfig(
                input_dir=folder,
                output_dir=folder / "out",
                gemini_max_concurrency=13,
            )

            with self.assertRaises(LectureProcessorError):
                config.validate()

    def test_defaults_match_app_recommendations(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = BatchConfig(input_dir=folder, output_dir=folder / "out")

            self.assertIs(config.audio_quality, AudioQuality.HIGH)
            self.assertIs(config.audio_enhancement, AudioEnhancementMode.NONE)
            self.assertEqual(config.transcription_profile, QUALITY_PROFILE_ID)
            self.assertIs(config.transcription_quality, TranscriptionQuality.ACCURATE)
            self.assertEqual(config.whisper_model, "medium.en")
            self.assertEqual(config.gemini_max_concurrency, 3)
            self.assertEqual(config.mlx_text_base_url, "http://192.168.86.101:1234/v1")
            self.assertEqual(config.mlx_vision_base_url, "http://192.168.86.101:1234/v1")
            self.assertEqual(config.mlx_request_timeout_seconds, 120)
            self.assertEqual(
                config.ai_model_routing,
                {
                    "overview": {"provider": "mlx-text", "model": "default"},
                    "transcript": {"provider": "mlx-text", "model": "default"},
                    "slides": {"provider": "mlx-vision", "model": "default"},
                    "resources": {"provider": "mlx-text", "model": "default"},
                },
            )

    def test_ai_route_helpers_detect_gemini_and_local_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = BatchConfig(
                input_dir=folder,
                output_dir=folder / "out",
                ai_provider=AIProviderName.GEMINI,
                ai_overview_provider=AIModelProvider.GEMINI,
                ai_transcript_provider=AIModelProvider.MLX_TEXT,
                ai_resources_provider=AIModelProvider.OFF,
            )
            config.validate()

            self.assertTrue(config.ai_uses_gemini)
            self.assertFalse(config.ai_uses_local_stub)
            self.assertTrue(config.ai_uses_mlx)
            self.assertEqual(config.ai_step_model("transcript"), "default")
            self.assertEqual(config.ai_step_model("resources"), "off")


if __name__ == "__main__":
    unittest.main()
