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
            self.assertEqual(config.mlx_text_base_url, "http://192.168.86.22:1234/v1")
            self.assertEqual(config.mlx_vision_base_url, "http://192.168.86.22:1234/v1")
            self.assertEqual(config.mlx_request_timeout_seconds, 120)
            self.assertEqual(config.min_duration_seconds, 30.0)
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

    def test_min_duration_defaults_to_30_and_is_overrideable(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = BatchConfig(input_dir=folder, output_dir=folder / "out")
            self.assertEqual(config.min_duration_seconds, 30.0)
            config.validate()

            custom = BatchConfig(
                input_dir=folder,
                output_dir=folder / "out",
                min_duration_seconds=45.5,
            )
            custom.validate()
            self.assertEqual(custom.min_duration_seconds, 45.5)

    def test_ollama_cloud_default_url_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = BatchConfig(input_dir=folder, output_dir=folder / "out")
            self.assertEqual(config.ollama_cloud_base_url, "https://ollama.com/v1")
            config.validate()

            # Invalid URL is rejected
            bad = BatchConfig(
                input_dir=folder,
                output_dir=folder / "out",
                ollama_cloud_base_url="not-a-url",
            )
            with self.assertRaises(LectureProcessorError) as context:
                bad.validate()
            self.assertIn("Invalid URL for ollama_cloud_base_url", str(context.exception))

            # Valid override works
            custom = BatchConfig(
                input_dir=folder,
                output_dir=folder / "out",
                ollama_cloud_base_url="https://ollama.com/v1/custom",
            )
            custom.validate()
            self.assertEqual(custom.ollama_cloud_base_url, "https://ollama.com/v1/custom")

    def test_ollama_cloud_route_helpers_and_default_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = BatchConfig(
                input_dir=folder,
                output_dir=folder / "out",
                ai_provider=AIProviderName.GEMINI,
                ai_overview_provider=AIModelProvider.OLLAMA_CLOUD,
                ai_transcript_provider=AIModelProvider.OFF,
                ai_slides_provider=AIModelProvider.OFF,
                ai_resources_provider=AIModelProvider.OFF,
            )
            config.validate()

            self.assertTrue(config.ai_uses_ollama_cloud)
            self.assertFalse(config.ai_uses_mlx)
            self.assertEqual(config.ai_step_model("overview"), "llama3.3:70b")
            self.assertEqual(config.ai_step_model("transcript"), "off")

    def test_validates_mlx_url_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)

            # Invalid URL scheme
            config = BatchConfig(
                input_dir=folder,
                output_dir=folder / "out",
                mlx_text_base_url="not-a-url",
            )
            with self.assertRaises(LectureProcessorError) as context:
                config.validate()
            self.assertIn("Invalid URL for mlx_text_base_url", str(context.exception))

            # Missing netloc
            config2 = BatchConfig(
                input_dir=folder,
                output_dir=folder / "out",
                mlx_vision_base_url="http:///path/without/host",
            )
            with self.assertRaises(LectureProcessorError) as context:
                config2.validate()
            self.assertIn("Invalid URL for mlx_vision_base_url", str(context.exception))

            # Valid URL works
            config3 = BatchConfig(
                input_dir=folder,
                output_dir=folder / "out",
                mlx_text_base_url="https://localhost:1234",
            )
            config3.validate()


if __name__ == "__main__":
    unittest.main()
