import tempfile
import unittest
from pathlib import Path

from lecture_processor.config import AIModelProvider, BatchConfig
from lecture_processor.pipeline import _slide_extractor_for_config


class SmartSlidePipelineTests(unittest.TestCase):
    def test_slide_classifier_uses_selected_vision_model_when_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = BatchConfig(
                input_dir=Path(tmp),
                output_dir=Path(tmp) / "out",
                ai_slides_provider=AIModelProvider.MLX_VISION,
                ai_slides_model="gemma-vision",
                mlx_vision_base_url="http://localhost:1234/v1",
            )

            extractor = _slide_extractor_for_config(config)

        self.assertIsNotNone(extractor.classifier)
        self.assertEqual("gemma-vision", extractor.classifier.model)

    def test_slide_classifier_is_disabled_without_selected_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = BatchConfig(
                input_dir=Path(tmp),
                output_dir=Path(tmp) / "out",
                ai_slides_provider=AIModelProvider.MLX_VISION,
                ai_slides_model="",
            )

            extractor = _slide_extractor_for_config(config)

        self.assertIsNone(extractor.classifier)


if __name__ == "__main__":
    unittest.main()
