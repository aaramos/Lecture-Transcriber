import json
import tempfile
import unittest
from pathlib import Path

from lecture_processor.ai.providers.base import AnalyzeLectureRequest
from lecture_processor.ai.providers.mlx_openai import MLXTextProvider, MLXVisionProvider, _Usage


class MLXOpenAIProviderTests(unittest.TestCase):
    def test_text_provider_parses_openai_compatible_overview_response(self):
        provider = MLXTextProvider(base_url="http://local.test/v1", model="default")
        provider._client = FakeClient(
            {
                "title": "Local Model Lecture",
                "executive_summary": "A local summary.",
                "outline": [{"id": 1, "heading": "Intro", "slide_ids": [1]}],
                "warnings": [],
            }
        )

        overview = provider.analyze_overview(_request())

        self.assertEqual(overview["title"], "Local Model Lecture")
        self.assertEqual(overview["outline"][0]["heading"], "Intro")
        self.assertGreater(overview["input_tokens"], 0)
        self.assertIn("local MLX text", " ".join(overview["warnings"]))

    def test_vision_provider_sends_slide_image_as_data_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slide_path = root / "slide.png"
            slide_path.write_bytes(b"not-a-real-png-but-valid-test-bytes")
            provider = MLXVisionProvider(base_url="http://local.test/v1", model="default")
            provider._client = FakeClient(
                {
                    "slide_analysis": [
                        {
                            "slide_id": 1,
                            "descriptive_filename": "slide_0001.png",
                            "caption": "A title slide.",
                            "summary": "The slide introduces the lecture.",
                            "tags": ["intro"],
                            "instructor_commentary": "The instructor opens the lecture.",
                        }
                    ],
                    "warnings": [],
                }
            )

            result = provider.analyze_slides(_request(lecture_dir=root))

            self.assertEqual(result["slide_analysis"][0]["caption"], "A title slide.")
            messages = provider._client.messages
            content = messages[1]["content"]
            image_items = [item for item in content if item.get("type") == "image_url"]
            self.assertEqual(len(image_items), 1)
            self.assertTrue(image_items[0]["image_url"]["url"].startswith("data:image/png;base64,"))


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.messages = None

    def chat_json(self, messages, *, max_tokens, temperature, context):
        self.messages = messages
        return self.payload, _Usage(input_tokens=12, output_tokens=max(1, len(json.dumps(self.payload)) // 4))


def _request(*, lecture_dir=None):
    return AnalyzeLectureRequest(
        lecture_id="lecture",
        transcript_text="hello local models",
        segments=[{"id": 1, "text": "hello local models"}],
        slides=[{"id": 1, "relative_path": "slide.png", "linked_segment_ids": [1]}],
        duration_minutes=2.0,
        lecture_dir=lecture_dir,
    )


if __name__ == "__main__":
    unittest.main()
