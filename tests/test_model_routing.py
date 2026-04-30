import json
import tempfile
import unittest
from pathlib import Path

from lecture_processor.ai.enrichment import enrich_lecture_artifact
from lecture_processor.ai.model_routing import routed_analyze_lecture, uses_experimental_routing
from lecture_processor.ai.providers.base import AnalyzeLectureRequest, AnalyzeLectureResponse
from lecture_processor.config import AIModelProvider, AIProviderName, BatchConfig


class ModelRoutingTests(unittest.TestCase):
    def test_all_local_routes_enrich_without_gemini_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            lecture_dir = Path(tmp)
            lecture_json = lecture_dir / "lecture.json"
            lecture_json.write_text(json.dumps(_artifact()), encoding="utf-8")
            config = BatchConfig(
                input_dir=lecture_dir,
                output_dir=lecture_dir,
                ai_provider=AIProviderName.GEMINI,
                ai_overview_provider=AIModelProvider.LOCAL_STUB,
                ai_transcript_provider=AIModelProvider.LOCAL_STUB,
                ai_slides_provider=AIModelProvider.LOCAL_STUB,
                ai_resources_provider=AIModelProvider.LOCAL_STUB,
            )
            config.validate()

            artifact = enrich_lecture_artifact(lecture_json, config)

            enrichment = artifact["enrichment"]
            self.assertEqual(enrichment["provider"], "experimental-routing")
            self.assertEqual(enrichment["model_routing"]["overview"]["provider"], "local-stub")
            self.assertIn("local stub", " ".join(enrichment["warnings"]).lower())
            self.assertGreater(enrichment["input_token_estimate"], 0)

    def test_mixed_routes_only_ask_gemini_for_gemini_steps(self):
        request = AnalyzeLectureRequest(
            lecture_id="lecture",
            transcript_text="hello lecture",
            segments=[{"id": 1, "text": "hello lecture"}],
            slides=[{"id": 1, "linked_segment_ids": [1]}],
            duration_minutes=2.0,
        )
        config = BatchConfig(
            input_dir=Path("."),
            output_dir=Path("."),
            ai_provider=AIProviderName.GEMINI,
            ai_overview_provider=AIModelProvider.LOCAL_STUB,
            ai_transcript_provider=AIModelProvider.GEMINI,
            ai_slides_provider=AIModelProvider.OFF,
            ai_resources_provider=AIModelProvider.GEMINI,
        )
        provider = FakeGeminiProvider()

        response = routed_analyze_lecture(request, config, gemini_provider=provider)

        self.assertTrue(uses_experimental_routing(config))
        self.assertFalse(provider.kwargs["include_overview"])
        self.assertTrue(provider.kwargs["include_transcript"])
        self.assertFalse(provider.kwargs["include_slides"])
        self.assertTrue(provider.kwargs["include_resources"])
        self.assertEqual(provider.kwargs["overview_override"]["title"], "Hello Lecture")
        self.assertEqual(response.formatted_transcript, "Gemini transcript")
        self.assertEqual(response.resources[0]["title"], "Gemini resource")
        self.assertIn("local-stub", response.slide_analysis[0]["tags"])


class FakeGeminiProvider:
    def __init__(self):
        self.kwargs = {}

    def analyze_lecture_chunked(self, request, **kwargs):
        self.kwargs = kwargs
        return AnalyzeLectureResponse(
            title="Gemini title",
            executive_summary="Gemini summary",
            formatted_transcript="Gemini transcript",
            outline=[{"id": 1, "heading": "Gemini", "slide_ids": [1]}],
            slide_analysis=[
                {
                    "slide_id": 1,
                    "descriptive_filename": "slide.png",
                    "caption": "Gemini caption",
                    "summary": "Gemini slide",
                    "tags": ["gemini"],
                    "instructor_commentary": "Gemini notes",
                }
            ],
            resources=[
                {
                    "title": "Gemini resource",
                    "url": "https://example.com",
                    "summary": "A resource.",
                    "source_quality": "medium",
                }
            ],
            input_token_estimate=10,
            output_token_estimate=5,
            raw_response_id="fake",
            warnings=[],
        )


def _artifact():
    return {
        "lecture_id": "lecture",
        "source": {"filename": "lecture.mov"},
        "media": {"duration_seconds": 120},
        "transcript": {
            "text": "hello lecture",
            "segments": [{"id": 1, "start": 0, "end": 1, "text": "hello lecture"}],
        },
        "slides": [{"id": 1, "linked_segment_ids": [1]}],
        "processing": {"status": "completed"},
        "enrichment": None,
    }


if __name__ == "__main__":
    unittest.main()
