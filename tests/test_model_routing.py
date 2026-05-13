import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from lecture_processor.ai.enrichment import enrich_lecture_artifact, enrich_lecture_artifacts_staged
from lecture_processor.ai.model_routing import (
    RouteAvailability,
    probe_lm_studio_model_availability,
    routed_analyze_lecture,
    routed_analyze_lectures_staged,
    uses_experimental_routing,
)
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
            self.assertEqual("1.1.0", enrichment["schema_version"])
            self.assertEqual(enrichment["provider"], "experimental-routing")
            self.assertEqual(enrichment["model_routing"]["overview"]["provider"], "local-stub")
            self.assertIn("local stub", " ".join(enrichment["warnings"]).lower())
            self.assertGreater(enrichment["input_token_estimate"], 0)
            self.assertIn("overview", enrichment["step_timings"])
            self.assertIn("transcript_cleanup", enrichment["step_timings"])

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

    def test_mlx_routes_are_experimental_and_skip_gemini_for_local_steps(self):
        request = AnalyzeLectureRequest(
            lecture_id="lecture",
            transcript_text="hello lecture",
            segments=[{"id": 1, "text": "hello lecture"}],
            slides=[],
            duration_minutes=2.0,
        )
        config = BatchConfig(
            input_dir=Path("."),
            output_dir=Path("."),
            ai_provider=AIProviderName.GEMINI,
            ai_overview_provider=AIModelProvider.MLX_TEXT,
            ai_transcript_provider=AIModelProvider.GEMINI,
            ai_slides_provider=AIModelProvider.OFF,
            ai_resources_provider=AIModelProvider.GEMINI,
        )
        provider = FakeGeminiProvider()

        with mock.patch(
            "lecture_processor.ai.model_routing.MLXTextProvider",
            return_value=FakeMLXTextProvider(),
        ):
            response = routed_analyze_lecture(request, config, gemini_provider=provider)

        self.assertTrue(uses_experimental_routing(config))
        self.assertFalse(provider.kwargs["include_overview"])
        self.assertTrue(provider.kwargs["include_transcript"])
        self.assertTrue(provider.kwargs["include_resources"])
        self.assertEqual(provider.kwargs["overview_override"]["title"], "MLX title")
        self.assertEqual(response.title, "MLX title")
        self.assertEqual(response.formatted_transcript, "Gemini transcript")

    def test_probe_resolves_defaults_and_marks_missing_role(self):
        config = BatchConfig(
            input_dir=Path("."),
            output_dir=Path("."),
            ai_provider=AIProviderName.GEMINI,
            ai_overview_provider=AIModelProvider.MLX_TEXT,
            ai_transcript_provider=AIModelProvider.MLX_TEXT,
            ai_slides_provider=AIModelProvider.MLX_VISION,
            ai_slides_model="missing-vision",
            ai_resources_provider=AIModelProvider.MLX_TEXT,
            ai_resources_model="text-model",
        )

        def fake_models(provider, _config):
            if provider is AIModelProvider.MLX_VISION:
                return ["vision-model"]
            return ["text-model"]

        with mock.patch("lecture_processor.ai.model_routing._list_models_for_provider", side_effect=fake_models):
            availability = probe_lm_studio_model_availability(config)

        self.assertEqual("text-model", availability.model_for_step("overview", config))
        self.assertEqual("text-model", availability.model_for_step("resources", config))
        self.assertIn("vision", availability.skipped_roles)
        self.assertIn("missing-vision", availability.skipped_roles["vision"])

    def test_staged_routing_groups_all_files_by_role(self):
        calls = []
        jobs = [
            ("a", _request("a")),
            ("b", _request("b")),
        ]
        config = BatchConfig(
            input_dir=Path("."),
            output_dir=Path("."),
            ai_provider=AIProviderName.GEMINI,
            ai_overview_provider=AIModelProvider.MLX_TEXT,
            ai_overview_model="text-model",
            ai_transcript_provider=AIModelProvider.MLX_TEXT,
            ai_transcript_model="text-model",
            ai_slides_provider=AIModelProvider.MLX_VISION,
            ai_slides_model="vision-model",
            ai_resources_provider=AIModelProvider.MLX_TEXT,
            ai_resources_model="resource-model",
        )

        with mock.patch(
            "lecture_processor.ai.model_routing.MLXTextProvider",
            side_effect=lambda **kwargs: RecordingTextProvider(calls, **kwargs),
        ), mock.patch(
            "lecture_processor.ai.model_routing.MLXVisionProvider",
            side_effect=lambda **kwargs: RecordingVisionProvider(calls, **kwargs),
        ):
            responses = routed_analyze_lectures_staged(
                jobs,
                config,
                availability=RouteAvailability(
                    resolved_step_models={
                        "overview": "text-model",
                        "transcript": "text-model",
                        "slides": "vision-model",
                        "resources": "resource-model",
                    }
                ),
            )

        self.assertEqual(["a", "b"], sorted(responses))
        self.assertEqual(
            [
                ("text-model", "overview", "a"),
                ("text-model", "overview", "b"),
                ("text-model", "transcript", "a"),
                ("text-model", "transcript", "b"),
                ("vision-model", "slides", "a"),
                ("vision-model", "slides", "b"),
                ("resource-model", "resources", "a"),
                ("resource-model", "resources", "b"),
            ],
            calls,
        )
        self.assertIn("overview", responses["a"].step_token_usage)
        self.assertIn("transcript_cleanup", responses["a"].step_token_usage)

    def test_staged_progress_reports_file_and_step_token_usage(self):
        calls = []
        events = []
        jobs = [
            ("a", _request("a")),
            ("b", _request("b")),
        ]
        config = BatchConfig(
            input_dir=Path("."),
            output_dir=Path("."),
            ai_provider=AIProviderName.GEMINI,
            ai_overview_provider=AIModelProvider.MLX_TEXT,
            ai_overview_model="text-model",
            ai_transcript_provider=AIModelProvider.MLX_TEXT,
            ai_transcript_model="text-model",
            ai_slides_provider=AIModelProvider.OFF,
            ai_resources_provider=AIModelProvider.OFF,
        )

        with mock.patch(
            "lecture_processor.ai.model_routing.MLXTextProvider",
            side_effect=lambda **kwargs: RecordingTextProvider(calls, **kwargs),
        ):
            routed_analyze_lectures_staged(
                jobs,
                config,
                progress_callback=events.append,
                availability=RouteAvailability(
                    resolved_step_models={
                        "overview": "text-model",
                        "transcript": "text-model",
                    }
                ),
            )

        a_events = [event for event in events if event["lecture_id"] == "a"]
        b_events = [event for event in events if event["lecture_id"] == "b"]
        self.assertEqual(1, a_events[0]["input_tokens"])
        self.assertEqual(1, a_events[0]["output_tokens"])
        self.assertEqual(1, b_events[0]["input_tokens"])
        self.assertEqual(1, b_events[0]["output_tokens"])
        self.assertEqual(1, a_events[0]["step_input_tokens"])
        self.assertEqual(1, a_events[0]["step_output_tokens"])
        self.assertEqual(1, a_events[1]["step_input_tokens"])
        self.assertEqual(1, a_events[1]["step_output_tokens"])
        self.assertEqual("success", a_events[0]["step_outcome"])
        self.assertEqual("Generated", a_events[0]["step_message"])

    def test_staged_progress_reports_remaining_slide_count(self):
        events = []
        request = AnalyzeLectureRequest(
            lecture_id="a",
            transcript_text="hello lecture",
            segments=[{"id": 1, "text": "hello lecture"}],
            slides=[
                {"id": 1, "linked_segment_ids": [1]},
                {"id": 2, "linked_segment_ids": [1]},
            ],
            duration_minutes=2.0,
        )
        config = BatchConfig(
            input_dir=Path("."),
            output_dir=Path("."),
            ai_provider=AIProviderName.GEMINI,
            ai_overview_provider=AIModelProvider.OFF,
            ai_transcript_provider=AIModelProvider.OFF,
            ai_slides_provider=AIModelProvider.LOCAL_STUB,
            ai_resources_provider=AIModelProvider.OFF,
        )

        routed_analyze_lectures_staged(
            [("a", request)],
            config,
            progress_callback=events.append,
        )

        slide_event = [event for event in events if event["step"] == "staged vision slides"][0]
        self.assertEqual(slide_event["input_slide_count"], 2)
        self.assertEqual(slide_event["slide_count_remaining"], 2)

    def test_staged_routing_reuses_text_model_for_resources(self):
        calls = []
        jobs = [
            ("a", _request("a")),
            ("b", _request("b")),
        ]
        config = BatchConfig(
            input_dir=Path("."),
            output_dir=Path("."),
            ai_provider=AIProviderName.GEMINI,
            ai_overview_provider=AIModelProvider.MLX_TEXT,
            ai_overview_model="shared-text",
            ai_transcript_provider=AIModelProvider.MLX_TEXT,
            ai_transcript_model="shared-text",
            ai_slides_provider=AIModelProvider.MLX_VISION,
            ai_slides_model="vision-model",
            ai_resources_provider=AIModelProvider.MLX_TEXT,
            ai_resources_model="shared-text",
        )

        with mock.patch(
            "lecture_processor.ai.model_routing.MLXTextProvider",
            side_effect=lambda **kwargs: RecordingTextProvider(calls, **kwargs),
        ), mock.patch(
            "lecture_processor.ai.model_routing.MLXVisionProvider",
            side_effect=lambda **kwargs: RecordingVisionProvider(calls, **kwargs),
        ):
            routed_analyze_lectures_staged(
                jobs,
                config,
                availability=RouteAvailability(
                    resolved_step_models={
                        "overview": "shared-text",
                        "transcript": "shared-text",
                        "slides": "vision-model",
                        "resources": "shared-text",
                    }
                ),
            )

        self.assertEqual(
            [
                ("shared-text", "overview", "a"),
                ("shared-text", "overview", "b"),
                ("shared-text", "transcript", "a"),
                ("shared-text", "transcript", "b"),
                ("shared-text", "resources", "a"),
                ("shared-text", "resources", "b"),
                ("vision-model", "slides", "a"),
                ("vision-model", "slides", "b"),
            ],
            calls,
        )

    def test_staged_routing_does_not_unload_between_local_model_steps(self):
        calls = []
        offloads = []
        config = BatchConfig(
            input_dir=Path("."),
            output_dir=Path("."),
            ai_provider=AIProviderName.GEMINI,
            ai_overview_provider=AIModelProvider.MLX_TEXT,
            ai_overview_model="text-model",
            ai_transcript_provider=AIModelProvider.MLX_TEXT,
            ai_transcript_model="text-model",
            ai_slides_provider=AIModelProvider.MLX_VISION,
            ai_slides_model="vision-model",
            ai_resources_provider=AIModelProvider.MLX_TEXT,
            ai_resources_model="resource-model",
        )
        availability = RouteAvailability(
            resolved_step_models={
                "overview": "text-model",
                "transcript": "text-model",
                "slides": "vision-model",
                "resources": "resource-model",
            }
        )

        with mock.patch(
            "lecture_processor.ai.model_routing.MLXTextProvider",
            side_effect=lambda **kwargs: RecordingTextProvider(calls, **kwargs),
        ), mock.patch(
            "lecture_processor.ai.model_routing.MLXVisionProvider",
            side_effect=lambda **kwargs: RecordingVisionProvider(calls, **kwargs),
        ), mock.patch(
            "lecture_processor.ai.model_routing._OpenAICompatibleClient",
            side_effect=lambda *args: RecordingUnloadClient(offloads, *args),
        ):
            routed_analyze_lectures_staged(
                [("a", _request("a"))],
                config,
                availability=availability,
            )

        self.assertEqual([], offloads)

    def test_staged_routing_keeps_loaded_model_between_role_switches(self):
        offloads = []
        config = BatchConfig(
            input_dir=Path("."),
            output_dir=Path("."),
            ai_provider=AIProviderName.GEMINI,
            ai_overview_provider=AIModelProvider.MLX_TEXT,
            ai_overview_model="shared-text",
            ai_transcript_provider=AIModelProvider.MLX_TEXT,
            ai_transcript_model="shared-text",
            ai_slides_provider=AIModelProvider.MLX_VISION,
            ai_slides_model="vision-model",
            ai_resources_provider=AIModelProvider.MLX_TEXT,
            ai_resources_model="shared-text",
        )
        availability = RouteAvailability(
            resolved_step_models={
                "overview": "shared-text",
                "transcript": "shared-text",
                "slides": "vision-model",
                "resources": "shared-text",
            }
        )

        with mock.patch(
            "lecture_processor.ai.model_routing.MLXTextProvider",
            side_effect=lambda **kwargs: RecordingTextProvider([], **kwargs),
        ), mock.patch(
            "lecture_processor.ai.model_routing.MLXVisionProvider",
            side_effect=lambda **kwargs: RecordingVisionProvider([], **kwargs),
        ), mock.patch(
            "lecture_processor.ai.model_routing._OpenAICompatibleClient",
            side_effect=lambda *args: RecordingUnloadClient(offloads, *args),
        ):
            routed_analyze_lectures_staged(
                [("a", _request("a"))],
                config,
                availability=availability,
            )

        self.assertEqual([], offloads)

    def test_single_lecture_routing_keeps_loaded_model_between_role_switches(self):
        offloads = []
        config = BatchConfig(
            input_dir=Path("."),
            output_dir=Path("."),
            ai_provider=AIProviderName.GEMINI,
            ai_overview_provider=AIModelProvider.MLX_TEXT,
            ai_overview_model="shared-text",
            ai_transcript_provider=AIModelProvider.MLX_TEXT,
            ai_transcript_model="shared-text",
            ai_slides_provider=AIModelProvider.MLX_VISION,
            ai_slides_model="vision-model",
            ai_resources_provider=AIModelProvider.MLX_TEXT,
            ai_resources_model="shared-text",
        )
        availability = RouteAvailability(
            resolved_step_models={
                "overview": "shared-text",
                "transcript": "shared-text",
                "slides": "vision-model",
                "resources": "shared-text",
            }
        )

        with mock.patch(
            "lecture_processor.ai.model_routing.MLXTextProvider",
            side_effect=lambda **kwargs: RecordingTextProvider([], **kwargs),
        ), mock.patch(
            "lecture_processor.ai.model_routing.MLXVisionProvider",
            side_effect=lambda **kwargs: RecordingVisionProvider([], **kwargs),
        ), mock.patch(
            "lecture_processor.ai.model_routing._OpenAICompatibleClient",
            side_effect=lambda *args: RecordingUnloadClient(offloads, *args),
        ):
            routed_analyze_lecture(
                _request("a"),
                config,
                availability=availability,
            )

        self.assertEqual([], offloads)

    def test_single_lecture_routing_does_not_unload_between_local_model_steps(self):
        calls = []
        offloads = []
        config = BatchConfig(
            input_dir=Path("."),
            output_dir=Path("."),
            ai_provider=AIProviderName.GEMINI,
            ai_overview_provider=AIModelProvider.MLX_TEXT,
            ai_overview_model="text-model",
            ai_transcript_provider=AIModelProvider.MLX_TEXT,
            ai_transcript_model="text-model",
            ai_slides_provider=AIModelProvider.MLX_VISION,
            ai_slides_model="vision-model",
            ai_resources_provider=AIModelProvider.MLX_TEXT,
            ai_resources_model="resource-model",
        )
        availability = RouteAvailability(
            resolved_step_models={
                "overview": "text-model",
                "transcript": "text-model",
                "slides": "vision-model",
                "resources": "resource-model",
            }
        )

        with mock.patch(
            "lecture_processor.ai.model_routing.MLXTextProvider",
            side_effect=lambda **kwargs: RecordingTextProvider(calls, **kwargs),
        ), mock.patch(
            "lecture_processor.ai.model_routing.MLXVisionProvider",
            side_effect=lambda **kwargs: RecordingVisionProvider(calls, **kwargs),
        ), mock.patch(
            "lecture_processor.ai.model_routing._OpenAICompatibleClient",
            side_effect=lambda *args: RecordingUnloadClient(offloads, *args),
        ):
            routed_analyze_lecture(
                _request("a"),
                config,
                availability=availability,
            )

        self.assertEqual([], offloads)

    def test_staged_routing_skips_unavailable_role_without_calling_provider(self):
        config = BatchConfig(
            input_dir=Path("."),
            output_dir=Path("."),
            ai_provider=AIProviderName.GEMINI,
            ai_overview_provider=AIModelProvider.LOCAL_STUB,
            ai_transcript_provider=AIModelProvider.LOCAL_STUB,
            ai_slides_provider=AIModelProvider.MLX_VISION,
            ai_slides_model="missing-vision",
            ai_resources_provider=AIModelProvider.LOCAL_STUB,
        )

        with mock.patch("lecture_processor.ai.model_routing.MLXVisionProvider") as provider:
            response = routed_analyze_lectures_staged(
                [("a", _request("a"))],
                config,
                availability=RouteAvailability(
                    skipped_roles={"vision": "LM Studio vision model unavailable: missing-vision"}
                ),
            )["a"]

        provider.assert_not_called()
        self.assertIn("missing-vision", " ".join(response.warnings))
        self.assertEqual(1, len(response.slide_analysis))

    def test_staged_enrichment_writes_upfront_model_warning_to_artifact(self):
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
                ai_slides_provider=AIModelProvider.MLX_VISION,
                ai_slides_model="missing-vision",
                ai_resources_provider=AIModelProvider.LOCAL_STUB,
            )
            availability = RouteAvailability(
                skipped_roles={"vision": "LM Studio vision model unavailable: missing-vision"}
            )

            with mock.patch(
                "lecture_processor.ai.enrichment.probe_lm_studio_model_availability",
                return_value=availability,
            ):
                artifacts = enrich_lecture_artifacts_staged([lecture_json], config)

            artifact = artifacts[lecture_json]
            self.assertIn("missing-vision", " ".join(artifact["processing"]["warnings"]))
            self.assertIn("missing-vision", " ".join(artifact["enrichment"]["warnings"]))

    def test_single_lecture_routing_clears_model_cache_before_each_step(self):
        config = BatchConfig(
            input_dir=Path("."),
            output_dir=Path("."),
            ai_provider=AIProviderName.GEMINI,
            ai_overview_provider=AIModelProvider.MLX_TEXT,
            ai_overview_model="text-model",
            ai_transcript_provider=AIModelProvider.MLX_TEXT,
            ai_transcript_model="text-model",
            ai_slides_provider=AIModelProvider.MLX_VISION,
            ai_slides_model="vision-model",
            ai_resources_provider=AIModelProvider.LOCAL_STUB,
        )
        availability = RouteAvailability(
            resolved_step_models={
                "overview": "text-model",
                "transcript": "text-model",
                "slides": "vision-model",
            }
        )

        with mock.patch(
            "lecture_processor.ai.model_routing.MLXTextProvider",
            side_effect=lambda **kwargs: RecordingTextProvider([], **kwargs),
        ), mock.patch(
            "lecture_processor.ai.model_routing.MLXVisionProvider",
            side_effect=lambda **kwargs: RecordingVisionProvider([], **kwargs),
        ), mock.patch(
            "lecture_processor.ai.model_routing._OpenAICompatibleClient",
            side_effect=lambda *args: RecordingUnloadClient([], *args),
        ), mock.patch(
            "lecture_processor.ai.model_routing.clear_lm_studio_model_verification_cache",
        ) as clear_cache:
            routed_analyze_lecture(
                _request("a"),
                config,
                availability=availability,
            )

        self.assertEqual(4, clear_cache.call_count)

    def test_staged_routing_clears_model_cache_at_model_stage_changes(self):
        jobs = [
            ("a", _request("a")),
            ("b", _request("b")),
        ]
        config = BatchConfig(
            input_dir=Path("."),
            output_dir=Path("."),
            ai_provider=AIProviderName.GEMINI,
            ai_overview_provider=AIModelProvider.MLX_TEXT,
            ai_overview_model="shared-text",
            ai_transcript_provider=AIModelProvider.MLX_TEXT,
            ai_transcript_model="shared-text",
            ai_slides_provider=AIModelProvider.MLX_VISION,
            ai_slides_model="vision-model",
            ai_resources_provider=AIModelProvider.MLX_TEXT,
            ai_resources_model="shared-text",
        )
        availability = RouteAvailability(
            resolved_step_models={
                "overview": "shared-text",
                "transcript": "shared-text",
                "slides": "vision-model",
                "resources": "shared-text",
            }
        )

        with mock.patch(
            "lecture_processor.ai.model_routing.MLXTextProvider",
            side_effect=lambda **kwargs: RecordingTextProvider([], **kwargs),
        ), mock.patch(
            "lecture_processor.ai.model_routing.MLXVisionProvider",
            side_effect=lambda **kwargs: RecordingVisionProvider([], **kwargs),
        ), mock.patch(
            "lecture_processor.ai.model_routing._OpenAICompatibleClient",
            side_effect=lambda *args: RecordingUnloadClient([], *args),
        ), mock.patch(
            "lecture_processor.ai.model_routing.clear_lm_studio_model_verification_cache",
        ) as clear_cache:
            routed_analyze_lectures_staged(
                jobs,
                config,
                availability=availability,
            )

        self.assertEqual(2, clear_cache.call_count)


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


class FakeMLXTextProvider:
    def analyze_overview(self, request):
        return {
            "title": "MLX title",
            "executive_summary": "MLX summary",
            "outline": [{"id": 1, "heading": "MLX", "slide_ids": []}],
            "warnings": [],
            "input_tokens": 10,
            "output_tokens": 5,
        }


class RecordingTextProvider:
    def __init__(self, calls, *, model, **_kwargs):
        self.calls = calls
        self.model = model

    def analyze_overview(self, request):
        self.calls.append((self.model, "overview", request.lecture_id))
        return {
            "title": f"{request.lecture_id} title",
            "executive_summary": f"{request.lecture_id} summary",
            "outline": [{"id": 1, "heading": "Intro", "slide_ids": [1]}],
            "warnings": [],
            "input_tokens": 1,
            "output_tokens": 1,
        }

    def analyze_transcript(self, request):
        self.calls.append((self.model, "transcript", request.lecture_id))
        return {
            "formatted_transcript": request.transcript_text,
            "warnings": [],
            "input_tokens": 1,
            "output_tokens": 1,
        }

    def analyze_resources(self, request, *, overview=None):
        self.calls.append((self.model, "resources", request.lecture_id))
        return {
            "resources": [],
            "warnings": [],
            "input_tokens": 1,
            "output_tokens": 1,
        }


class RecordingVisionProvider:
    def __init__(self, calls, *, model, **_kwargs):
        self.calls = calls
        self.model = model

    def analyze_slides(self, request):
        self.calls.append((self.model, "slides", request.lecture_id))
        return {
            "slide_analysis": [
                {
                    "slide_id": 1,
                    "descriptive_filename": "slide.png",
                    "caption": "A slide.",
                    "summary": "Slide summary.",
                    "tags": ["slide"],
                    "instructor_commentary": "Notes.",
                }
            ],
            "warnings": [],
            "input_tokens": 1,
            "output_tokens": 1,
        }


class RecordingUnloadClient:
    def __init__(self, offloads, base_url, model, timeout_seconds):
        self.offloads = offloads
        self.base_url = base_url
        self.model = model
        self.timeout_seconds = timeout_seconds

    def unload_model(self, model, context):
        self.offloads.append((self.base_url, model, context))
        return True


def _request(lecture_id="lecture"):
    return AnalyzeLectureRequest(
        lecture_id=lecture_id,
        transcript_text="hello lecture",
        segments=[{"id": 1, "text": "hello lecture"}],
        slides=[{"id": 1, "linked_segment_ids": [1]}],
        duration_minutes=2.0,
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
