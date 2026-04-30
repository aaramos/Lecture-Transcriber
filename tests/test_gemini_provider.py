import tempfile
import threading
import time
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from lecture_processor.ai.providers.base import AnalyzeLectureRequest
from lecture_processor.ai.providers.gemini import (
    GeminiProvider,
    _contents_for_request,
    _filter_slides_for_gemini,
    _generate_content_config,
    _generate_content_with_retry,
    _json_from_response_text,
    _response_schema,
    _slide_batch_response_schema,
    _slide_batches,
    _transcript_chunks,
)


class FakeGoogleSearch:
    pass


class FakeTool:
    def __init__(self, *, google_search):
        self.google_search = google_search


class FakeGenerateContentConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakePart:
    @classmethod
    def from_text(cls, *, text):
        return {"kind": "text", "text": text}

    @classmethod
    def from_bytes(cls, *, data, mime_type):
        return {"kind": "bytes", "data": data, "mime_type": mime_type}


class FakeContent:
    def __init__(self, *, role, parts):
        self.role = role
        self.parts = parts


class FakeTypes:
    GoogleSearch = FakeGoogleSearch
    Tool = FakeTool
    GenerateContentConfig = FakeGenerateContentConfig
    Part = FakePart
    Content = FakeContent


class FakeUsage:
    prompt_token_count = 10
    candidates_token_count = 5


class FakeResponse:
    def __init__(self, payload):
        self.text = json.dumps(payload)
        self.usage_metadata = FakeUsage()


class FakeTextResponse:
    def __init__(self, text):
        self.text = text
        self.usage_metadata = FakeUsage()


class FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, *, model, contents, config):
        prompt = _prompt_from_contents(contents)
        self.calls.append({"model": model, "prompt": prompt, "contents": contents, "config": config})
        if "high-level study structure" in prompt:
            return FakeResponse(
                {
                    "title": "Chunked Lecture",
                    "executive_summary": "A concise chunked summary.",
                    "outline": [{"id": 1, "heading": "Opening", "slide_ids": [1, 2]}],
                    "warnings": [],
                }
            )
        if "lightly editing one chunk" in prompt:
            return FakeResponse({"formatted_transcript": "Edited transcript chunk.", "warnings": []})
        if "slide-by-slide study notes" in prompt:
            slide_ids = [int(line.split(":", 1)[0].split()[-1]) for line in prompt.splitlines() if line.startswith("- Slide ")]
            return FakeResponse(
                {
                    "slide_analysis": [
                        {
                            "slide_id": slide_id,
                            "descriptive_filename": f"slide_{slide_id:04d}.png",
                            "caption": f"Caption {slide_id}",
                            "summary": f"Summary {slide_id}",
                            "tags": ["lecture"],
                            "instructor_commentary": f"Commentary {slide_id}",
                        }
                        for slide_id in slide_ids
                    ],
                    "warnings": [],
                }
            )
        if "Google Search grounding" in prompt:
            return FakeResponse(
                {
                    "resources": [
                        {
                            "title": "Resource",
                            "url": "https://example.com",
                            "summary": "Useful source.",
                            "source_quality": "high",
                        }
                    ],
                    "warnings": [],
                }
            )
        raise AssertionError(f"Unexpected prompt: {prompt[:120]}")


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


class OverlapFakeModels(FakeModels):
    def __init__(self):
        super().__init__()
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def generate_content(self, *, model, contents, config):
        prompt = _prompt_from_contents(contents)
        if "high-level study structure" not in prompt:
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.03)
                return super().generate_content(model=model, contents=contents, config=config)
            finally:
                with self.lock:
                    self.active -= 1
        return super().generate_content(model=model, contents=contents, config=config)


class OverlapFakeClient:
    def __init__(self):
        self.models = OverlapFakeModels()


class FlakyModels:
    def __init__(self):
        self.calls = 0

    def generate_content(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            raise Exception("429 rate limit")
        return "ok"


class ResourceRepairFakeModels(FakeModels):
    def generate_content(self, *, model, contents, config):
        prompt = _prompt_from_contents(contents)
        if "Google Search grounding" in prompt:
            self.calls.append({"model": model, "prompt": prompt, "contents": contents, "config": config})
            return FakeTextResponse(
                '{"resources":[{"title":"Resource","url":"https://example.com","summary":"line one\nline two","source_quality":"high",}], "warnings":[]}'
            )
        return super().generate_content(model=model, contents=contents, config=config)


class ResourceRepairFakeClient:
    def __init__(self):
        self.models = ResourceRepairFakeModels()


class ResourceRetryFakeModels(FakeModels):
    def generate_content(self, *, model, contents, config):
        prompt = _prompt_from_contents(contents)
        if "Google Search grounding" in prompt:
            self.calls.append({"model": model, "prompt": prompt, "contents": contents, "config": config})
            if "previous resource attempt" in prompt.lower():
                return FakeResponse(
                    {
                        "resources": [
                            {
                                "title": f"Resource {index}",
                                "url": f"https://example.com/{index}",
                                "summary": f"Useful source {index}.",
                                "source_quality": "high",
                            }
                            for index in range(1, 4)
                        ],
                        "warnings": [],
                    }
                )
            return FakeResponse(
                {
                    "resources": [
                        {
                            "title": "Incomplete",
                            "url": "https://example.com/incomplete",
                            "summary": "",
                            "source_quality": "medium",
                        }
                    ],
                    "warnings": [],
                }
            )
        return super().generate_content(model=model, contents=contents, config=config)


class ResourceRetryFakeClient:
    def __init__(self):
        self.models = ResourceRetryFakeModels()


def _prompt_from_contents(contents):
    if isinstance(contents, str):
        return contents
    return contents[0].parts[0]["text"]


class GeminiProviderTests(unittest.TestCase):
    def test_generate_content_config_enables_google_search_grounding(self):
        config = _generate_content_config(FakeTypes)

        self.assertEqual(1, len(config.kwargs["tools"]))
        self.assertIsInstance(config.kwargs["tools"][0].google_search, FakeGoogleSearch)
        self.assertNotIn("response_mime_type", config.kwargs)
        self.assertNotIn("response_json_schema", config.kwargs)

    def test_generate_content_config_sets_max_output_tokens(self):
        config = _generate_content_config(FakeTypes, use_google_search=False, max_output_tokens=8192)

        self.assertEqual(8192, config.kwargs["max_output_tokens"])

    def test_generate_content_config_uses_structured_json_without_search(self):
        schema = {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}

        config = _generate_content_config(FakeTypes, use_google_search=False, response_schema=schema)

        self.assertEqual("application/json", config.kwargs["response_mime_type"])
        self.assertEqual(schema, config.kwargs["response_json_schema"])
        self.assertNotIn("tools", config.kwargs)

    def test_generate_content_config_does_not_mix_search_with_structured_json(self):
        schema = {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}

        config = _generate_content_config(FakeTypes, use_google_search=True, response_schema=schema)

        self.assertIn("tools", config.kwargs)
        self.assertNotIn("response_mime_type", config.kwargs)
        self.assertNotIn("response_schema", config.kwargs)
        self.assertNotIn("response_json_schema", config.kwargs)

    def test_slide_batches_use_ten_slides(self):
        slides = [{"id": index} for index in range(1, 22)]

        batches = _slide_batches(slides)

        self.assertEqual([10, 10, 1], [len(batch) for batch in batches])

    def test_transcript_chunks_target_six_thousand_characters(self):
        request = AnalyzeLectureRequest(
            lecture_id="lecture",
            transcript_text="",
            segments=[
                {"id": index, "start": float(index), "end": float(index + 1), "text": "word " * 240}
                for index in range(1, 8)
            ],
            slides=[],
            duration_minutes=1.0,
        )

        chunks = _transcript_chunks(request)

        self.assertEqual(2, len(chunks))
        self.assertEqual([1, 2], [chunk["index"] for chunk in chunks])
        self.assertEqual([2, 2], [chunk["total"] for chunk in chunks])

    def test_generate_content_retries_transient_rate_limits(self):
        models = FlakyModels()

        with patch("lecture_processor.ai.providers.gemini.time.sleep") as sleep:
            result = _generate_content_with_retry(models, model="gemini-test", contents="prompt", config=None)

        self.assertEqual("ok", result)
        self.assertEqual(2, models.calls)
        sleep.assert_called_once()

    def test_response_schema_requires_formatted_transcript(self):
        schema = _response_schema()

        self.assertIn("formatted_transcript", schema["required"])
        self.assertEqual("string", schema["properties"]["formatted_transcript"]["type"])

    def test_slide_schema_uses_gemini_nullable_fields(self):
        item = _slide_batch_response_schema()["properties"]["slide_analysis"]["items"]

        self.assertEqual("string", item["properties"]["caption"]["type"])
        self.assertTrue(item["properties"]["caption"]["nullable"])
        self.assertEqual("string", item["properties"]["descriptive_filename"]["type"])
        self.assertTrue(item["properties"]["descriptive_filename"]["nullable"])

    def test_json_parser_accepts_markdown_fenced_json(self):
        payload = _json_from_response_text('```json\n{"title": "Lecture"}\n```')

        self.assertEqual({"title": "Lecture"}, payload)

    def test_json_parser_repairs_common_gemini_json_issues(self):
        payload = _json_from_response_text('{"resources":[{"title":"A","summary":"line one\nline two",}],}')

        self.assertEqual({"resources": [{"title": "A", "summary": "line one line two"}]}, payload)

    def test_contents_include_slide_images_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slides_dir = root / "slides"
            slides_dir.mkdir()
            (slides_dir / "slide_0001_00-00-00.png").write_bytes(b"png-bytes")
            request = AnalyzeLectureRequest(
                lecture_id="lecture",
                transcript_text="hello",
                segments=[],
                slides=[
                    {
                        "id": 1,
                        "filename": "slide_0001_00-00-00.png",
                        "relative_path": "slides/slide_0001_00-00-00.png",
                    }
                ],
                duration_minutes=1.0,
                lecture_dir=root,
            )

            contents = _contents_for_request(request, "prompt", FakeTypes)

            self.assertEqual(1, len(contents))
            self.assertEqual("user", contents[0].role)
            self.assertEqual({"kind": "text", "text": "prompt"}, contents[0].parts[0])
            self.assertIn("Slide 1", contents[0].parts[1]["text"])
            self.assertEqual("image/png", contents[0].parts[2]["mime_type"])
            self.assertEqual(b"png-bytes", contents[0].parts[2]["data"])

    def test_contents_downscale_slide_images_to_cached_webp(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slides_dir = root / "slides"
            slides_dir.mkdir()
            image_path = slides_dir / "slide_0001_00-00-00.png"
            Image.new("RGB", (1600, 900), color=(255, 255, 255)).save(image_path)
            request = AnalyzeLectureRequest(
                lecture_id="lecture",
                transcript_text="hello",
                segments=[],
                slides=[
                    {
                        "id": 1,
                        "filename": image_path.name,
                        "relative_path": f"slides/{image_path.name}",
                    }
                ],
                duration_minutes=1.0,
                lecture_dir=root,
            )

            contents = _contents_for_request(request, "prompt", FakeTypes)

            self.assertEqual("image/webp", contents[0].parts[2]["mime_type"])
            self.assertTrue((slides_dir / ".cache" / "slide_0001_00-00-00.webp").exists())

    def test_slide_prefilter_skips_blank_and_keeps_last_duplicate(self):
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            self.skipTest("Pillow is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slides_dir = root / "slides"
            slides_dir.mkdir()
            Image.new("RGB", (320, 180), color=(255, 255, 255)).save(slides_dir / "slide_0001.png")
            for filename in ("slide_0002.png", "slide_0003.png"):
                image = Image.new("RGB", (320, 180), color=(255, 255, 255))
                draw = ImageDraw.Draw(image)
                draw.rectangle((20, 20, 220, 100), fill=(0, 0, 0))
                draw.rectangle((40, 125, 260, 145), fill=(80, 80, 80))
                image.save(slides_dir / filename)
            slides = [
                {"id": index, "filename": f"slide_{index:04d}.png", "relative_path": f"slides/slide_{index:04d}.png"}
                for index in range(1, 4)
            ]
            request = AnalyzeLectureRequest(
                lecture_id="lecture",
                transcript_text="hello",
                segments=[],
                slides=slides,
                duration_minutes=1.0,
                lecture_dir=root,
            )

            kept, filtered = _filter_slides_for_gemini(request)

            self.assertEqual([3], [slide["id"] for slide in kept])
            self.assertEqual("blank", filtered[1]["status"])
            self.assertEqual("build_precursor", filtered[2]["status"])
            self.assertEqual(3, filtered[2]["parent_slide_id"])
            self.assertEqual({"status": "skipped", "reason": "blank"}, slides[0]["filter_status"])
            self.assertEqual(
                {"status": "merged", "reason": "near_duplicate", "parent_slide_id": 3},
                slides[1]["filter_status"],
            )
            self.assertEqual({"status": "sent_to_gemini"}, slides[2]["filter_status"])

    def test_contents_fall_back_to_text_when_no_images_are_available(self):
        request = AnalyzeLectureRequest(
            lecture_id="lecture",
            transcript_text="hello",
            segments=[],
            slides=[],
            duration_minutes=1.0,
        )

        self.assertEqual("prompt", _contents_for_request(request, "prompt", FakeTypes))

    def test_chunked_analysis_saves_partials_and_batches_slides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slides_dir = root / "slides"
            slides_dir.mkdir()
            slides = []
            for slide_id in range(1, 7):
                filename = f"slide_{slide_id:04d}.png"
                (slides_dir / filename).write_bytes(b"png")
                slides.append(
                    {
                        "id": slide_id,
                        "filename": filename,
                        "relative_path": f"slides/{filename}",
                        "timestamp_seconds": slide_id * 10.0,
                        "linked_segment_ids": [slide_id - 1],
                    }
                )
            request = AnalyzeLectureRequest(
                lecture_id="lecture",
                transcript_text="hello lecture " * 20,
                segments=[
                    {"id": index, "start": index * 3.0, "end": index * 3.0 + 2.0, "text": f"segment {index}"}
                    for index in range(6)
                ],
                slides=slides,
                duration_minutes=12.0,
                lecture_dir=root,
            )
            provider = object.__new__(GeminiProvider)
            provider.model = "gemini-test"
            provider.max_concurrency = 6
            provider._client = FakeClient()
            provider._types = FakeTypes
            events = []
            cache_path = root / ".enrichment_partial.json"

            response = provider.analyze_lecture_chunked(request, cache_path=cache_path, progress_callback=events.append)

            self.assertEqual("Chunked Lecture", response.title)
            self.assertEqual(6, len(response.slide_analysis))
            self.assertEqual("Edited transcript chunk.", response.formatted_transcript)
            self.assertEqual(1, len(response.resources))
            self.assertTrue(cache_path.exists())
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertIn("overview", cache["chunks"])
            self.assertEqual(["1-2-3-4-5-6"], list(cache["chunks"]["slides"].keys()))
            slide_calls = [
                call for call in provider._client.models.calls if "slide-by-slide study notes" in call["prompt"]
            ]
            self.assertEqual(1, len(slide_calls))
            transcript_calls = [
                call for call in provider._client.models.calls if "lightly editing one chunk" in call["prompt"]
            ]
            resource_calls = [
                call for call in provider._client.models.calls if "Google Search grounding" in call["prompt"]
            ]
            self.assertEqual("application/json", transcript_calls[0]["config"].kwargs["response_mime_type"])
            self.assertIn("response_json_schema", transcript_calls[0]["config"].kwargs)
            self.assertEqual("application/json", slide_calls[0]["config"].kwargs["response_mime_type"])
            self.assertIn("response_json_schema", slide_calls[0]["config"].kwargs)
            self.assertNotIn("response_mime_type", resource_calls[0]["config"].kwargs)
            self.assertIn("resources", [event["step"] for event in events])
            self.assertEqual(response.input_token_estimate, events[-1]["input_tokens"])
            self.assertEqual(response.output_token_estimate, events[-1]["output_tokens"])

            resumed_provider = object.__new__(GeminiProvider)
            resumed_provider.model = "gemini-test"
            resumed_provider.max_concurrency = 6
            resumed_provider._client = FakeClient()
            resumed_provider._types = FakeTypes

            resumed = resumed_provider.analyze_lecture_chunked(request, cache_path=cache_path)

            self.assertEqual("Chunked Lecture", resumed.title)
            self.assertEqual(0, len(resumed_provider._client.models.calls))

    def test_chunked_analysis_repairs_grounded_resource_json(self):
        request = AnalyzeLectureRequest(
            lecture_id="lecture",
            transcript_text="hello lecture " * 20,
            segments=[{"id": 1, "start": 0.0, "end": 2.0, "text": "hello lecture"}],
            slides=[],
            duration_minutes=1.0,
        )
        provider = object.__new__(GeminiProvider)
        provider.model = "gemini-test"
        provider.max_concurrency = 6
        provider._client = ResourceRepairFakeClient()
        provider._types = FakeTypes

        response = provider.analyze_lecture_chunked(request)

        self.assertEqual(1, len(response.resources))
        self.assertEqual("line one line two", response.resources[0]["summary"])

    def test_chunked_analysis_retries_when_resources_are_incomplete(self):
        request = AnalyzeLectureRequest(
            lecture_id="lecture",
            transcript_text="hello lecture " * 20,
            segments=[{"id": 1, "start": 0.0, "end": 2.0, "text": "hello lecture"}],
            slides=[],
            duration_minutes=1.0,
        )
        provider = object.__new__(GeminiProvider)
        provider.model = "gemini-test"
        provider.max_concurrency = 6
        provider._client = ResourceRetryFakeClient()
        provider._types = FakeTypes

        response = provider.analyze_lecture_chunked(request)

        resource_calls = [
            call for call in provider._client.models.calls if "Google Search grounding" in call["prompt"]
        ]
        self.assertEqual(2, len(resource_calls))
        self.assertEqual(3, len(response.resources))
        self.assertEqual(["Resource 1", "Resource 2", "Resource 3"], [item["title"] for item in response.resources])

    def test_chunked_analysis_backfills_filtered_slides(self):
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            self.skipTest("Pillow is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slides_dir = root / "slides"
            slides_dir.mkdir()
            Image.new("RGB", (320, 180), color=(255, 255, 255)).save(slides_dir / "slide_0001.png")
            for filename in ("slide_0002.png", "slide_0003.png"):
                image = Image.new("RGB", (320, 180), color=(255, 255, 255))
                draw = ImageDraw.Draw(image)
                draw.rectangle((20, 20, 220, 100), fill=(0, 0, 0))
                draw.rectangle((40, 125, 260, 145), fill=(80, 80, 80))
                image.save(slides_dir / filename)
            slides = [
                {
                    "id": index,
                    "filename": f"slide_{index:04d}.png",
                    "relative_path": f"slides/slide_{index:04d}.png",
                    "timestamp_seconds": index * 10.0,
                    "linked_segment_ids": [index],
                }
                for index in range(1, 4)
            ]
            request = AnalyzeLectureRequest(
                lecture_id="lecture",
                transcript_text="hello lecture " * 20,
                segments=[
                    {"id": index, "start": index * 3.0, "end": index * 3.0 + 2.0, "text": f"segment {index}"}
                    for index in range(1, 4)
                ],
                slides=slides,
                duration_minutes=1.0,
                lecture_dir=root,
            )
            provider = object.__new__(GeminiProvider)
            provider.model = "gemini-test"
            provider.max_concurrency = 6
            provider._client = FakeClient()
            provider._types = FakeTypes

            response = provider.analyze_lecture_chunked(request)

            self.assertEqual([1, 2, 3], [item["slide_id"] for item in response.slide_analysis])
            self.assertIsNone(response.slide_analysis[0]["caption"])
            self.assertIn("Blank or transition", response.slide_analysis[0]["summary"])
            self.assertIn("build-precursor", response.slide_analysis[1]["tags"])
            slide_calls = [
                call for call in provider._client.models.calls if "slide-by-slide study notes" in call["prompt"]
            ]
            self.assertEqual(1, len(slide_calls))
            self.assertNotIn("- Slide 1:", slide_calls[0]["prompt"])
            self.assertNotIn("- Slide 2:", slide_calls[0]["prompt"])
            self.assertIn("- Slide 3:", slide_calls[0]["prompt"])
            self.assertTrue(any("Slide pre-filter skipped" in warning for warning in response.warnings))

    def test_chunked_analysis_parallelizes_post_overview_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slides_dir = root / "slides"
            slides_dir.mkdir()
            (slides_dir / "slide_0001.png").write_bytes(b"png")
            request = AnalyzeLectureRequest(
                lecture_id="lecture",
                transcript_text="hello lecture " * 20,
                segments=[{"id": 1, "start": 0.0, "end": 2.0, "text": "hello lecture"}],
                slides=[
                    {
                        "id": 1,
                        "filename": "slide_0001.png",
                        "relative_path": "slides/slide_0001.png",
                        "timestamp_seconds": 1.0,
                        "linked_segment_ids": [1],
                    }
                ],
                duration_minutes=1.0,
                lecture_dir=root,
            )
            provider = object.__new__(GeminiProvider)
            provider.model = "gemini-test"
            provider.max_concurrency = 3
            provider._client = OverlapFakeClient()
            provider._types = FakeTypes

            provider.analyze_lecture_chunked(request)

            self.assertGreater(provider._client.models.max_active, 1)


if __name__ == "__main__":
    unittest.main()
