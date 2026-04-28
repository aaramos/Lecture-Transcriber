import tempfile
import unittest
import json
from pathlib import Path

from lecture_processor.ai.providers.base import AnalyzeLectureRequest
from lecture_processor.ai.providers.gemini import (
    GeminiProvider,
    _contents_for_request,
    _generate_content_config,
    _json_from_response_text,
    _response_schema,
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

    def test_response_schema_requires_formatted_transcript(self):
        schema = _response_schema()

        self.assertIn("formatted_transcript", schema["required"])
        self.assertEqual("string", schema["properties"]["formatted_transcript"]["type"])

    def test_json_parser_accepts_markdown_fenced_json(self):
        payload = _json_from_response_text('```json\n{"title": "Lecture"}\n```')

        self.assertEqual({"title": "Lecture"}, payload)

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
            self.assertEqual(["1-2-3-4-5", "6"], list(cache["chunks"]["slides"].keys()))
            slide_calls = [
                call for call in provider._client.models.calls if "slide-by-slide study notes" in call["prompt"]
            ]
            self.assertEqual(2, len(slide_calls))
            self.assertIn("resources", [event["step"] for event in events])

            resumed_provider = object.__new__(GeminiProvider)
            resumed_provider.model = "gemini-test"
            resumed_provider._client = FakeClient()
            resumed_provider._types = FakeTypes

            resumed = resumed_provider.analyze_lecture_chunked(request, cache_path=cache_path)

            self.assertEqual("Chunked Lecture", resumed.title)
            self.assertEqual(0, len(resumed_provider._client.models.calls))


if __name__ == "__main__":
    unittest.main()
