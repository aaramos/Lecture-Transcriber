import tempfile
import unittest
from pathlib import Path

from lecture_processor.ai.providers.base import AnalyzeLectureRequest
from lecture_processor.ai.providers.gemini import (
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


if __name__ == "__main__":
    unittest.main()
