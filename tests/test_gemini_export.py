import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from lecture_processor.gemini_export import export_gemini_test_package


class GeminiExportTests(unittest.TestCase):
    def test_exports_prompt_and_processed_lecture_files_to_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            lecture_dir = Path(tmp) / "lecture"
            slides_dir = lecture_dir / "slides"
            slides_dir.mkdir(parents=True)
            (lecture_dir / "transcript.txt").write_text("hello transcript", encoding="utf-8")
            (lecture_dir / "transcript.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nhello\n", encoding="utf-8")
            (lecture_dir / "processing_log.txt").write_text("Complete\n", encoding="utf-8")
            (slides_dir / "slide_0001_00-00-01.png").write_bytes(b"not-really-png")
            (lecture_dir / "lecture.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "lecture_id": "lecture",
                        "source": {"filename": "lecture.mov"},
                        "media": {"duration_seconds": 120.0},
                        "transcript": {
                            "text": "hello transcript",
                            "segments": [{"id": 0, "start": 1.0, "end": 2.0, "text": "hello transcript"}],
                        },
                        "slides": [
                            {
                                "id": 1,
                                "relative_path": "slides/slide_0001_00-00-01.png",
                                "timestamp_seconds": 1.0,
                                "linked_segment_ids": [0],
                            }
                        ],
                        "processing": {},
                        "enrichment": None,
                    }
                ),
                encoding="utf-8",
            )

            zip_path = export_gemini_test_package(lecture_dir)

            self.assertTrue(zip_path.exists())
            with zipfile.ZipFile(zip_path) as archive:
                names = set(archive.namelist())
                self.assertIn("README.md", names)
                self.assertIn("prompt.md", names)
                self.assertIn("expected-response-schema.json", names)
                self.assertIn("lecture.json", names)
                self.assertIn("transcript.txt", names)
                self.assertIn("transcript.srt", names)
                self.assertIn("processing_log.txt", names)
                self.assertIn("slides/slide_0001_00-00-01.png", names)
                prompt = archive.read("prompt.md").decode("utf-8")
                self.assertIn("Return JSON only", prompt)
                self.assertIn("slide_analysis", prompt)
                self.assertIn("Google Search grounding", prompt)
                self.assertIn("Slide captions", prompt)
                self.assertIn("formatted_transcript", prompt)
                schema = json.loads(archive.read("expected-response-schema.json").decode("utf-8"))
                self.assertIn("formatted_transcript", schema["required"])
                slide_item = schema["properties"]["slide_analysis"]["items"]
                self.assertIn("caption", slide_item["required"])
                self.assertEqual(
                    ["high", "medium"],
                    schema["properties"]["resources"]["items"]["properties"]["source_quality"]["enum"],
                )

    def test_export_accepts_lecture_json_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            lecture_dir = Path(tmp) / "lecture"
            lecture_dir.mkdir()
            lecture_json = lecture_dir / "lecture.json"
            lecture_json.write_text(
                json.dumps(
                    {
                        "lecture_id": "lecture",
                        "media": {"duration_seconds": 0.0},
                        "transcript": {"text": "", "segments": []},
                        "slides": [],
                    }
                ),
                encoding="utf-8",
            )

            zip_path = export_gemini_test_package(lecture_json)

            self.assertEqual(zip_path, (lecture_dir / "gemini_test_package.zip").resolve())


if __name__ == "__main__":
    unittest.main()
