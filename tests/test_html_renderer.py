import json
import tempfile
import unittest
from pathlib import Path

from lecture_processor.html_renderer import render_lecture_page


class HtmlRendererTests(unittest.TestCase):
    def test_enriched_lecture_uses_flow_template_and_formatted_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            lecture_dir = Path(tmp) / "lecture"
            slides_dir = lecture_dir / "slides"
            slides_dir.mkdir(parents=True)
            (slides_dir / "slide_0001_00-00-01.png").write_text("png", encoding="utf-8")
            lecture_json = lecture_dir / "lecture.json"
            lecture_json.write_text(
                json.dumps(
                    {
                        "lecture_id": "lecture",
                        "source": {"filename": "lecture.mov"},
                        "media": {"duration_seconds": 120},
                        "transcript": {"text": "raw transcript", "segments": []},
                        "slides": [
                            {
                                "id": 1,
                                "filename": "slide_0001_00-00-01.png",
                                "relative_path": "slides/slide_0001_00-00-01.png",
                            }
                        ],
                        "processing": {},
                        "enrichment": {
                            "title": "Lecture Title",
                            "executive_summary": "A useful summary.",
                            "formatted_transcript": "Lightly edited transcript.\n\nSecond paragraph.",
                            "outline": [{"id": 1, "heading": "Opening", "slide_ids": [1]}],
                            "slide_analysis": [
                                {
                                    "slide_id": 1,
                                    "descriptive_filename": "opening.png",
                                    "caption": "A title slide is visible.",
                                    "summary": "The lecture opens.",
                                    "tags": ["opening"],
                                    "instructor_commentary": "The instructor introduces the topic.",
                                }
                            ],
                            "resources": [
                                {
                                    "title": "Resource",
                                    "url": "https://example.com",
                                    "summary": "More reading.",
                                    "source_quality": "high",
                                }
                            ],
                            "warnings": [],
                        },
                    }
                ),
                encoding="utf-8",
            )

            html_path = render_lecture_page(lecture_json)
            html = html_path.read_text(encoding="utf-8")

            self.assertIn("AI Study Notes", html)
            self.assertIn("class=\"flow-section\" open", html)
            self.assertIn("Visible on slide", html)
            self.assertIn("Lightly edited transcript.", html)
            self.assertIn("Lecture Transcript", html)
            self.assertIn("Further Learning", html)
            self.assertNotIn("Slides And Commentary", html)

    def test_outline_accepts_gemini_slide_id_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            lecture_dir = Path(tmp) / "lecture"
            slides_dir = lecture_dir / "slides"
            slides_dir.mkdir(parents=True)
            (slides_dir / "slide_0001_00-00-00.png").write_text("png", encoding="utf-8")
            (slides_dir / "slide_0002_00-00-14.png").write_text("png", encoding="utf-8")
            lecture_json = lecture_dir / "lecture.json"
            lecture_json.write_text(
                json.dumps(
                    {
                        "lecture_id": "lecture",
                        "source": {"filename": "lecture.mov"},
                        "media": {"duration_seconds": 120},
                        "transcript": {"text": "raw transcript", "segments": []},
                        "slides": [
                            {
                                "id": 1,
                                "filename": "slide_0001_00-00-00.png",
                                "relative_path": "slides/slide_0001_00-00-00.png",
                            },
                            {
                                "id": 2,
                                "filename": "slide_0002_00-00-14.png",
                                "relative_path": "slides/slide_0002_00-00-14.png",
                            },
                        ],
                        "processing": {},
                        "enrichment": {
                            "title": "Lecture Title",
                            "executive_summary": "A useful summary.",
                            "formatted_transcript": "Readable transcript.",
                            "outline": [
                                {
                                    "slide_id": "Slide 1",
                                    "title": "Introduction to AI in Creative Industries",
                                },
                                {
                                    "slide_id": "Slide 2",
                                    "title": "Speaker Introduction",
                                },
                            ],
                            "slide_analysis": [
                                {
                                    "slide_id": "Slide 1",
                                    "descriptive_filename": "opening.png",
                                    "summary": "The lecture opens.",
                                    "tags": [],
                                },
                                {
                                    "slide_id": "Slide 2",
                                    "descriptive_filename": "speaker.png",
                                    "summary": "The speaker appears.",
                                    "tags": [],
                                },
                            ],
                            "resources": [],
                            "warnings": [],
                        },
                    }
                ),
                encoding="utf-8",
            )

            html_path = render_lecture_page(lecture_json)
            html = html_path.read_text(encoding="utf-8")

            self.assertIn("Introduction to AI in Creative Industries", html)
            self.assertIn("Speaker Introduction", html)
            self.assertIn("The lecture opens.", html)
            self.assertIn("The speaker appears.", html)
            self.assertNotIn("Additional slides", html)
            self.assertNotIn("0 slides", html)


if __name__ == "__main__":
    unittest.main()
