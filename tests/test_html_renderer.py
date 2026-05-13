import json
import tempfile
import unittest
from pathlib import Path

from lecture_processor.html_renderer import render_batch_index, render_lecture_page
from lecture_processor.models import BatchSummary, FileResult, FileStatus


class HtmlRendererTests(unittest.TestCase):
    def test_enriched_lecture_uses_flow_template_and_formatted_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            lecture_dir = Path(tmp) / "lecture"
            slides_dir = lecture_dir / "slides"
            slides_dir.mkdir(parents=True)
            (lecture_dir / "lecture.mp4").write_text("video", encoding="utf-8")
            (slides_dir / "slide_0001_00-00-01.png").write_text("png", encoding="utf-8")
            lecture_json = lecture_dir / "lecture.json"
            lecture_json.write_text(
                json.dumps(
                    {
                        "lecture_id": "lecture",
                        "source": {"filename": "lecture.mov"},
                        "media": {"duration_seconds": 120, "normalized_path": "lecture.mp4"},
                        "transcript": {"text": "raw transcript", "segments": []},
                        "slides": [
                            {
                                "id": 1,
                                "filename": "slide_0001_00-00-01.png",
                                "relative_path": "slides/slide_0001_00-00-01.png",
                                "title": "Opening Ideas",
                                "linked_segment_ids": [0],
                            }
                        ],
                        "processing": {},
                        "enrichment": {
                            "title": "Lecture Title",
                            "executive_summary": "A useful summary.",
                            "formatted_transcript": "Lightly edited transcript.\n\nSecond paragraph.",
                            "outline": [{"id": 1, "heading": "Slide 1 discussion", "slide_ids": [1]}],
                            "slide_analysis": [
                                {
                                    "slide_id": 1,
                                    "descriptive_filename": "opening.png",
                                    "caption": "The frame features a blue background and a person on screen.",
                                    "summary": "The lecture opens.",
                                    "tags": [
                                        "opening",
                                        "smart-slide-extraction",
                                        "slide-1",
                                        "transitioning",
                                        "split-left",
                                    ],
                                    "instructor_commentary": "[0] The instructor introduces the topic.",
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

            self.assertNotIn("AI Study Notes", html)
            self.assertNotIn("A useful summary.", html)
            self.assertIn("<video controls", html)
            self.assertIn("lecture.mp4", html)
            self.assertIn('<p class="eyebrow">lecture</p>', html)
            self.assertIn("class=\"flow-section\" open", html)
            self.assertIn("Opening Ideas", html)
            self.assertNotIn("Slide 1 discussion", html)
            self.assertNotIn("Slide idea", html)
            self.assertNotIn("Visible on slide", html)
            self.assertIn("<figcaption class=\"caption\">", html)
            self.assertIn("data-full-image=\"../slides/slide_0001_00-00-01.png\"", html)
            self.assertNotIn("slide-title-row", html)
            self.assertNotIn("slide-id", html)
            self.assertNotIn("class=\"slide-summary\"", html)
            self.assertIn("The instructor introduces the topic.", html)
            self.assertNotIn("[0]", html)
            self.assertIn("Lightly edited transcript.", html)
            self.assertNotIn("Lecture Transcript", html)
            self.assertIn("Further Learning", html)
            self.assertNotIn(">Resources</p>", html)
            self.assertNotIn(">Outline</p>", html)
            self.assertNotIn("opening.png", html)
            self.assertIn("opening", html)
            self.assertNotIn("smart-slide-extraction", html)
            self.assertNotIn("<span class=\"tag\">slide-1</span>", html)
            self.assertNotIn("transitioning", html)
            self.assertNotIn("split-left", html)
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

    def test_raw_transcript_is_split_into_readable_paragraphs(self):
        with tempfile.TemporaryDirectory() as tmp:
            lecture_dir = Path(tmp) / "lecture"
            lecture_dir.mkdir(parents=True)
            long_transcript = (
                "First sentence introduces the lecture. Second sentence adds context. "
                "Third sentence gives an example. Fourth sentence closes the opening. "
                "Fifth sentence starts a new idea. Sixth sentence develops that idea. "
                "Seventh sentence adds detail. Eighth sentence concludes the thought."
            )
            lecture_json = lecture_dir / "lecture.json"
            lecture_json.write_text(
                json.dumps(
                    {
                        "lecture_id": "lecture",
                        "source": {"filename": "lecture.mov"},
                        "media": {"duration_seconds": 120},
                        "transcript": {"text": long_transcript, "segments": []},
                        "slides": [],
                        "processing": {},
                        "enrichment": None,
                    }
                ),
                encoding="utf-8",
            )

            html_path = render_lecture_page(lecture_json)
            html = html_path.read_text(encoding="utf-8")

            self.assertGreaterEqual(html.count("<p>"), 2)
            self.assertIn("First sentence introduces the lecture.", html)
            self.assertIn("Eighth sentence concludes the thought.", html)

    def test_batch_index_uses_clean_title_and_hides_internal_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "Video_7-1_AI_Strategy_and_the_C-Suite_Agenda_processed"
            lecture_dir = output_dir / "Video_7-1_AI_Strategy_and_the_C-Suite_Agenda"
            html_dir = lecture_dir / "html"
            html_dir.mkdir(parents=True)
            (html_dir / "index.html").write_text("<html></html>", encoding="utf-8")
            lecture_json = lecture_dir / "lecture.json"
            lecture_json.write_text(
                json.dumps(
                    {
                        "lecture_id": "Video_7-1_AI_Strategy_and_the_C-Suite_Agenda",
                        "source": {"filename": "Video_7-1_AI_Strategy_and_the_C-Suite_Agenda.mov"},
                        "media": {"duration_seconds": 120},
                        "transcript": {"text": "raw transcript", "segments": []},
                        "slides": [],
                        "processing": {},
                        "enrichment": {
                            "title": "So What Is Ai Strategy I Define It",
                            "executive_summary": (
                                "Local overview stub. This placeholder keeps the experimental routing path working "
                                "until a real local overview model is configured."
                            ),
                        },
                    }
                ),
                encoding="utf-8",
            )
            summary = BatchSummary(
                attempted=1,
                completed=1,
                failed=0,
                skipped=0,
                stopped=0,
                results=[
                    FileResult(
                        source=Path("Video_7-1_AI_Strategy_and_the_C-Suite_Agenda.mov"),
                        output_dir=lecture_dir,
                        status=FileStatus.COMPLETED,
                        html_path=html_dir / "index.html",
                    )
                ],
            )

            index_path = render_batch_index(output_dir, summary)
            html = index_path.read_text(encoding="utf-8")

            self.assertIn("<h1>Video 7-1 AI Strategy and the C-Suite Agenda</h1>", html)
            self.assertNotIn("Batch Study Index", html)
            self.assertNotIn("1 completed, 0 failed, 0 skipped, 0 stopped.", html)
            self.assertNotIn("Local overview stub", html)
            self.assertIn("Overview unavailable", html)
            self.assertIn("So What Is Ai Strategy I Define It", html)


if __name__ == "__main__":
    unittest.main()
