import tempfile
import unittest
import json
from pathlib import Path

from lecture_processor.artifacts import build_slide_records
from lecture_processor.models import TranscriptResult, TranscriptSegment


class ArtifactTests(unittest.TestCase):
    def test_slide_records_link_segments_by_display_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            slides_dir = Path(tmp) / "slides"
            slides_dir.mkdir()
            for name in [
                "slide_0001_00-00-00.png",
                "slide_0002_00-00-10.png",
                "slide_0003_00-00-20.png",
            ]:
                (slides_dir / name).write_bytes(b"not-a-real-png")
            transcript = TranscriptResult(
                text="one two three",
                segments=[
                    TranscriptSegment(start=0.0, end=9.0, text="one"),
                    TranscriptSegment(start=10.0, end=19.0, text="two"),
                    TranscriptSegment(start=20.0, end=29.0, text="three"),
                ],
            )

            records = build_slide_records(slides_dir, transcript)

        self.assertEqual([[0], [1], [2]], [record["linked_segment_ids"] for record in records])

    def test_short_slide_window_keeps_overlapping_segment(self):
        with tempfile.TemporaryDirectory() as tmp:
            slides_dir = Path(tmp) / "slides"
            slides_dir.mkdir()
            for name in ["slide_0001_00-00-00.png", "slide_0002_00-00-02.png"]:
                (slides_dir / name).write_bytes(b"not-a-real-png")
            transcript = TranscriptResult(
                text="long segment",
                segments=[TranscriptSegment(start=0.0, end=10.0, text="long segment")],
            )

            records = build_slide_records(slides_dir, transcript)

        self.assertEqual([0], records[0]["linked_segment_ids"])

    def test_slide_records_include_smart_extraction_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            slides_dir = Path(tmp) / "slides"
            slides_dir.mkdir()
            image = slides_dir / "slide_0001_00-00-12.png"
            image.write_bytes(b"not-a-real-png")
            image.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "title": "Questions for Business Leaders",
                        "build_stage": "full",
                        "layout": "full-screen",
                        "description": "A full slide with four columns.",
                    }
                ),
                encoding="utf-8",
            )
            transcript = TranscriptResult(text="", segments=[])

            records = build_slide_records(slides_dir, transcript)

        self.assertEqual("Questions for Business Leaders", records[0]["title"])
        self.assertEqual("full", records[0]["build_stage"])
        self.assertEqual("full-screen", records[0]["layout"])
        self.assertEqual("A full slide with four columns.", records[0]["description"])
        self.assertIsNone(records[0]["slide_analysis"])


if __name__ == "__main__":
    unittest.main()
