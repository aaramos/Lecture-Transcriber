import io
import json
import os
import re
import unittest
from pathlib import Path

from lecture_processor.notebook_export import export_notebook_markdown

GOLDEN_PATH = (
    Path(__file__).resolve().parent.parent
    / "handoff" / "open-notebook-export" / "examples" / "expected-output.md"
)


def _golden_slides() -> list:
    return [
        {
            "id": 4,
            "timestamp_seconds": 750.0,
            "title": "Arrhenius Equation",
            "linked_segment_ids": [0],
            "description": None,
        },
        {
            "id": 5,
            "timestamp_seconds": 902.0,
            "title": "Worked Example: Finding Ea",
            "linked_segment_ids": [1],
            "description": None,
        },
        {
            "id": 6,
            "timestamp_seconds": 1187.0,
            "title": "Catalysis and the Reaction Coordinate",
            "linked_segment_ids": [2],
            "description": None,
        },
    ]


def _golden_fixture() -> dict:
    return {
        "schema_version": "1.0.0",
        "lecture_id": "Video_6-1_AI_and_Creativity_Foundations",
        "source": {"filename": "Module6.mp4"},
        "media": {"duration_seconds": 3120.0},
        "transcript": {
            "text": "",
            "segments": [
                {
                    "id": 0,
                    "start": 745.0,
                    "end": 770.0,
                    "text": "So the slope of this line gives you negative Ea over R, which "
                    "means if you have rate constants at two different temperatures you can "
                    "back out the activation energy without ever measuring it directly. "
                    "That's the whole reason we bother linearizing it. In practice you'll "
                    "almost never get a single clean rate constant handed to you on an "
                    "exam — you'll get two temperatures and two rates, and this is the tool "
                    "that connects them.",
                },
                {
                    "id": 1,
                    "start": 840.0,
                    "end": 905.0,
                    "text": "Notice I did not need the pre-exponential factor anywhere in "
                    "this calculation. It cancels when you take the ratio. Students lose a "
                    "lot of points on this problem by trying to solve for A first — you "
                    "don't need it, and on a timed exam that detour will cost you five "
                    "minutes you don't have.",
                },
                {
                    "id": 2,
                    "start": 1050.0,
                    "end": 1120.0,
                    "text": "The thing to hold onto here is that the catalyst changes the "
                    "barrier, not the destination. Reactants and products sit at exactly "
                    "the same energy in both curves. So a catalyst speeds up how fast you "
                    "get to equilibrium — it does not move where equilibrium is. That "
                    "distinction shows up on every exam I write.",
                },
            ],
        },
        "slides": _golden_slides(),
        "processing": {"status": "completed"},
        "enrichment": {
            "title": "Module 6: Reaction Kinetics",
            "executive_summary": (
                "Covers the temperature dependence of reaction rates, building from the "
                "empirical rate law to the Arrhenius relationship and its linearized form. "
                "The final third works two exam-style problems: extracting activation "
                "energy from a two-point rate measurement, and predicting a rate constant "
                "at an unmeasured temperature."
            ),
            "outline": [
                {"heading": "Rate laws and reaction order"},
                {"heading": "The Arrhenius equation"},
                {"heading": "Linearized Arrhenius plots"},
                {"heading": "Activation energy from experimental data"},
                {"heading": "Catalysis and the reaction coordinate"},
            ],
            "slide_analysis": [
                {
                    "slide_id": 4,
                    "caption": (
                        "Plot of ln(k) against 1/T showing a straight line with negative "
                        "slope. Axes labeled ln(k) and 1/T (K\u207b\u00b9). The equation "
                        "k = Ae^(-Ea/RT) is boxed in the top right, with its linearized "
                        "form ln(k) = ln(A) - Ea/R \u00b7 (1/T) below it."
                    ),
                    "summary": "",
                },
                {
                    "slide_id": 5,
                    "caption": (
                        "Two-column worked problem. Given: k\u2081 = 2.5 \u00d7 10\u207b\u00b3 "
                        "s\u207b\u00b9 at 300 K, k\u2082 = 1.4 \u00d7 10\u207b\u00b2 "
                        "s\u207b\u00b9 at 320 K. Solution steps show substitution into the "
                        "two-point form ln(k\u2082/k\u2081) = -Ea/R \u00b7 (1/T\u2082 - "
                        "1/T\u2081), arriving at Ea \u2248 68.5 kJ/mol."
                    ),
                    "summary": "",
                },
                {
                    "slide_id": 6,
                    "caption": (
                        "Reaction coordinate diagram with two overlaid energy curves. The "
                        "uncatalyzed path shows a higher transition-state barrier; the "
                        "catalyzed path shows a lower barrier reaching the same product "
                        "energy. Both curves share identical reactant and product energy "
                        "levels."
                    ),
                    "summary": "",
                },
            ],
            "resources": [
                {
                    "title": "Arrhenius equation — LibreTexts",
                    "url": "https://chem.libretexts.org/example",
                    "summary": (
                        "Derivation and worked problems at roughly this course's level."
                    ),
                },
                {
                    "title": "Reaction coordinate diagrams explained",
                    "url": "https://example.edu/rxn-coord",
                    "summary": (
                        "Visual treatment of catalysis, useful if the overlay diagram "
                        "above was hard to read."
                    ),
                },
            ],
        },
    }


def _write_fixture(tmp_dir: Path, artifact: dict) -> Path:
    lecture_json = tmp_dir / "lecture.json"
    lecture_json.write_text(json.dumps(artifact), encoding="utf-8")
    return lecture_json


class NotebookExportTests(unittest.TestCase):
    def test_golden_output_matches_expected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            lecture_json = _write_fixture(Path(tmp), _golden_fixture())
            result = export_notebook_markdown(lecture_json, course="CHEM 301")
            self.assertIsNotNone(result)
            produced = result.read_text(encoding="utf-8")
            golden = GOLDEN_PATH.read_text(encoding="utf-8")
            self.assertEqual(produced, golden)

    def test_failed_lecture_writes_no_file_and_returns_none(self):
        import tempfile

        artifact = _golden_fixture()
        artifact["processing"] = {"status": "failed"}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lecture_json = _write_fixture(tmp_path, artifact)
            result = export_notebook_markdown(lecture_json, course="CHEM 301")
            self.assertIsNone(result)
            self.assertFalse((tmp_path / "notebook.md").exists())

    def test_enrichment_null_omits_summary_topics_reading(self):
        import tempfile

        artifact = _golden_fixture()
        artifact["enrichment"] = None
        for slide in artifact["slides"]:
            slide["title"] = slide["title"]
            slide["description"] = "Slide visual description fallback."
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lecture_json = _write_fixture(tmp_path, artifact)
            result = export_notebook_markdown(lecture_json, course="CHEM 301")
            self.assertIsNotNone(result)
            produced = result.read_text(encoding="utf-8")
            self.assertNotIn("## Summary", produced)
            self.assertNotIn("## Key Topics", produced)
            self.assertNotIn("## Further Reading", produced)
            self.assertIn("### Slide 4", produced)
            self.assertIn("**On the slide:** Slide visual description fallback.", produced)
            self.assertIn("**Instructor said:**", produced)

    def test_audio_only_segments_slides_empty(self):
        import tempfile

        artifact = _golden_fixture()
        artifact["slides"] = []
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lecture_json = _write_fixture(tmp_path, artifact)
            result = export_notebook_markdown(lecture_json, course="CHEM 301")
            self.assertIsNotNone(result)
            produced = result.read_text(encoding="utf-8")
            self.assertNotIn("### Slide", produced)
            self.assertIn("### Transcript (part 1)", produced)
            self.assertIn("## Summary", produced)

    def test_slide_without_linked_segments_omits_instructor_line(self):
        import tempfile

        artifact = _golden_fixture()
        for slide in artifact["slides"]:
            slide["linked_segment_ids"] = []
        artifact["transcript"]["segments"] = []
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lecture_json = _write_fixture(tmp_path, artifact)
            result = export_notebook_markdown(lecture_json, course="CHEM 301")
            self.assertIsNotNone(result)
            produced = result.read_text(encoding="utf-8")
            self.assertNotIn("**Instructor said:**", produced)
            self.assertIn("**On the slide:**", produced)

    def test_markdown_hostile_transcript_neutralizes_leading_hash(self):
        import tempfile

        artifact = _golden_fixture()
        hostile = (
            "## Stray Heading that starts a line. "
            "Normal sentence one. "
            "# Another heading that starts a line. "
            "Normal sentence two, with *emphasis* and _underscore_ and `code`."
        )
        artifact["slides"] = []
        artifact["transcript"]["segments"] = [
            {"id": 0, "start": 0.0, "end": 30.0, "text": hostile}
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lecture_json = _write_fixture(tmp_path, artifact)
            result = export_notebook_markdown(lecture_json, course="CHEM 301")
            self.assertIsNotNone(result)
            produced = result.read_text(encoding="utf-8")
            self.assertNotRegex(produced, r"(?m)^## Stray")
            self.assertIn("\\## Stray", produced)
            self.assertNotRegex(produced, r"(?m)^# Another")
            three_count = produced.count("\n### ")
            self.assertGreaterEqual(three_count, 1)

    def test_duplicate_slide_timestamps_emit_point_timecode(self):
        import tempfile

        artifact = _golden_fixture()
        artifact["slides"][1]["timestamp_seconds"] = artifact["slides"][0]["timestamp_seconds"]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lecture_json = _write_fixture(tmp_path, artifact)
            result = export_notebook_markdown(lecture_json, course="CHEM 301")
            self.assertIsNotNone(result)
            produced = result.read_text(encoding="utf-8")
            self.assertNotIn("(12:30\u201312:30)", produced)
            self.assertIn("(12:30)", produced)

    def test_chunk_boundary_no_section_exceeds_600_tokens(self):
        import tempfile

        artifact = _golden_fixture()
        long_text = " ".join(
            ["This is sentence number %d about reaction kinetics." % i for i in range(200)]
        )
        artifact["transcript"]["segments"] = [
            {"id": 0, "start": 745.0, "end": 900.0, "text": long_text}
        ]
        for slide in artifact["slides"]:
            slide["linked_segment_ids"] = [0]
        artifact["slides"] = [artifact["slides"][0]]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lecture_json = _write_fixture(tmp_path, artifact)
            result = export_notebook_markdown(lecture_json, course="CHEM 301")
            self.assertIsNotNone(result)
            produced = result.read_text(encoding="utf-8")
            sections = re.split(r"\n### ", produced)
            for section in sections[1:]:
                tokens = len(section.split())
                self.assertLess(tokens, 600, f"section too long: {tokens} tokens")

    def test_slide_heading_count_matches_slide_count_when_short(self):
        import tempfile

        artifact = _golden_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lecture_json = _write_fixture(tmp_path, artifact)
            result = export_notebook_markdown(lecture_json, course="CHEM 301")
            produced = result.read_text(encoding="utf-8")
            headings = re.findall(r"^### Slide \d+ \u2014", produced, re.MULTILINE)
            self.assertEqual(len(headings), len(artifact["slides"]))

    def test_course_falls_back_to_folder_name_when_empty(self):
        import tempfile

        artifact = _golden_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "My Course Folder"
            tmp_path.mkdir()
            lecture_json = _write_fixture(tmp_path, artifact)
            result = export_notebook_markdown(lecture_json, course="")
            self.assertIsNotNone(result)
            produced = result.read_text(encoding="utf-8")
            self.assertIn("# My Course Folder \u2014 Module 6: Reaction Kinetics", produced)

    def test_linked_segment_ids_supersedes_nearest_timestamp(self):
        import tempfile

        artifact = _golden_fixture()
        artifact["media"] = {"duration_seconds": 1300.0}
        artifact["transcript"]["segments"] = [
            {"id": 0, "start": 745.0, "end": 760.0, "text": "NEAR_SEGMENT_TEXT_MARKER_AAA"},
            {"id": 1, "start": 1100.0, "end": 1150.0, "text": "FAR_SEGMENT_TEXT_MARKER_BBB"},
            {"id": 2, "start": 1180.0, "end": 1200.0, "text": "LAST_SEGMENT_TEXT_MARKER_CCC"},
        ]
        artifact["slides"] = [
            {
                "id": 4,
                "timestamp_seconds": 750.0,
                "title": "First Slide",
                "linked_segment_ids": [1],
                "description": None,
            },
            {
                "id": 5,
                "timestamp_seconds": 1187.0,
                "title": "Last Slide",
                "linked_segment_ids": [2],
                "description": None,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            lecture_json = _write_fixture(Path(tmp), artifact)
            result = export_notebook_markdown(lecture_json, course="CHEM 301")
            self.assertIsNotNone(result)
            produced = result.read_text(encoding="utf-8")
            self.assertIn("FAR_SEGMENT_TEXT_MARKER_BBB", produced)
            self.assertIn("LAST_SEGMENT_TEXT_MARKER_CCC", produced)
            self.assertNotIn("NEAR_SEGMENT_TEXT_MARKER_AAA", produced)

    def test_boundary_straddling_segment_appears_in_both_slides(self):
        import tempfile

        artifact = _golden_fixture()
        artifact["media"] = {"duration_seconds": 1000.0}
        artifact["transcript"]["segments"] = [
            {"id": 0, "start": 745.0, "end": 770.0, "text": "UNIQUE_MARKER_LEFT"},
            {"id": 1, "start": 895.0, "end": 910.0, "text": "SHARED_BOUNDARY_MARKER"},
            {"id": 2, "start": 1050.0, "end": 1120.0, "text": "UNIQUE_MARKER_RIGHT"},
        ]
        artifact["slides"] = [
            {
                "id": 4,
                "timestamp_seconds": 750.0,
                "title": "Left Slide",
                "linked_segment_ids": [0, 1],
                "description": None,
            },
            {
                "id": 5,
                "timestamp_seconds": 902.0,
                "title": "Right Slide",
                "linked_segment_ids": [1, 2],
                "description": None,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            lecture_json = _write_fixture(Path(tmp), artifact)
            result = export_notebook_markdown(lecture_json, course="CHEM 301")
            self.assertIsNotNone(result)
            produced = result.read_text(encoding="utf-8")
            slide4_section = produced.split("### Slide 4")[1].split("### Slide 5")[0]
            slide5_section = produced.split("### Slide 5")[1].split("## Further Reading")[0]
            self.assertIn("SHARED_BOUNDARY_MARKER", slide4_section)
            self.assertIn("SHARED_BOUNDARY_MARKER", slide5_section)

    def test_final_slide_range_ends_at_transcript_end_not_duration(self):
        import tempfile

        artifact = _golden_fixture()
        artifact["media"] = {"duration_seconds": 80.0}
        artifact["transcript"]["segments"] = [
            {"id": 0, "start": 745.0, "end": 770.0, "text": "First segment."},
            {"id": 1, "start": 840.0, "end": 905.0, "text": "Second segment."},
            {"id": 2, "start": 1050.0, "end": 1500.0, "text": "Third segment extends past last slide."},
        ]
        artifact["slides"] = [
            {
                "id": 4,
                "timestamp_seconds": 750.0,
                "title": "First Slide",
                "linked_segment_ids": [0],
                "description": None,
            },
            {
                "id": 5,
                "timestamp_seconds": 902.0,
                "title": "Last Slide",
                "linked_segment_ids": [1, 2],
                "description": None,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            lecture_json = _write_fixture(Path(tmp), artifact)
            result = export_notebook_markdown(lecture_json, course="CHEM 301")
            self.assertIsNotNone(result)
            produced = result.read_text(encoding="utf-8")
            self.assertIn("15:02\u201325:00", produced)
            self.assertNotIn("(15:02)", produced)
            self.assertIn("25 min", produced)

    def test_continuation_headings_have_distinct_increasing_timestamps(self):
        import tempfile

        artifact = _golden_fixture()
        artifact["media"] = {"duration_seconds": 100.0}
        long_text = " ".join(
            ["Sentence number %d about the topic under discussion here." % i for i in range(120)]
        )
        artifact["transcript"]["segments"] = [
            {"id": 0, "start": 750.0, "end": 1500.0, "text": long_text},
        ]
        artifact["slides"] = [
            {
                "id": 4,
                "timestamp_seconds": 750.0,
                "title": "Solo Slide",
                "linked_segment_ids": [0],
                "description": None,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            lecture_json = _write_fixture(Path(tmp), artifact)
            result = export_notebook_markdown(lecture_json, course="CHEM 301")
            self.assertIsNotNone(result)
            produced = result.read_text(encoding="utf-8")
            cont_headings = re.findall(r"### Slide 4.*?\(([\d:]+(?:\u2013[\d:]+)?)\)", produced)
            self.assertGreater(len(cont_headings), 1)
            ranges = []
            for h in cont_headings:
                if "\u2013" in h:
                    parts = h.split("\u2013")
                    ranges.append((parts[0], parts[1]))
                else:
                    ranges.append((h, h))
            starts = [r[0] for r in ranges]
            self.assertEqual(starts, sorted(set(starts)), f"continuation starts not distinct: {starts}")


if __name__ == "__main__":
    unittest.main()