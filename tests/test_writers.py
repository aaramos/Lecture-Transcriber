import tempfile
import unittest
from pathlib import Path

from lecture_processor.models import TranscriptResult, TranscriptSegment
from lecture_processor.writers import write_processing_log, write_transcript


class WriterTests(unittest.TestCase):
    def test_transcript_writer_creates_missing_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "missing" / "lecture"
            transcript = TranscriptResult(
                text="hello lecture",
                segments=[TranscriptSegment(start=1.0, end=2.0, text="hello lecture")],
            )

            write_transcript(output_dir, transcript)

            self.assertEqual((output_dir / "transcript.txt").read_text(encoding="utf-8"), "hello lecture\n")
            self.assertTrue((output_dir / "transcript.srt").exists())
            self.assertFalse((output_dir / ".transcript.txt.tmp").exists())
            self.assertFalse((output_dir / ".transcript.srt.tmp").exists())

    def test_processing_log_writer_replaces_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "lecture"

            write_processing_log(output_dir, ["old"])
            write_processing_log(output_dir, ["new"])

            self.assertEqual((output_dir / "processing_log.txt").read_text(encoding="utf-8"), "new\n")
            self.assertFalse((output_dir / ".processing_log.txt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
