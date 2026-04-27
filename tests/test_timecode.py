import unittest

from lecture_processor.models import TranscriptSegment
from lecture_processor.timecode import (
    format_srt_timestamp,
    format_timestamp_for_filename,
    segments_to_srt,
)


class TimecodeTests(unittest.TestCase):
    def test_filename_timestamp_uses_portable_separators(self):
        self.assertEqual(format_timestamp_for_filename(330), "00-05-30")

    def test_srt_timestamp(self):
        self.assertEqual(format_srt_timestamp(65.125), "00:01:05,125")

    def test_segments_to_srt(self):
        srt = segments_to_srt([TranscriptSegment(start=1, end=2.5, text="Hello")])
        self.assertIn("1\n00:00:01,000 --> 00:00:02,500\nHello", srt)


if __name__ == "__main__":
    unittest.main()
