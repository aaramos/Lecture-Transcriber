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

    def test_transcript_segment_scaling_validation(self):
        segment = TranscriptSegment(start=1.5, end=3.0, text="Scaling test")
        scaled_positive = segment.scaled(2.0)
        self.assertEqual(scaled_positive.start, 3.0)
        self.assertEqual(scaled_positive.end, 6.0)

        # Scale by negative value
        scaled_negative = segment.scaled(-1.0)
        self.assertEqual(scaled_negative.start, 0.0)
        self.assertEqual(scaled_negative.end, 0.0)

        # Segment with start > end initially
        invalid_segment = TranscriptSegment(start=5.0, end=2.0, text="Invalid timestamps")
        scaled_invalid = invalid_segment.scaled(2.0)
        self.assertEqual(scaled_invalid.start, 10.0)
        self.assertEqual(scaled_invalid.end, 10.0)  # end is clamped to start


if __name__ == "__main__":
    unittest.main()
