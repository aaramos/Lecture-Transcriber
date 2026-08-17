import unittest

from lecture_processor.models import TranscriptSegment
from lecture_processor.timecode import (
    format_srt_timestamp,
    format_timecode,
    format_timecode_range,
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

        scaled_negative = segment.scaled(-1.0)
        self.assertEqual(scaled_negative.start, 0.0)
        self.assertEqual(scaled_negative.end, 0.0)

        invalid_segment = TranscriptSegment(start=5.0, end=2.0, text="Invalid timestamps")
        scaled_invalid = invalid_segment.scaled(2.0)
        self.assertEqual(scaled_invalid.start, 10.0)
        self.assertEqual(scaled_invalid.end, 10.0)

    def test_format_timecode_under_one_minute(self):
        self.assertEqual(format_timecode(0), "0:00")
        self.assertEqual(format_timecode(5.4), "0:05")
        self.assertEqual(format_timecode(59.6), "1:00")

    def test_format_timecode_minutes_seconds(self):
        self.assertEqual(format_timecode(90), "1:30")
        self.assertEqual(format_timecode(750), "12:30")
        self.assertEqual(format_timecode(3599.6), "1:00:00")

    def test_format_timecode_with_hours(self):
        self.assertEqual(format_timecode(3600), "1:00:00")
        self.assertEqual(format_timecode(5415), "1:30:15")
        self.assertEqual(format_timecode(36000), "10:00:00")

    def test_format_timecode_negative_clamped_to_zero(self):
        self.assertEqual(format_timecode(-5), "0:00")

    def test_format_timecode_range_normal(self):
        self.assertEqual(format_timecode_range(750, 1187), "(12:30\u201319:47)")
        self.assertEqual(format_timecode_range(0, 60), "(0:00\u20131:00)")

    def test_format_timecode_range_duplicate_timestamp_is_point(self):
        self.assertEqual(format_timecode_range(750, 750), "(12:30)")
        self.assertEqual(format_timecode_range(0, 0), "(0:00)")

    def test_format_timecode_range_end_before_start_treated_as_point(self):
        self.assertEqual(format_timecode_range(750, 700), "(12:30)")


if __name__ == "__main__":
    unittest.main()
