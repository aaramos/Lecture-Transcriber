import unittest

from lecture_processor.errors import DependencyMissingError
from lecture_processor.media import ensure_media_tools


class MediaDependencyTests(unittest.TestCase):
    def test_preflight_reports_missing_ffprobe(self):
        with self.assertRaises(DependencyMissingError) as context:
            ensure_media_tools(
                ffprobe_path="definitely-not-ffprobe",
                ffmpeg_path="definitely-not-ffmpeg",
                needs_ffmpeg=False,
            )

        self.assertIn("ffprobe", str(context.exception))


if __name__ == "__main__":
    unittest.main()
