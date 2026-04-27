import tempfile
import unittest
from pathlib import Path

from lecture_processor.errors import DependencyMissingError
from lecture_processor.media import _require_command, ensure_media_tools


class MediaDependencyTests(unittest.TestCase):
    def test_preflight_reports_missing_ffprobe(self):
        with self.assertRaises(DependencyMissingError) as context:
            ensure_media_tools(
                ffprobe_path="definitely-not-ffprobe",
                ffmpeg_path="definitely-not-ffmpeg",
                needs_ffmpeg=False,
            )

        self.assertIn("ffprobe", str(context.exception))

    def test_finds_workspace_local_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tool = root / ".tools" / "darwin_arm64" / "ffmpeg"
            tool.parent.mkdir(parents=True)
            tool.write_text("binary", encoding="utf-8")
            previous = Path.cwd()
            try:
                import os

                os.chdir(root)
                self.assertEqual(Path(_require_command("ffmpeg")).resolve(), tool.resolve())
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
