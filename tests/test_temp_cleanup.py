import json
import os
import tempfile
import unittest
from pathlib import Path

from lecture_processor.temp_cleanup import cleanup_output_temp_files, cleanup_slide_temp_dirs


class TempCleanupTests(unittest.TestCase):
    def test_output_cleanup_removes_known_temp_files_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            nested = output / "lecture"
            nested.mkdir(parents=True)
            (nested / ".normalized_work.mp4").write_text("temp", encoding="utf-8")
            (nested / ".transcript.txt.tmp").write_text("temp", encoding="utf-8")
            (nested / ".lecture.mp4.ffmpeg.tmp").write_text("temp", encoding="utf-8")
            (nested / "keep.tmp").write_text("not ours", encoding="utf-8")

            removed = cleanup_output_temp_files(output)

            self.assertEqual(removed, 3)
            self.assertFalse((nested / ".normalized_work.mp4").exists())
            self.assertFalse((nested / ".transcript.txt.tmp").exists())
            self.assertFalse((nested / ".lecture.mp4.ffmpeg.tmp").exists())
            self.assertTrue((nested / "keep.tmp").exists())

    def test_output_cleanup_keeps_active_lock_and_removes_stale_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            output.mkdir()
            lock_path = output / ".lecture_processor.lock"
            lock_path.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")

            self.assertEqual(cleanup_output_temp_files(output), 0)
            self.assertTrue(lock_path.exists())

            lock_path.write_text(json.dumps({"pid": -1}), encoding="utf-8")

            self.assertEqual(cleanup_output_temp_files(output), 1)
            self.assertFalse(lock_path.exists())

    def test_slide_cleanup_removes_stale_slide_temp_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            stale = temp_root / "lecture-slides-999999-slide"
            stale.mkdir()
            keep = temp_root / f"lecture-slides-{os.getpid()}-active"
            keep.mkdir()

            removed = cleanup_slide_temp_dirs(temp_root)

            self.assertEqual(removed, 1)
            self.assertFalse(stale.exists())
            self.assertTrue(keep.exists())


if __name__ == "__main__":
    unittest.main()
