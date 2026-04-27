import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from lecture_processor.cli import main


class CliTests(unittest.TestCase):
    def test_empty_folder_reports_no_mov_before_optional_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main(["process", str(Path(tmp)), "--transcription-engine", "auto"])

            self.assertEqual(exit_code, 2)
            self.assertIn("No .mov files found", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
