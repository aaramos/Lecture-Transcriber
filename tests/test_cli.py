import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from lecture_processor.cli import EVENT_PREFIX, _build_event_printer, main


class CliTests(unittest.TestCase):
    def test_empty_folder_reports_no_mov_before_optional_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main(["process", str(Path(tmp)), "--transcription-engine", "auto"])

            self.assertEqual(exit_code, 2)
            self.assertIn("No .mov files found", stderr.getvalue())

    def test_startup_error_writes_run_level_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "out"
            input_dir = root / "input"
            input_dir.mkdir()
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                exit_code = main(
                    [
                        "process",
                        str(input_dir),
                        "--output",
                        str(output),
                        "--transcription-engine",
                        "none",
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertTrue((output / "batch_error.txt").exists())
            summary = (output / "batch_summary.txt").read_text(encoding="utf-8")
            self.assertIn("Run failed before file processing completed", summary)

    def test_json_event_printer_emits_parseable_prefixed_line(self):
        stdout = io.StringIO()
        printer = _build_event_printer(True)

        with redirect_stdout(stdout):
            printer({"kind": "batch_started", "attempted": 2})

        line = stdout.getvalue().strip()
        self.assertTrue(line.startswith(EVENT_PREFIX))
        payload = json.loads(line.removeprefix(EVENT_PREFIX))
        self.assertEqual(payload["kind"], "batch_started")
        self.assertEqual(payload["attempted"], 2)


if __name__ == "__main__":
    unittest.main()
