import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from lecture_processor.cli import EVENT_PREFIX, _build_event_printer, _prepend_tool_parent_to_path, main


class CliTests(unittest.TestCase):
    def test_empty_folder_reports_no_supported_files_before_optional_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main(["process", str(Path(tmp)), "--transcription-engine", "auto"])

            self.assertEqual(exit_code, 2)
            self.assertIn("No supported lecture files found", stderr.getvalue())

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

    def test_prepend_tool_parent_to_path_supports_internal_ffmpeg_users(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool = Path(tmp) / "ffmpeg"
            tool.write_text("binary", encoding="utf-8")
            previous_path = os.environ.get("PATH")
            try:
                os.environ["PATH"] = "/usr/bin"
                _prepend_tool_parent_to_path(str(tool))
                self.assertEqual(os.environ["PATH"].split(os.pathsep)[0], str(tool.parent))
            finally:
                if previous_path is None:
                    os.environ.pop("PATH", None)
                else:
                    os.environ["PATH"] = previous_path

    def test_export_gemini_test_command_writes_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            lecture = Path(tmp) / "lecture"
            lecture.mkdir()
            (lecture / "lecture.json").write_text(
                json.dumps(
                    {
                        "lecture_id": "lecture",
                        "media": {"duration_seconds": 0.0},
                        "transcript": {"text": "", "segments": []},
                        "slides": [],
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(["export-gemini-test", str(lecture)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((lecture / "gemini_test_package.zip").exists())
            self.assertIn("Gemini test package:", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
