from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import harness
import harness_web


class VisionHarnessWebTests(unittest.TestCase):
    def isolated_web_paths(self, root: Path):
        return patch.multiple(
            harness_web,
            SESSION_DIR=root / ".harness_sessions",
            TESTS_DIR=root / ".harness_tests",
            TEST_REGISTRY_PATH=root / ".harness_tests" / "registry.json",
        )

    def test_token_status_reports_keychain_without_returning_token(self) -> None:
        with patch.dict("os.environ", {"LM_STUDIO_API_KEY": "", "LM_API_TOKEN": ""}, clear=False):
            with patch.object(harness_web, "read_keychain_token", return_value="secret-token"):
                status = harness_web.token_status()

        self.assertTrue(status["saved"])
        self.assertEqual(status["source"], "keychain")
        self.assertNotIn("secret-token", json.dumps(status))

    def test_ensure_browser_batch_does_not_create_empty_baseline(self) -> None:
        results = harness.fresh_results(Path("browser:batch"))

        harness_web.ensure_browser_batch(results, "browser:batch", [Path("slide_0001.png")])

        self.assertEqual(results["codex_verdicts"], {})

    def test_latest_job_snapshot_returns_newest_job(self) -> None:
        old_jobs = dict(harness_web.JOBS)
        try:
            harness_web.JOBS.clear()
            harness_web.JOBS["job_old"] = {
                "id": "job_old",
                "status": "complete",
                "started_at": "2026-05-04T10:00:00Z",
            }
            harness_web.JOBS["job_new"] = {
                "id": "job_new",
                "status": "running",
                "started_at": "2026-05-04T11:00:00Z",
                "logs": [{"at": "2026-05-04T11:00:01Z", "message": "still running"}],
            }

            latest = harness_web.latest_job_snapshot()
            summaries = harness_web.job_summaries()

            self.assertEqual(latest["id"], "job_new")
            self.assertEqual(summaries[0]["id"], "job_new")
            self.assertEqual(latest["logs"][0]["message"], "still running")
        finally:
            harness_web.JOBS.clear()
            harness_web.JOBS.update(old_jobs)

    def test_web_profile_summaries_stay_vision_scoped(self) -> None:
        names = [profile["name"] for profile in harness_web.profile_summaries()]

        self.assertEqual(names, ["VISION_HQ", "VISION_TURBO"])

    def test_safe_upload_name_uses_image_basename(self) -> None:
        seen = set()

        self.assertEqual(
            harness_web.safe_upload_name("batchImageTest/slide_0001.png", seen),
            "slide_0001.png",
        )

    def test_safe_upload_name_deduplicates_flattened_names(self) -> None:
        seen = set()

        first = harness_web.safe_upload_name("a/slide.png", seen)
        second = harness_web.safe_upload_name("b/slide.png", seen)

        self.assertEqual(first, "slide.png")
        self.assertEqual(second, "slide_2.png")

    def test_run_job_against_mock_lm_studio(self) -> None:
        mock_server = MockLMStudioServer()
        mock_server.start()
        old_results = harness.RESULTS_PATH
        old_report = harness.REPORT_PATH
        old_sessions = dict(harness_web.SESSIONS)
        old_jobs = dict(harness_web.JOBS)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                image_path = root / "slide_0001.png"
                image_path.write_bytes(b"fake image bytes")
                harness.RESULTS_PATH = root / "results.json"
                harness.REPORT_PATH = root / "report.md"
                harness_web.SESSIONS.clear()
                harness_web.JOBS.clear()
                harness_web.SESSIONS["session_test"] = {
                    "id": "session_test",
                    "folder_label": "mock",
                    "batch_path": "browser:mock",
                    "path": str(root),
                    "images": [image_path.name],
                    "created_at": harness.utc_now(),
                }
                harness_web.JOBS["job_test"] = {
                    "id": "job_test",
                    "status": "queued",
                    "session_id": "session_test",
                    "models": ["mock-vision"],
                    "prompt": "classify_v1",
                    "base_url": mock_server.base_url,
                    "timeout": 10,
                    "current_model": "",
                    "current_image": "",
                    "completed_images": 0,
                    "total_images": 1,
                    "logs": [],
                    "error": "",
                    "runs": [],
                    "started_at": harness.utc_now(),
                    "finished_at": "",
                }

                harness_web.run_job("job_test")

                self.assertEqual(harness_web.JOBS["job_test"]["status"], "complete")
                self.assertEqual(harness_web.JOBS["job_test"]["completed_images"], 1)
                self.assertTrue(harness.RESULTS_PATH.exists())
                self.assertTrue(harness.REPORT_PATH.exists())
                results = json.loads(harness.RESULTS_PATH.read_text(encoding="utf-8"))
                self.assertEqual(results["runs"][0]["images"][0]["verdict"], "SLIDE")
                self.assertGreaterEqual(mock_server.state["load_calls"], 1)
                self.assertEqual(mock_server.state["chat_calls"], 1)
        finally:
            harness.RESULTS_PATH = old_results
            harness.REPORT_PATH = old_report
            harness_web.SESSIONS.clear()
            harness_web.SESSIONS.update(old_sessions)
            harness_web.JOBS.clear()
            harness_web.JOBS.update(old_jobs)
            mock_server.stop()

    def test_create_test_copies_baseline_to_separate_results(self) -> None:
        old_results = harness.RESULTS_PATH
        old_report = harness.REPORT_PATH
        old_prompts = harness.PROMPTS_DIR
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                with self.isolated_web_paths(root):
                    harness.RESULTS_PATH = root / "results.json"
                    harness.REPORT_PATH = root / "report.md"
                    harness.PROMPTS_DIR = root / "prompts"
                    harness.PROMPTS_DIR.mkdir()
                    (harness.PROMPTS_DIR / "classify_v1.txt").write_text("baseline prompt", encoding="utf-8")

                    source_session = harness_web.SESSION_DIR / "session_source"
                    source_session.mkdir(parents=True)
                    (source_session / "slide_0001.png").write_bytes(b"image")
                    (source_session / "slide_0002.png").write_bytes(b"image")

                    source_results = harness.fresh_results(Path("browser:batchImageTest"))
                    source_results["codex_model"] = "Codex manual baseline"
                    source_results["codex_verdicts"] = {
                        "slide_0001.png": "SLIDE",
                        "slide_0002.png": "NOT SLIDE",
                    }
                    source_results["runs"] = [{"run_id": "run_001", "prompt": "classify_v1", "images": []}]
                    harness.save_results(source_results, harness.RESULTS_PATH)

                    harness_web.ensure_test_registry()
                    harness_web.update_test_profile("vision_test_1", "VISION_TURBO")
                    test = harness_web.create_test("Vision Test #2", "classify_v2", "vision_test_1")
                    new_results_path, new_report_path = harness_web.test_paths(str(test["id"]))
                    new_results = harness.load_results(new_results_path)

                    self.assertEqual(test["id"], "vision_test_2")
                    self.assertEqual(harness_web.active_test_id(), "vision_test_2")
                    self.assertEqual(new_results["codex_verdicts"], source_results["codex_verdicts"])
                    self.assertEqual(new_results["codex_model"], "Codex manual baseline")
                    self.assertEqual(new_results["runs"], [])
                    self.assertTrue(new_report_path.exists())
                    self.assertEqual(test["default_profile"], "VISION_TURBO")
                    self.assertEqual((harness.PROMPTS_DIR / "classify_v2.txt").read_text(encoding="utf-8"), "baseline prompt")
        finally:
            harness.RESULTS_PATH = old_results
            harness.REPORT_PATH = old_report
            harness.PROMPTS_DIR = old_prompts

    def test_session_from_test_reuses_persisted_image_folder(self) -> None:
        old_results = harness.RESULTS_PATH
        old_report = harness.REPORT_PATH
        old_prompts = harness.PROMPTS_DIR
        old_sessions = dict(harness_web.SESSIONS)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                with self.isolated_web_paths(root):
                    harness.RESULTS_PATH = root / "results.json"
                    harness.REPORT_PATH = root / "report.md"
                    harness.PROMPTS_DIR = root / "prompts"
                    harness.PROMPTS_DIR.mkdir()
                    (harness.PROMPTS_DIR / "classify_v1.txt").write_text("prompt", encoding="utf-8")

                    source_session = harness_web.SESSION_DIR / "session_source"
                    source_session.mkdir(parents=True)
                    (source_session / "b.png").write_bytes(b"image")
                    (source_session / "a.png").write_bytes(b"image")

                    source_results = harness.fresh_results(Path("browser:batchImageTest"))
                    source_results["codex_verdicts"] = {"a.png": "SLIDE", "b.png": "NOT SLIDE"}
                    harness.save_results(source_results, harness.RESULTS_PATH)
                    harness_web.SESSIONS.clear()

                    harness_web.ensure_test_registry()
                    session = harness_web.session_from_test("vision_test_1")

                    self.assertIsNotNone(session)
                    self.assertEqual(session["images"], ["a.png", "b.png"])
                    self.assertIn("test_vision_test_1", harness_web.SESSIONS)
        finally:
            harness.RESULTS_PATH = old_results
            harness.REPORT_PATH = old_report
            harness.PROMPTS_DIR = old_prompts
            harness_web.SESSIONS.clear()
            harness_web.SESSIONS.update(old_sessions)

    def test_clear_test_results_keeps_baseline_and_regenerates_report(self) -> None:
        old_results = harness.RESULTS_PATH
        old_report = harness.REPORT_PATH
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                with self.isolated_web_paths(root):
                    harness.RESULTS_PATH = root / "results.json"
                    harness.REPORT_PATH = root / "report.md"
                    results = harness.fresh_results(Path("browser:batchImageTest"))
                    results["codex_verdicts"] = {"slide_0001.png": "SLIDE"}
                    results["runs"] = [{"run_id": "run_001", "images": []}]
                    harness.save_results(results, harness.RESULTS_PATH)
                    harness.generate_report(results, harness.REPORT_PATH)
                    harness_web.ensure_test_registry()

                    harness_web.clear_test_results("vision_test_1")

                    updated = harness.load_results(harness.RESULTS_PATH)
                    self.assertEqual(updated["codex_verdicts"], {"slide_0001.png": "SLIDE"})
                    self.assertEqual(updated["runs"], [])
                    self.assertIn("Total runs recorded: 0", harness.REPORT_PATH.read_text(encoding="utf-8"))
        finally:
            harness.RESULTS_PATH = old_results
            harness.REPORT_PATH = old_report

    def test_delete_run_removes_only_requested_run(self) -> None:
        old_results = harness.RESULTS_PATH
        old_report = harness.REPORT_PATH
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                with self.isolated_web_paths(root):
                    harness.RESULTS_PATH = root / "results.json"
                    harness.REPORT_PATH = root / "report.md"
                    results = harness.fresh_results(Path("browser:batchImageTest"))
                    results["runs"] = [
                        {"run_id": "run_001", "images": []},
                        {"run_id": "run_002", "images": []},
                    ]
                    harness.save_results(results, harness.RESULTS_PATH)
                    harness.generate_report(results, harness.REPORT_PATH)
                    harness_web.ensure_test_registry()

                    harness_web.delete_run("vision_test_1", "run_001")

                    updated = harness.load_results(harness.RESULTS_PATH)
                    self.assertEqual([run["run_id"] for run in updated["runs"]], ["run_002"])
                    self.assertIn("Total runs recorded: 1", harness.REPORT_PATH.read_text(encoding="utf-8"))
        finally:
            harness.RESULTS_PATH = old_results
            harness.REPORT_PATH = old_report

    def test_delete_prompt_updates_default_prompt(self) -> None:
        old_results = harness.RESULTS_PATH
        old_report = harness.REPORT_PATH
        old_prompts = harness.PROMPTS_DIR
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                with self.isolated_web_paths(root):
                    harness.RESULTS_PATH = root / "results.json"
                    harness.REPORT_PATH = root / "report.md"
                    harness.PROMPTS_DIR = root / "prompts"
                    harness.PROMPTS_DIR.mkdir()
                    (harness.PROMPTS_DIR / "classify_v1.txt").write_text("baseline prompt", encoding="utf-8")
                    (harness.PROMPTS_DIR / "custom_v1.txt").write_text("custom prompt", encoding="utf-8")
                    harness.save_results(harness.fresh_results(Path("browser:batchImageTest")), harness.RESULTS_PATH)
                    harness_web.ensure_test_registry()
                    harness_web.update_test_prompt("vision_test_1", "custom_v1")

                    deleted = harness_web.delete_prompt("custom_v1")
                    tests = harness_web.tests_payload()

                    self.assertEqual(deleted, "custom_v1")
                    self.assertFalse((harness.PROMPTS_DIR / "custom_v1.txt").exists())
                    self.assertEqual(tests["active_test"]["default_prompt"], "classify_v1")
        finally:
            harness.RESULTS_PATH = old_results
            harness.REPORT_PATH = old_report
            harness.PROMPTS_DIR = old_prompts

    def test_delete_prompt_blocks_last_prompt(self) -> None:
        old_prompts = harness.PROMPTS_DIR
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                harness.PROMPTS_DIR = root / "prompts"
                harness.PROMPTS_DIR.mkdir()
                (harness.PROMPTS_DIR / "classify_v1.txt").write_text("baseline prompt", encoding="utf-8")

                with self.assertRaises(harness.HarnessError):
                    harness_web.delete_prompt("classify_v1")
        finally:
            harness.PROMPTS_DIR = old_prompts

    def test_run_model_batch_stops_after_model_level_failure_retry(self) -> None:
        old_jobs = dict(harness_web.JOBS)
        try:
            harness_web.JOBS.clear()
            harness_web.JOBS["job_failure"] = {
                "id": "job_failure",
                "completed_images": 0,
                "logs": [],
            }
            images = [Path(f"slide_{index:04d}.png") for index in range(1, 4)]
            results = harness.fresh_results(Path("browser:batch"))
            failure = harness.error_image_result(
                "LM Studio returned HTTP 400: No models loaded. Please load a model.",
                response_time_sec=0.1,
            )

            with patch.object(harness, "ensure_lm_studio_model_loaded") as load_mock:
                with patch.object(harness, "call_lm_studio", return_value=failure) as call_mock:
                    run = harness_web.run_model_batch(
                        job_id="job_failure",
                        results=results,
                        model="crashy-vision",
                        prompt_name="classify_v1",
                        prompt_text="prompt",
                        prompt_hash="hash",
                        profile_name="VISION_TURBO",
                        load_info={"load_config_applied": {}, "load_time_seconds": None},
                        images=images,
                        base_url="http://127.0.0.1:1234",
                        timeout_seconds=10,
                    )

            self.assertEqual(call_mock.call_count, 2)
            self.assertEqual(load_mock.call_count, 1)
            self.assertEqual(len(run["images"]), 3)
            self.assertEqual(run["profile"], "VISION_TURBO")
            self.assertEqual(run["inference_config"]["max_tokens"], 175)
            self.assertEqual(run["requested_load_config"]["context_length"], 16384)
            self.assertEqual(run["requested_load_config"]["eval_batch_size"], 1024)
            self.assertEqual(run["batch_stats"]["error_count"], 3)
            self.assertEqual(harness_web.JOBS["job_failure"]["completed_images"], 3)
        finally:
            harness_web.JOBS.clear()
            harness_web.JOBS.update(old_jobs)


class MockLMStudioServer:
    def __init__(self) -> None:
        self.state = {"loaded": False, "load_calls": 0, "chat_calls": 0}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self.handler_class())
        self.server.state = self.state  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def handler_class(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):  # type: ignore[no-untyped-def]
                return

            def do_GET(self):  # type: ignore[no-untyped-def]
                if self.path == "/api/v1/models":
                    loaded = [{"id": "mock-vision:1"}] if self.server.state["loaded"] else []  # type: ignore[attr-defined]
                    self.send_payload(
                        {
                            "models": [
                                {
                                    "key": "mock-vision",
                                    "type": "llm",
                                    "capabilities": ["text", "vision"],
                                    "loaded_instances": loaded,
                                }
                            ]
                        }
                    )
                elif self.path == "/v1/models":
                    self.send_payload({"data": [{"id": "mock-vision"}]})
                else:
                    self.send_error(404)

            def do_POST(self):  # type: ignore[no-untyped-def]
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                if self.path == "/api/v1/models/load":
                    self.server.state["loaded"] = True  # type: ignore[attr-defined]
                    self.server.state["load_calls"] += 1  # type: ignore[attr-defined]
                    self.send_payload({"instance_id": "mock-vision:1"})
                elif self.path == "/api/v1/models/unload":
                    self.server.state["loaded"] = False  # type: ignore[attr-defined]
                    self.send_payload({"ok": True})
                elif self.path == "/v1/chat/completions":
                    self.server.state["chat_calls"] += 1  # type: ignore[attr-defined]
                    self.send_payload(
                        {
                            "choices": [
                                {
                                    "message": {
                                        "content": (
                                            "DESCRIPTION: A presentation slide fills the frame.\n"
                                            "EVIDENCE: TEXT: \"Example\" | PERSON: no | BLANK: no\n"
                                            "VERDICT: SLIDE"
                                        )
                                    }
                                }
                            ],
                            "usage": {"completion_tokens": 12, "prompt_tokens": 100},
                        }
                    )
                else:
                    self.send_error(404)

            def send_payload(self, payload):  # type: ignore[no-untyped-def]
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()


if __name__ == "__main__":
    unittest.main()
