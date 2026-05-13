#!/usr/bin/env python3
"""Local Web UI for the LM Studio vision-model harness."""

from __future__ import annotations

import argparse
import cgi
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import parse_qs, unquote, urlparse

import harness


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "harness_web_static"
SESSION_DIR = ROOT / ".harness_sessions"
TESTS_DIR = ROOT / ".harness_tests"
TEST_REGISTRY_PATH = TESTS_DIR / "registry.json"
DEFAULT_TEST_ID = "vision_test_1"
KEYCHAIN_SERVICE = "Vision Harness"
KEYCHAIN_ACCOUNT = "lm_studio_api_token"

SESSIONS: Dict[str, Dict[str, Any]] = {}
JOBS: Dict[str, Dict[str, Any]] = {}
STATE_LOCK = threading.Lock()
RESULTS_LOCK = threading.Lock()
TEST_LOCK = threading.Lock()


def json_bytes(payload: Dict[str, Any], status: int = 200) -> bytes:
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


class HarnessWebHandler(SimpleHTTPRequestHandler):
    server_version = "VisionHarnessWeb/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path == "/":
                self.send_static_file(STATIC_DIR / "index.html")
            elif path.startswith("/static/"):
                self.send_static_file(STATIC_DIR / path.removeprefix("/static/"))
            elif path == "/api/config":
                tests = tests_payload()
                session = session_from_test(str(tests.get("active_test_id") or DEFAULT_TEST_ID))
                results_path, report_path = test_paths(str(tests.get("active_test_id") or DEFAULT_TEST_ID))
                self.send_json(
                    {
                        "base_url": harness.DEFAULT_BASE_URL,
                        "prompts": prompt_summaries(),
                        "profiles": profile_summaries(),
                        "has_results": results_path.exists(),
                        "has_report": report_path.exists(),
                        "token": token_status(),
                        "tests": tests["tests"],
                        "active_test_id": tests["active_test_id"],
                        "active_test": tests["active_test"],
                        "session": session,
                    }
                )
            elif path == "/api/token":
                self.send_json({"token": token_status()})
            elif path == "/api/tests":
                self.send_json(tests_payload())
            elif path == "/api/prompts":
                self.send_json({"prompts": prompt_summaries()})
            elif path.startswith("/api/prompts/"):
                name = unquote(path.removeprefix("/api/prompts/"))
                self.send_json(prompt_payload(name))
            elif path == "/api/models":
                base_url = single_query_value(query, "base_url") or harness.DEFAULT_BASE_URL
                models = harness.fetch_lm_studio_models(base_url, timeout_seconds=20)
                self.send_json(models)
            elif path == "/api/jobs":
                test_id = single_query_value(query, "test_id")
                self.send_json({"jobs": job_summaries(test_id), "latest": latest_job_snapshot(test_id)})
            elif path.startswith("/api/jobs/"):
                job_id = unquote(path.removeprefix("/api/jobs/"))
                self.send_json(job_snapshot(job_id))
            elif path == "/api/results":
                self.send_json(results_payload())
            elif path == "/api/report":
                _, report_path = test_paths(active_test_id())
                self.send_text(report_path.read_text(encoding="utf-8") if report_path.exists() else "")
            else:
                self.send_error(404, "Not found")
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/sessions":
                self.handle_session_upload()
            elif parsed.path == "/api/tests":
                self.handle_test_create()
            elif parsed.path == "/api/tests/active":
                self.handle_test_switch()
            elif parsed.path == "/api/tests/profile":
                self.handle_test_profile()
            elif parsed.path == "/api/tests/prompt":
                self.handle_test_prompt()
            elif parsed.path == "/api/jobs":
                self.handle_job_start()
            elif parsed.path == "/api/token":
                self.handle_token_save()
            else:
                self.send_error(404, "Not found")
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/tests/results":
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                test_id = str(payload.get("test_id") or single_query_value(query, "test_id") or active_test_id())
                clear_test_results(test_id)
                self.send_json({"results": results_payload(), "tests": tests_payload(), "report": current_report_text()})
            elif parsed.path.startswith("/api/runs/"):
                run_id = unquote(parsed.path.removeprefix("/api/runs/"))
                test_id = single_query_value(query, "test_id") or active_test_id()
                delete_run(test_id, run_id)
                self.send_json({"results": results_payload(), "tests": tests_payload(), "report": current_report_text()})
            elif parsed.path.startswith("/api/prompts/"):
                name = unquote(parsed.path.removeprefix("/api/prompts/"))
                deleted = delete_prompt(name)
                self.send_json(
                    {
                        "deleted": deleted,
                        "prompts": prompt_summaries(),
                        "tests": tests_payload(),
                    }
                )
            else:
                self.send_error(404, "Not found")
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/prompts/"):
                name = unquote(parsed.path.removeprefix("/api/prompts/"))
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                text = str(payload.get("text") or "")
                prompt_name = save_prompt(name, text)
                self.send_json(prompt_payload(prompt_name))
            else:
                self.send_error(404, "Not found")
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def handle_session_upload(self) -> None:
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type"),
            },
        )
        folder_label = str(form.getfirst("folder_label") or "browser-folder").strip() or "browser-folder"
        session_id = f"session_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        target_dir = SESSION_DIR / session_id
        target_dir.mkdir(parents=True, exist_ok=True)

        fields = form["files"] if "files" in form else []
        if not isinstance(fields, list):
            fields = [fields]
        saved: List[Path] = []
        seen: set = set()
        for field in fields:
            if not getattr(field, "filename", ""):
                continue
            safe_name = safe_upload_name(str(field.filename), seen)
            if Path(safe_name).suffix.lower() not in harness.IMAGE_EXTENSIONS:
                continue
            output_path = target_dir / safe_name
            with output_path.open("wb") as handle:
                handle.write(field.file.read())
            saved.append(output_path)

        saved = sorted(saved, key=lambda path: path.name)
        if not saved:
            raise harness.HarnessError("No supported images were uploaded. Use .jpg, .jpeg, or .png files.")

        session = {
            "id": session_id,
            "folder_label": folder_label,
            "batch_path": f"browser:{folder_label}",
            "path": str(target_dir),
            "images": [path.name for path in saved],
            "created_at": harness.utc_now(),
        }
        with STATE_LOCK:
            SESSIONS[session_id] = session
        update_test_after_session(session)
        self.send_json(session)

    def handle_test_create(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        source_test_id = str(payload.get("source_test_id") or active_test_id())
        name = str(payload.get("name") or "").strip()
        prompt_name = str(payload.get("prompt_name") or "classify_v2").strip()
        test = create_test(name, prompt_name, source_test_id)
        session = session_from_test(str(test["id"]))
        tests = tests_payload()
        self.send_json(
            {
                "test": public_test_summary(test),
                "tests": tests["tests"],
                "active_test_id": tests["active_test_id"],
                "active_test": tests["active_test"],
                "session": session,
                "prompts": prompt_summaries(),
            }
        )

    def handle_test_switch(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        test_id = str(payload.get("test_id") or "").strip()
        if not test_id:
            raise harness.HarnessError("Choose a vision test first.")
        tests = set_active_test(test_id)
        session = session_from_test(test_id)
        self.send_json(
            {
                "tests": tests["tests"],
                "active_test_id": tests["active_test_id"],
                "active_test": tests["active_test"],
                "session": session,
            }
        )

    def handle_test_profile(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        test_id = str(payload.get("test_id") or active_test_id()).strip()
        profile_name = harness.normalize_profile_name(str(payload.get("profile") or harness.DEFAULT_PROFILE))
        update_test_profile(test_id, profile_name)
        tests = tests_payload()
        self.send_json(
            {
                "tests": tests["tests"],
                "active_test_id": tests["active_test_id"],
                "active_test": tests["active_test"],
            }
        )

    def handle_test_prompt(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        test_id = str(payload.get("test_id") or active_test_id()).strip()
        prompt_name = str(payload.get("prompt") or "").strip()
        if not prompt_name:
            raise harness.HarnessError("Choose a prompt first.")
        harness.load_prompt(prompt_name)
        update_test_prompt(test_id, prompt_name)
        tests = tests_payload()
        self.send_json(
            {
                "tests": tests["tests"],
                "active_test_id": tests["active_test_id"],
                "active_test": tests["active_test"],
            }
        )

    def handle_job_start(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        session_id = str(payload.get("session_id") or "")
        test_id = str(payload.get("test_id") or active_test_id())
        prompt = str(payload.get("prompt") or "")
        profile = harness.normalize_profile_name(str(payload.get("profile") or test_profile(test_id)))
        models = [str(model) for model in payload.get("models") or [] if str(model).strip()]
        base_url = str(payload.get("base_url") or harness.DEFAULT_BASE_URL).strip() or harness.DEFAULT_BASE_URL
        timeout_seconds = max(10, min(600, int(payload.get("timeout") or harness.DEFAULT_TIMEOUT_SECONDS)))

        with STATE_LOCK:
            session = SESSIONS.get(session_id)
        if not session:
            session = session_from_test(test_id)
            session_id = str(session.get("id") if session else "")
        if not session:
            raise harness.HarnessError("Upload an image folder before starting a run.")
        if not prompt:
            raise harness.HarnessError("Choose a prompt before starting a run.")
        if not models:
            raise harness.HarnessError("Choose at least one model before starting a run.")
        update_test_prompt(test_id, prompt)
        update_test_profile(test_id, profile)

        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        job = {
            "id": job_id,
            "status": "queued",
            "test_id": test_id,
            "session_id": session_id,
            "models": models,
            "prompt": prompt,
            "profile": profile,
            "base_url": base_url,
            "timeout": timeout_seconds,
            "current_model": "",
            "current_image": "",
            "completed_images": 0,
            "total_images": len(session["images"]) * len(models),
            "logs": [],
            "error": "",
            "runs": [],
            "started_at": harness.utc_now(),
            "finished_at": "",
        }
        with STATE_LOCK:
            JOBS[job_id] = job
        thread = threading.Thread(target=run_job, args=(job_id,), daemon=True)
        thread.start()
        self.send_json(job)

    def handle_token_save(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        token = str(payload.get("token") or "").strip()
        if not token:
            raise harness.HarnessError("Paste an LM Studio API token before saving.")
        save_keychain_token(token)
        self.send_json({"token": token_status()})

    def send_static_file(self, path: Path) -> None:
        if not path.resolve().is_relative_to(STATIC_DIR.resolve()):
            self.send_error(403, "Forbidden")
            return
        if not path.exists() or not path.is_file():
            self.send_error(404, "Not found")
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json_bytes(payload, status=status)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, status: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/markdown; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def safe_upload_name(filename: str, seen: set) -> str:
    cleaned = filename.replace("\\", "/").strip("/ ")
    parts = [part for part in cleaned.split("/") if part not in {"", ".", ".."}]
    name = parts[-1] if parts else "image"
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .") or "image"
    stem = Path(name).stem or "image"
    suffix = Path(name).suffix.lower()
    candidate = f"{stem}{suffix}"
    index = 2
    while candidate in seen:
        candidate = f"{stem}_{index}{suffix}"
        index += 1
    seen.add(candidate)
    return candidate


def single_query_value(query: Dict[str, List[str]], key: str) -> str:
    values = query.get(key) or []
    return values[0] if values else ""


def registry_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def path_from_registry(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def image_names_from_results(results: Dict[str, Any]) -> List[str]:
    codex = results.get("codex_verdicts")
    if isinstance(codex, dict) and codex:
        return sorted(str(name) for name in codex.keys())
    runs = [run for run in results.get("runs") or [] if isinstance(run, dict)]
    for run in reversed(runs):
        images = run.get("images")
        if isinstance(images, list) and images:
            return sorted(
                str(item.get("filename"))
                for item in images
                if isinstance(item, dict) and item.get("filename")
            )
    return []


def find_session_for_results(results: Dict[str, Any]) -> Optional[Path]:
    expected = set(image_names_from_results(results))
    if not expected or not SESSION_DIR.exists():
        return None

    matches: List[tuple] = []
    for candidate in SESSION_DIR.iterdir():
        if not candidate.is_dir():
            continue
        images = {path.name for path in harness.discover_images(candidate)}
        if expected.issubset(images):
            exact = images == expected
            matches.append((exact, candidate.stat().st_mtime, candidate))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return matches[0][2]


def latest_prompt_name(results: Dict[str, Any]) -> str:
    runs = [run for run in results.get("runs") or [] if isinstance(run, dict)]
    for run in reversed(runs):
        prompt = str(run.get("prompt") or "").strip()
        if prompt:
            return prompt
    prompts = harness.available_prompts()
    return "classify_v1" if "classify_v1" in prompts else (prompts[0] if prompts else "classify_v1")


def profile_summaries() -> List[Dict[str, Any]]:
    return [
        {
            "name": name,
            "load": dict(harness.PROFILES[name]["load"]),
            "inference": dict(harness.PROFILES[name]["inference"]),
        }
        for name in harness.profile_names_for_task("VISION")
    ]


def sanitize_prompt_name(name: str, fallback: str) -> str:
    stem = Path(str(name or fallback)).stem
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_-")
    return cleaned or fallback


def ensure_prompt_copy(prompt_name: str, source_prompt: str) -> str:
    prompt_name = sanitize_prompt_name(prompt_name, "classify_v2")
    path = harness.prompt_path(prompt_name)
    if path.exists():
        return path.stem

    source_name = sanitize_prompt_name(source_prompt, "classify_v1")
    source_path = harness.prompt_path(source_name)
    if source_path.exists():
        text = source_path.read_text(encoding="utf-8")
    else:
        text = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path.stem


def load_test_registry_unlocked() -> Dict[str, Any]:
    if TEST_REGISTRY_PATH.exists():
        try:
            registry = json.loads(TEST_REGISTRY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise harness.HarnessError(f"Test registry is malformed: {TEST_REGISTRY_PATH.name}") from exc
        if isinstance(registry, dict) and isinstance(registry.get("tests"), list):
            registry.setdefault("active_test_id", DEFAULT_TEST_ID)
            registry.setdefault("created_at", harness.utc_now())
            return registry

    results = harness.load_results(harness.RESULTS_PATH) or harness.fresh_results()
    source_dir = find_session_for_results(results)
    image_names = image_names_from_results(results)
    batch_path = str(results.get("batch_path") or "")
    test = {
        "id": DEFAULT_TEST_ID,
        "name": "Vision Test #1",
        "created_at": results.get("created_at") or harness.utc_now(),
        "results_path": registry_path(harness.RESULTS_PATH),
        "report_path": registry_path(harness.REPORT_PATH),
        "batch_path": batch_path,
        "image_source_dir": registry_path(source_dir) if source_dir else "",
        "image_count": len(image_names),
        "default_prompt": latest_prompt_name(results),
        "default_profile": harness.DEFAULT_PROFILE,
    }
    registry = {
        "created_at": harness.utc_now(),
        "active_test_id": DEFAULT_TEST_ID,
        "tests": [test],
    }
    save_test_registry_unlocked(registry)
    return registry


def save_test_registry_unlocked(registry: Dict[str, Any]) -> None:
    TEST_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = TEST_REGISTRY_PATH.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(TEST_REGISTRY_PATH)


def ensure_test_registry() -> Dict[str, Any]:
    with TEST_LOCK:
        return load_test_registry_unlocked()


def find_test_unlocked(registry: Dict[str, Any], test_id: str) -> Dict[str, Any]:
    for test in registry.get("tests") or []:
        if isinstance(test, dict) and test.get("id") == test_id:
            return test
    raise harness.HarnessError("Vision test was not found.")


def active_test_id() -> str:
    registry = ensure_test_registry()
    return str(registry.get("active_test_id") or DEFAULT_TEST_ID)


def test_paths(test_id: str) -> tuple[Path, Path]:
    registry = ensure_test_registry()
    test = find_test_unlocked(registry, test_id)
    return path_from_registry(str(test.get("results_path") or "")), path_from_registry(str(test.get("report_path") or ""))


def job_result_paths(test_id: str) -> tuple[Path, Path]:
    if not test_id:
        return harness.RESULTS_PATH, harness.REPORT_PATH
    return test_paths(test_id)


def public_test_summary(test: Dict[str, Any]) -> Dict[str, Any]:
    results_path = path_from_registry(str(test.get("results_path") or ""))
    report_path = path_from_registry(str(test.get("report_path") or ""))
    results = harness.load_results(results_path) or {}
    image_source_dir = str(test.get("image_source_dir") or "")
    image_count = int(test.get("image_count") or 0)
    if image_source_dir:
        source_path = path_from_registry(image_source_dir)
        if source_path.exists():
            image_count = len(harness.discover_images(source_path))
    runs = [run for run in results.get("runs") or [] if isinstance(run, dict)]
    return {
        "id": test.get("id") or "",
        "name": test.get("name") or "",
        "created_at": test.get("created_at") or "",
        "batch_path": results.get("batch_path") or test.get("batch_path") or "",
        "default_prompt": test.get("default_prompt") or latest_prompt_name(results),
        "default_profile": harness.normalize_profile_name(str(test.get("default_profile") or harness.DEFAULT_PROFILE)),
        "image_count": image_count,
        "baseline_count": len(harness.known_codex_verdicts(results)),
        "run_count": len(runs),
        "results_exists": results_path.exists(),
        "report_exists": report_path.exists(),
    }


def tests_payload() -> Dict[str, Any]:
    registry = ensure_test_registry()
    active_id = str(registry.get("active_test_id") or DEFAULT_TEST_ID)
    summaries = []
    for test in registry.get("tests") or []:
        if isinstance(test, dict):
            summary = public_test_summary(test)
            summary["active"] = summary["id"] == active_id
            summaries.append(summary)
    active = next((test for test in summaries if test.get("active")), summaries[0] if summaries else {})
    return {"active_test_id": active_id, "active_test": active, "tests": summaries}


def session_from_test(test_id: str) -> Optional[Dict[str, Any]]:
    registry = ensure_test_registry()
    test = find_test_unlocked(registry, test_id)
    source_dir = str(test.get("image_source_dir") or "")
    if not source_dir:
        return None
    source_path = path_from_registry(source_dir)
    if not source_path.exists():
        return None
    images = [path.name for path in harness.discover_images(source_path)]
    if not images:
        return None
    batch_path = str(test.get("batch_path") or "")
    folder_label = batch_path.removeprefix("browser:") if batch_path.startswith("browser:") else source_path.name
    session = {
        "id": f"test_{test_id}",
        "folder_label": folder_label or source_path.name,
        "batch_path": batch_path or f"browser:{folder_label or source_path.name}",
        "path": str(source_path),
        "images": images,
        "created_at": test.get("created_at") or harness.utc_now(),
    }
    with STATE_LOCK:
        SESSIONS[session["id"]] = session
    return session


def set_active_test(test_id: str) -> Dict[str, Any]:
    with TEST_LOCK:
        registry = load_test_registry_unlocked()
        find_test_unlocked(registry, test_id)
        registry["active_test_id"] = test_id
        save_test_registry_unlocked(registry)
    return tests_payload()


def update_test_after_session(session: Dict[str, Any]) -> None:
    with TEST_LOCK:
        registry = load_test_registry_unlocked()
        test = find_test_unlocked(registry, str(registry.get("active_test_id") or DEFAULT_TEST_ID))
        results_path = path_from_registry(str(test.get("results_path") or ""))
        results = harness.load_results(results_path) or {}
        existing_batch = str(results.get("batch_path") or "").strip()
        if existing_batch and existing_batch != str(session["batch_path"]):
            return
        test["batch_path"] = session["batch_path"]
        test["image_source_dir"] = registry_path(Path(str(session["path"])))
        test["image_count"] = len(session.get("images") or [])
        save_test_registry_unlocked(registry)


def update_test_prompt(test_id: str, prompt_name: str) -> None:
    if not test_id:
        return
    with TEST_LOCK:
        registry = load_test_registry_unlocked()
        test = find_test_unlocked(registry, test_id)
        test["default_prompt"] = prompt_name
        save_test_registry_unlocked(registry)


def test_profile(test_id: str) -> str:
    registry = ensure_test_registry()
    test = find_test_unlocked(registry, test_id)
    return harness.normalize_profile_name(str(test.get("default_profile") or harness.DEFAULT_PROFILE))


def update_test_profile(test_id: str, profile_name: str) -> None:
    if not test_id:
        return
    profile_name = harness.normalize_profile_name(profile_name)
    with TEST_LOCK:
        registry = load_test_registry_unlocked()
        test = find_test_unlocked(registry, test_id)
        test["default_profile"] = profile_name
        save_test_registry_unlocked(registry)


def test_has_active_job(test_id: str) -> bool:
    with STATE_LOCK:
        return any(
            str(job.get("test_id") or "") == test_id and str(job.get("status") or "") in {"queued", "running"}
            for job in JOBS.values()
        )


def clear_test_results(test_id: str) -> None:
    if test_has_active_job(test_id):
        raise harness.HarnessError("This test has a run in progress. Wait for it to finish before clearing results.")
    results_path, report_path = test_paths(test_id)
    with RESULTS_LOCK:
        results = harness.load_results(results_path) or harness.fresh_results()
        results["runs"] = []
        harness.save_results(results, results_path)
        harness.generate_report(results, report_path)


def delete_run(test_id: str, run_id: str) -> None:
    if test_has_active_job(test_id):
        raise harness.HarnessError("This test has a run in progress. Wait for it to finish before deleting a run.")
    clean_run_id = str(run_id or "").strip()
    if not clean_run_id:
        raise harness.HarnessError("Choose a run to delete.")
    results_path, report_path = test_paths(test_id)
    with RESULTS_LOCK:
        results = harness.load_results(results_path) or harness.fresh_results()
        runs = [run for run in results.get("runs") or [] if isinstance(run, dict)]
        remaining = [run for run in runs if str(run.get("run_id") or "") != clean_run_id]
        if len(remaining) == len(runs):
            raise harness.HarnessError(f"Run '{clean_run_id}' was not found.")
        results["runs"] = remaining
        harness.save_results(results, results_path)
        harness.generate_report(results, report_path)


def next_test_number(registry: Dict[str, Any]) -> int:
    highest = 0
    for test in registry.get("tests") or []:
        match = re.match(r"vision_test_(\d+)$", str(test.get("id") or ""))
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def create_test(name: str, prompt_name: str, source_test_id: str) -> Dict[str, Any]:
    with TEST_LOCK:
        registry = load_test_registry_unlocked()
        source = find_test_unlocked(registry, source_test_id or str(registry.get("active_test_id") or DEFAULT_TEST_ID))
        source_results_path = path_from_registry(str(source.get("results_path") or ""))
        source_results = harness.load_results(source_results_path) or harness.fresh_results()
        source_prompt = str(source.get("default_prompt") or latest_prompt_name(source_results))
        source_profile = harness.normalize_profile_name(str(source.get("default_profile") or harness.DEFAULT_PROFILE))
        prompt = ensure_prompt_copy(prompt_name, source_prompt)

        source_dir = str(source.get("image_source_dir") or "")
        if not source_dir:
            found = find_session_for_results(source_results)
            source_dir = registry_path(found) if found else ""
            if source_dir:
                source["image_source_dir"] = source_dir

        number = next_test_number(registry)
        test_id = f"vision_test_{number}"
        display_name = str(name or f"Vision Test #{number}").strip() or f"Vision Test #{number}"
        test_dir = TESTS_DIR / test_id
        results_path = test_dir / "results.json"
        report_path = test_dir / "report.md"
        batch_path = str(source_results.get("batch_path") or source.get("batch_path") or "")

        new_results = harness.fresh_results(Path(batch_path) if batch_path else None)
        new_results["codex_model"] = source_results.get("codex_model") or ""
        new_results["codex_verdicts"] = dict(source_results.get("codex_verdicts") or {})
        new_results["runs"] = []
        results_path.parent.mkdir(parents=True, exist_ok=True)
        harness.save_results(new_results, results_path)
        harness.generate_report(new_results, report_path)

        test = {
            "id": test_id,
            "name": display_name,
            "created_at": harness.utc_now(),
            "results_path": registry_path(results_path),
            "report_path": registry_path(report_path),
            "batch_path": batch_path,
            "image_source_dir": source_dir,
            "image_count": len(image_names_from_results(source_results)),
            "default_prompt": prompt,
            "default_profile": source_profile,
        }
        registry.setdefault("tests", []).append(test)
        registry["active_test_id"] = test_id
        save_test_registry_unlocked(registry)
    return test


def prompt_summaries() -> List[Dict[str, str]]:
    prompts = []
    for name in harness.available_prompts():
        prompt_name, text, prompt_hash = harness.load_prompt(name)
        prompts.append({"name": prompt_name, "hash": prompt_hash, "text": text})
    return prompts


def prompt_payload(name: str) -> Dict[str, str]:
    prompt_name, text, prompt_hash = harness.load_prompt(name)
    return {"name": prompt_name, "hash": prompt_hash, "text": text}


def save_prompt(name: str, text: str) -> str:
    path = harness.prompt_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path.stem


def delete_prompt(name: str) -> str:
    prompt_name = harness.load_prompt(name)[0]
    prompts = harness.available_prompts()
    if len(prompts) <= 1:
        raise harness.HarnessError("Keep at least one prompt available.")
    path = harness.prompt_path(prompt_name)
    path.unlink()
    remaining = harness.available_prompts()
    replacement = remaining[0] if remaining else ""
    with TEST_LOCK:
        registry = load_test_registry_unlocked()
        for test in registry.get("tests") or []:
            if isinstance(test, dict) and test.get("default_prompt") == prompt_name:
                test["default_prompt"] = replacement
        save_test_registry_unlocked(registry)
    return prompt_name


def current_report_text() -> str:
    _, report_path = test_paths(active_test_id())
    return report_path.read_text(encoding="utf-8") if report_path.exists() else ""


def token_status() -> Dict[str, Any]:
    env_saved = bool(str(os.environ.get("LM_STUDIO_API_KEY") or os.environ.get("LM_API_TOKEN") or "").strip())
    keychain_saved = bool(read_keychain_token())
    return {
        "saved": env_saved or keychain_saved,
        "source": "environment" if env_saved else ("keychain" if keychain_saved else ""),
        "keychain_service": KEYCHAIN_SERVICE,
    }


def read_keychain_token() -> str:
    if sys.platform != "darwin":
        return ""
    try:
        output = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-a",
                KEYCHAIN_ACCOUNT,
                "-s",
                KEYCHAIN_SERVICE,
                "-w",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return ""
    if output.returncode != 0:
        return ""
    return output.stdout.strip()


def save_keychain_token(token: str) -> None:
    if sys.platform != "darwin":
        raise harness.HarnessError("Secure token saving is only available through macOS Keychain.")
    output = subprocess.run(
        [
            "security",
            "add-generic-password",
            "-a",
            KEYCHAIN_ACCOUNT,
            "-s",
            KEYCHAIN_SERVICE,
            "-w",
            token,
            "-U",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if output.returncode != 0:
        detail = (output.stderr or output.stdout or "").strip()
        raise harness.HarnessError(f"Could not save token to macOS Keychain. {detail}".strip())


def job_snapshot(job_id: str) -> Dict[str, Any]:
    with STATE_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise harness.HarnessError("Job was not found.")
        return json.loads(json.dumps(job))


def job_summaries(test_id: str = "") -> List[Dict[str, Any]]:
    with STATE_LOCK:
        jobs = [
            job
            for job in JOBS.values()
            if not test_id or str(job.get("test_id") or "") == test_id
        ]
        jobs = sorted(
            jobs,
            key=lambda item: str(item.get("started_at") or item.get("id") or ""),
            reverse=True,
        )
        return [
            {
                "id": job.get("id"),
                "test_id": job.get("test_id"),
                "status": job.get("status"),
                "profile": job.get("profile"),
                "current_model": job.get("current_model"),
                "current_image": job.get("current_image"),
                "completed_images": job.get("completed_images"),
                "total_images": job.get("total_images"),
                "started_at": job.get("started_at"),
                "finished_at": job.get("finished_at"),
                "error": job.get("error"),
            }
            for job in jobs
        ]


def latest_job_snapshot(test_id: str = "") -> Dict[str, Any]:
    summaries = job_summaries(test_id)
    if not summaries:
        return {}
    return job_snapshot(str(summaries[0]["id"]))


def append_job_log(job_id: str, message: str) -> None:
    with STATE_LOCK:
        job = JOBS[job_id]
        job.setdefault("logs", []).append({"at": harness.utc_now(), "message": message})
        job["logs"] = job["logs"][-250:]


def update_job(job_id: str, **fields: Any) -> None:
    with STATE_LOCK:
        JOBS[job_id].update(fields)


def run_job(job_id: str) -> None:
    try:
        with STATE_LOCK:
            job = dict(JOBS[job_id])
            session = dict(SESSIONS[job["session_id"]])

        update_job(job_id, status="running")
        prompt_name, prompt_text, prompt_hash = harness.load_prompt(job["prompt"])
        profile_name = harness.normalize_profile_name(str(job.get("profile") or harness.DEFAULT_PROFILE))
        images = [Path(session["path"]) / name for name in session["images"]]
        model_count = len(job["models"])
        results_path, report_path = job_result_paths(str(job.get("test_id") or ""))

        with RESULTS_LOCK:
            results = harness.load_results(results_path) or harness.fresh_results(Path(session["batch_path"]))
            ensure_browser_batch(results, str(session["batch_path"]), images)

        for model_index, model in enumerate(job["models"], start=1):
            update_job(job_id, current_model=model, current_image="")
            append_job_log(job_id, f"Preparing model {model_index}/{model_count}: {model} ({profile_name})")
            load_info = harness.ensure_lm_studio_model_loaded(
                base_url=job["base_url"],
                model=model,
                timeout_seconds=job["timeout"],
                profile_name=profile_name,
                force_reload=True,
                logger=lambda message, job_id=job_id: append_job_log(job_id, message),
            )

            run = run_model_batch(
                job_id=job_id,
                results=results,
                model=model,
                prompt_name=prompt_name,
                prompt_text=prompt_text,
                prompt_hash=prompt_hash,
                profile_name=profile_name,
                load_info=load_info,
                images=images,
                base_url=job["base_url"],
                timeout_seconds=job["timeout"],
            )
            with RESULTS_LOCK:
                results.setdefault("runs", []).append(run)
                harness.save_results(results, results_path)
                harness.generate_report(results, report_path)

            with STATE_LOCK:
                JOBS[job_id].setdefault("runs", []).append(run)

            if model_index < model_count:
                append_job_log(job_id, f"Finished {model}; offloading before the next model.")

        update_job(job_id, status="complete", finished_at=harness.utc_now(), current_image="")
        append_job_log(job_id, "Testing complete. Results and report are updated.")
    except Exception as exc:
        update_job(job_id, status="failed", error=str(exc), finished_at=harness.utc_now())
        append_job_log(job_id, f"Run failed: {exc}")


def ensure_browser_batch(results: Dict[str, Any], batch_label: str, images: Sequence[Path]) -> None:
    existing = str(results.get("batch_path") or "").strip()
    if existing and existing != batch_label:
        raise harness.HarnessError(
            "results.json already belongs to a different image set. Move or delete results.json before starting a new dataset."
        )
    results["batch_path"] = batch_label
    results.setdefault("created_at", harness.utc_now())
    results.setdefault("codex_model", "")
    if not isinstance(results.get("codex_verdicts"), dict):
        results["codex_verdicts"] = {}
    results.setdefault("runs", [])


def run_model_batch(
    *,
    job_id: str,
    results: Dict[str, Any],
    model: str,
    prompt_name: str,
    prompt_text: str,
    prompt_hash: str,
    profile_name: str,
    load_info: Dict[str, Any],
    images: Sequence[Path],
    base_url: str,
    timeout_seconds: int,
) -> Dict[str, Any]:
    profile_name = harness.normalize_profile_name(profile_name)
    run_id = harness.next_run_id(results.get("runs") or [])
    started = harness.utc_now()
    image_results: List[Dict[str, Any]] = []
    run_start = time.perf_counter()
    reload_attempted = False

    for index, image_path in enumerate(images, start=1):
        update_job(job_id, current_image=image_path.name)
        append_job_log(job_id, f"[{index:02d}/{len(images):02d}] {model} -> {image_path.name}")
        result = harness.call_lm_studio(
            model=model,
            prompt_text=prompt_text,
            image_path=image_path,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            profile_name=profile_name,
        )

        if harness.is_model_level_failure(result) and not reload_attempted:
            reload_attempted = True
            append_job_log(job_id, f"{model} became unavailable. Reloading once and retrying this image.")
            try:
                harness.ensure_lm_studio_model_loaded(
                    base_url=base_url,
                    model=model,
                    timeout_seconds=timeout_seconds,
                    profile_name=profile_name,
                    force_reload=True,
                    logger=lambda message, job_id=job_id: append_job_log(job_id, message),
                )
                result = harness.call_lm_studio(
                    model=model,
                    prompt_text=prompt_text,
                    image_path=image_path,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                    profile_name=profile_name,
                )
            except Exception as exc:
                result = harness.error_image_result(
                    f"Model could not be reloaded after becoming unavailable: {exc}",
                    response_time_sec=0.0,
                )

        result["filename"] = image_path.name
        harness.attach_codex_agreement(result, results.get("codex_verdicts"))
        image_results.append(result)
        with STATE_LOCK:
            JOBS[job_id]["completed_images"] += 1
        append_job_log(job_id, harness.progress_line(index, len(images), image_path.name, str(result["verdict"]), result))

        if harness.is_model_level_failure(result):
            remaining = images[index:]
            if remaining:
                message = (
                    f"{model} is unavailable after a retry. "
                    f"{len(remaining)} remaining images were marked ERROR without additional API calls."
                )
                append_job_log(job_id, message)
                for skipped_path in remaining:
                    skipped = harness.error_image_result(message, response_time_sec=0.0)
                    skipped["filename"] = skipped_path.name
                    harness.attach_codex_agreement(skipped, results.get("codex_verdicts"))
                    image_results.append(skipped)
                with STATE_LOCK:
                    JOBS[job_id]["completed_images"] += len(remaining)
            break

    total_time_sec = time.perf_counter() - run_start
    return {
        "run_id": run_id,
        "model": model,
        "profile": profile_name,
        "prompt": prompt_name,
        "prompt_hash": prompt_hash,
        "inference_config": dict(harness.PROFILES[profile_name]["inference"]),
        "requested_load_config": harness.build_load_config(profile_name, model),
        "load_config_applied": dict(load_info.get("load_config_applied") or {}),
        "load_time_seconds": load_info.get("load_time_seconds"),
        "run_timestamp": started,
        "batch_stats": harness.calculate_batch_stats(image_results, total_time_sec),
        "images": image_results,
    }


def results_payload() -> Dict[str, Any]:
    test_id = active_test_id()
    results_path, report_path = test_paths(test_id)
    tests = tests_payload()
    results = harness.load_results(results_path) or harness.fresh_results()
    runs = [run for run in results.get("runs") or [] if isinstance(run, dict)]
    return {
        "test_id": test_id,
        "test_name": (tests.get("active_test") or {}).get("name") or "",
        "default_profile": (tests.get("active_test") or {}).get("default_profile") or harness.DEFAULT_PROFILE,
        "batch_path": results.get("batch_path") or "",
        "created_at": results.get("created_at") or "",
        "codex_model": results.get("codex_model") or "",
        "codex_verdict_count": len(harness.known_codex_verdicts(results)),
        "runs": runs,
        "report_exists": report_path.exists(),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    harness.set_token_provider(read_keychain_token)
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_test_registry()
    server = ThreadingHTTPServer((args.host, args.port), HarnessWebHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Vision harness Web UI running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
