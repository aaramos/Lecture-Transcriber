#!/usr/bin/env python3
"""Batch image test harness for local LM Studio vision models."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = ROOT / "prompts"
RESULTS_PATH = ROOT / "results.json"
REPORT_PATH = ROOT / "report.md"

DEFAULT_BASE_URL = "http://127.0.0.1:1234"
DEFAULT_TIMEOUT_SECONDS = 120
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
KNOWN_VERDICTS = {"SLIDE", "NOT SLIDE"}
DEFAULT_PROFILE = "VISION_HQ"
PROFILES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "VISION_HQ": {
        "load": {
            "context_length": 32768,
            "flash_attention": True,
            "offload_kv_cache_to_gpu": False,
            "echo_load_config": True,
        },
        "inference": {
            "temperature": 0,
            "top_p": 1.0,
            "top_k": 1,
            "seed": 42,
            "repeat_penalty": 1.0,
            "max_tokens": 250,
            "stop": ["\n\n"],
            "stream": False,
        },
    },
    "VISION_TURBO": {
        "load": {
            "context_length": 16384,
            "flash_attention": True,
            "offload_kv_cache_to_gpu": False,
            "eval_batch_size": 1024,
            "echo_load_config": True,
        },
        "inference": {
            "temperature": 0,
            "top_p": 1.0,
            "top_k": 1,
            "seed": 42,
            "repeat_penalty": 1.0,
            "max_tokens": 175,
            "stop": ["\n\n"],
            "stream": False,
        },
    },
    "SUMMARIZATION_HQ": {
        "load": {
            "context_length": 65536,
            "flash_attention": True,
            "offload_kv_cache_to_gpu": False,
            "echo_load_config": True,
        },
        "inference": {
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 40,
            "seed": 42,
            "repeat_penalty": 1.15,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.15,
            "max_tokens": 1024,
            "stream": False,
        },
    },
    "SUMMARIZATION_TURBO": {
        "load": {
            "context_length": 32768,
            "flash_attention": True,
            "offload_kv_cache_to_gpu": False,
            "eval_batch_size": 1024,
            "echo_load_config": True,
        },
        "inference": {
            "temperature": 0.2,
            "top_p": 0.85,
            "top_k": 20,
            "seed": 42,
            "repeat_penalty": 1.1,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.1,
            "max_tokens": 512,
            "stream": False,
        },
    },
}
MODEL_LEVEL_ERROR_MARKERS = (
    "no models loaded",
    "model has crashed",
    "model is not loaded",
    "model not loaded",
    "not currently loaded",
)
REQUESTS_MODULE: Optional[Any] = None
TOKEN_PROVIDER: Optional[Callable[[], str]] = None


class HarnessError(Exception):
    """Raised for user-facing harness errors."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


FIELD_LABEL_PATTERN = r"DESCRIPTION|EVIDENCE|VERDICT|TITLE|BUILD[_ ]STAGE|LAYOUT"


def parse_response(raw: str) -> Dict[str, Any]:
    result = {
        "description": "",
        "evidence": "",
        "verdict": "UNCERTAIN",
        "title": None,
        "build_stage": None,
        "layout": None,
    }
    text = raw or ""

    description = response_field(text, "DESCRIPTION")
    if description is not None:
        result["description"] = description

    evidence = response_field(text, "EVIDENCE")
    if evidence is not None:
        result["evidence"] = evidence

    # NOT SLIDE must be checked before SLIDE to avoid partial matches.
    if re.search(r"\bNOT SLIDE\b", text, re.IGNORECASE):
        result["verdict"] = "NOT SLIDE"
    elif re.search(r"\bSLIDE\b", text, re.IGNORECASE):
        result["verdict"] = "SLIDE"

    if result["verdict"] == "SLIDE":
        result["title"] = response_field(text, "TITLE")
        result["build_stage"] = response_field(text, r"BUILD[_ ]STAGE")
        result["layout"] = response_field(text, "LAYOUT")

    return result


def response_field(raw: str, label_pattern: str) -> Optional[str]:
    match = re.search(
        rf"^\s*{label_pattern}\s*:\s*(.*?)(?=^\s*(?:{FIELD_LABEL_PATTERN})\s*:|\Z)",
        raw or "",
        re.DOTALL | re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        return None
    return match.group(1).strip()


def encode_image(image_path: Path) -> str:
    with image_path.open("rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def image_mime_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    return "image/jpeg"


def image_data_url(image_path: Path) -> str:
    return f"data:{image_mime_type(image_path)};base64,{encode_image(image_path)}"


def discover_images(batch_path: Path) -> List[Path]:
    return sorted(
        (
            candidate
            for candidate in batch_path.iterdir()
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name,
    )


def available_prompts() -> List[str]:
    if not PROMPTS_DIR.exists():
        return []
    return sorted(path.stem for path in PROMPTS_DIR.glob("*.txt") if path.is_file())


def prompt_path(prompt_name: str) -> Path:
    clean_name = str(prompt_name or "").strip()
    if clean_name.endswith(".txt"):
        clean_name = clean_name[:-4]
    if not clean_name or Path(clean_name).name != clean_name:
        raise HarnessError("Prompt names must be simple filenames from the prompts folder.")
    return PROMPTS_DIR / f"{clean_name}.txt"


def load_prompt(prompt_name: str) -> Tuple[str, str, str]:
    path = prompt_path(prompt_name)
    if not path.exists():
        prompts = available_prompts()
        if prompts:
            prompt_list = ", ".join(prompts)
            raise HarnessError(f"Prompt '{prompt_name}' not found. Available prompts: {prompt_list}")
        raise HarnessError(f"Prompt '{prompt_name}' not found. No prompts are available in {PROMPTS_DIR}.")

    text = path.read_text(encoding="utf-8")
    prompt_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    return path.stem, text, prompt_hash


def load_results(path: Path = RESULTS_PATH) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        backup_path = backup_malformed_results(path)
        print(f"results.json was malformed. Backed it up to {backup_path.name} and started fresh.")
        return None
    if not isinstance(loaded, dict):
        backup_path = backup_malformed_results(path)
        print(f"results.json was malformed. Backed it up to {backup_path.name} and started fresh.")
        return None
    loaded.setdefault("runs", [])
    normalize_image_attribute_fields(loaded)
    return loaded


def backup_malformed_results(path: Path) -> Path:
    backup_path = path.with_name(f"{path.name}.bak")
    if backup_path.exists():
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_path = path.with_name(f"{path.name}.bak.{stamp}")
    shutil.move(str(path), str(backup_path))
    return backup_path


def fresh_results(batch_path: Optional[Path] = None) -> Dict[str, Any]:
    return {
        "batch_path": str(batch_path) if batch_path else "",
        "created_at": utc_now(),
        "codex_model": "",
        "codex_verdicts": {},
        "runs": [],
    }


def normalize_image_attribute_fields(results: Dict[str, Any]) -> None:
    for run in results.get("runs") or []:
        if not isinstance(run, dict):
            continue
        for image in run.get("images") or []:
            if not isinstance(image, dict):
                continue
            verdict = str(image.get("verdict") or "")
            for key in ("title", "build_stage", "layout"):
                value = image.get(key)
                if verdict == "SLIDE" and value:
                    image[key] = str(value).strip() or None
                else:
                    image[key] = None


def save_results(results: Dict[str, Any], path: Path = RESULTS_PATH) -> None:
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(path)


def ensure_same_batch(results: Dict[str, Any], batch_path: Path) -> None:
    existing_batch = str(results.get("batch_path") or "").strip()
    if not existing_batch:
        results["batch_path"] = str(batch_path)
        return

    if Path(existing_batch).expanduser().resolve() != batch_path:
        raise HarnessError(
            "results.json already belongs to a different image batch. "
            "Delete or move results.json before starting a new dataset."
        )


def next_run_id(runs: Sequence[Dict[str, Any]]) -> str:
    highest = 0
    for run in runs:
        match = re.match(r"run_(\d+)$", str(run.get("run_id") or ""))
        if match:
            highest = max(highest, int(match.group(1)))
    return f"run_{highest + 1:03d}"


def profile_names() -> List[str]:
    return sorted(PROFILES.keys())


def profile_names_for_task(task: str) -> List[str]:
    prefix = str(task or "").strip().upper()
    if not prefix:
        return profile_names()
    return sorted(name for name in PROFILES if name.startswith(f"{prefix}_"))


def normalize_profile_name(profile_name: Optional[str]) -> str:
    normalized = str(profile_name or DEFAULT_PROFILE).strip().upper()
    if normalized not in PROFILES:
        raise HarnessError(f"Unknown profile '{profile_name}'. Available profiles: {', '.join(profile_names())}")
    return normalized


def build_load_config(profile_name: str, model_id: str) -> Dict[str, Any]:
    profile = PROFILES[normalize_profile_name(profile_name)]
    return {
        "model": model_id,
        **profile["load"],
    }


def validate_applied_load_config(profile_name: str, applied_config: Any) -> None:
    if not isinstance(applied_config, dict):
        return
    profile_name = normalize_profile_name(profile_name)
    expected_context = PROFILES[profile_name]["load"].get("context_length")
    applied_context = applied_config.get("context_length")
    if applied_context is None:
        return
    if safe_int(applied_context) != safe_int(expected_context):
        raise HarnessError(
            "Load config mismatch: expected "
            f"context_length={expected_context}, got {applied_context}. "
            "Check if LM Studio JIT loading is overriding profile settings."
        )


def build_vision_messages(prompt_text: str, image_path: Path) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url(image_path),
                    },
                },
                {
                    "type": "text",
                    "text": prompt_text,
                },
            ],
        }
    ]


def build_inference_payload(
    profile_name: str,
    model_id: str,
    messages: Sequence[Dict[str, Any]],
    *,
    max_tokens_override: Optional[int] = None,
    stop: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    profile = PROFILES[normalize_profile_name(profile_name)]
    inference = dict(profile["inference"])
    if max_tokens_override is not None:
        inference["max_tokens"] = int(max_tokens_override)
    if stop is not None:
        inference["stop"] = list(stop)
    return {
        "model": model_id,
        "messages": list(messages),
        **inference,
    }


def chat_endpoint(base_url: str) -> str:
    normalized = str(base_url or DEFAULT_BASE_URL).strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def lm_studio_native_base_url(base_url: str) -> str:
    normalized = str(base_url or DEFAULT_BASE_URL).strip().rstrip("/")
    if normalized.endswith("/api/v1"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized[:-3]}/api/v1"
    return f"{normalized}/api/v1"


def lm_studio_openai_base_url(base_url: str) -> str:
    normalized = str(base_url or DEFAULT_BASE_URL).strip().rstrip("/")
    if normalized.endswith("/api/v1"):
        return f"{normalized[:-7]}/v1"
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def require_requests() -> Any:
    global REQUESTS_MODULE
    if REQUESTS_MODULE is not None:
        return REQUESTS_MODULE
    try:
        import requests as requests_module
    except ImportError:
        raise HarnessError("The 'requests' package is required. Install it, then run the harness again.")
    REQUESTS_MODULE = requests_module
    return REQUESTS_MODULE


def set_token_provider(provider: Optional[Callable[[], str]]) -> None:
    global TOKEN_PROVIDER
    TOKEN_PROVIDER = provider


def lm_studio_api_token() -> str:
    env_token = str(os.environ.get("LM_STUDIO_API_KEY") or os.environ.get("LM_API_TOKEN") or "").strip()
    if env_token:
        return env_token
    if TOKEN_PROVIDER:
        try:
            return str(TOKEN_PROVIDER() or "").strip()
        except Exception:
            return ""
    return ""


def request_headers(content_type: Optional[str] = None) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    token = lm_studio_api_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def fetch_lm_studio_models(base_url: str, timeout_seconds: int = 20) -> Dict[str, Any]:
    requests_module = require_requests()
    native_url = f"{lm_studio_native_base_url(base_url)}/models"
    openai_url = f"{lm_studio_openai_base_url(base_url)}/models"
    errors: List[str] = []
    native_payload: Dict[str, Any] = {}
    openai_payload: Dict[str, Any] = {}

    try:
        native_response = requests_module.get(native_url, headers=request_headers(), timeout=timeout_seconds)
        if native_response.status_code == 200:
            native_payload = native_response.json()
        else:
            errors.append(f"{native_url} returned HTTP {native_response.status_code}")
    except Exception as exc:
        errors.append(f"{native_url}: {exc}")

    try:
        openai_response = requests_module.get(openai_url, headers=request_headers(), timeout=timeout_seconds)
        if openai_response.status_code == 200:
            openai_payload = openai_response.json()
        else:
            errors.append(f"{openai_url} returned HTTP {openai_response.status_code}")
    except Exception as exc:
        errors.append(f"{openai_url}: {exc}")

    models = extract_model_infos(native_payload=native_payload, openai_payload=openai_payload)
    if not models and errors:
        raise HarnessError("Could not read LM Studio models. " + " | ".join(errors))
    return {
        "base_url": str(base_url or DEFAULT_BASE_URL).strip().rstrip("/") or DEFAULT_BASE_URL,
        "models": models,
        "errors": errors,
    }


def extract_model_infos(*, native_payload: Dict[str, Any], openai_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for item in native_model_items(native_payload):
        model_id = model_id_from_item(item)
        if not model_id:
            continue
        info = by_id.setdefault(model_id, {"id": model_id, "loaded": False, "loaded_instances": []})
        info.update(
            {
                "id": model_id,
                "display_name": str(item.get("display_name") or item.get("name") or model_id),
                "type": str(item.get("type") or ""),
                "source": "native",
            }
        )
        loaded_instances = loaded_instances_from_item(item)
        if loaded_instances:
            info["loaded"] = True
            info["loaded_instances"] = loaded_instances
        vision, reason = model_is_vision_capable(model_id, item)
        info["vision"] = vision
        info["vision_reason"] = reason

    for item in openai_model_items(openai_payload):
        model_id = model_id_from_item(item)
        if not model_id:
            continue
        info = by_id.setdefault(model_id, {"id": model_id, "loaded": False, "loaded_instances": []})
        info.setdefault("display_name", model_id)
        info.setdefault("type", str(item.get("type") or ""))
        if "vision" not in info:
            vision, reason = model_is_vision_capable(model_id, item)
            info["vision"] = vision
            info["vision_reason"] = reason
        if info.get("source") != "native":
            info["source"] = "openai"

    return sorted(by_id.values(), key=lambda model: str(model.get("id") or "").lower())


def native_model_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    models = payload.get("models") if isinstance(payload, dict) else []
    if not isinstance(models, list):
        return []
    return [item for item in models if isinstance(item, dict) and not model_is_embedding(item)]


def openai_model_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else []
    if not isinstance(data, list):
        data = payload.get("models") if isinstance(payload, dict) else []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict) and not model_is_embedding(item)]


def model_is_embedding(item: Dict[str, Any]) -> bool:
    return str(item.get("type") or "").strip().lower() == "embedding"


def model_id_from_item(item: Dict[str, Any]) -> str:
    return str(item.get("id") or item.get("key") or item.get("name") or "").strip()


def loaded_instances_from_item(item: Dict[str, Any]) -> List[str]:
    instances = item.get("loaded_instances") or []
    loaded: List[str] = []
    for instance in instances if isinstance(instances, list) else []:
        if isinstance(instance, dict):
            instance_id = str(instance.get("id") or "").strip()
        else:
            instance_id = str(instance or "").strip()
        if instance_id:
            loaded.append(instance_id)
    return loaded


def loaded_model_rows(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for item in native_model_items(payload):
        model_id = model_id_from_item(item)
        for instance_id in loaded_instances_from_item(item):
            rows.append({"model": model_id, "instance_id": instance_id})
    return rows


def model_is_vision_capable(model_id: str, item: Dict[str, Any]) -> Tuple[bool, str]:
    text = json.dumps(item, ensure_ascii=False).lower()
    model_lower = str(model_id or "").lower()
    metadata_markers = (
        '"vision"',
        '"image"',
        "image_url",
        "multimodal",
        "multi-modal",
        "visual",
        "vlm",
    )
    if any(marker in text for marker in metadata_markers):
        return True, "LM Studio metadata mentions image or vision support"

    name_patterns = (
        r"(^|[-_./])vl($|[-_./\d])",
        r"vision",
        r"qwen\d*(?:\.\d+)?-vl",
        r"qwen-vl",
        r"llava",
        r"pixtral",
        r"internvl",
        r"mllama",
        r"paligemma",
        r"minicpm[-_]?v",
        r"moondream",
        r"idefics",
        r"outlier",
        r"gemma-3",
        r"gemma-4",
    )
    if any(re.search(pattern, model_lower) for pattern in name_patterns):
        return True, "model name matches a known vision-model pattern"
    return False, "no vision capability marker found"


def unload_lm_studio_instance(
    *,
    base_url: str,
    instance_id: str,
    timeout_seconds: int,
) -> None:
    requests_module = require_requests()
    response = requests_module.post(
        f"{lm_studio_native_base_url(base_url)}/models/unload",
        json={"instance_id": instance_id},
        headers=request_headers(),
        timeout=timeout_seconds,
    )
    if response.status_code != 200:
        raise HarnessError(f"LM Studio could not unload {instance_id}: HTTP {response.status_code}: {response.text[:300]}")


def load_lm_studio_model(
    *,
    base_url: str,
    model: str,
    timeout_seconds: int,
    profile_name: str = DEFAULT_PROFILE,
) -> Dict[str, Any]:
    requests_module = require_requests()
    load_config = build_load_config(profile_name, model)
    response = requests_module.post(
        f"{lm_studio_native_base_url(base_url)}/models/load",
        json=load_config,
        headers=request_headers(),
        timeout=timeout_seconds,
    )
    if response.status_code != 200:
        raise HarnessError(f"LM Studio could not load {model}: HTTP {response.status_code}: {response.text[:300]}")
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    status = str(payload.get("status") or "").strip().lower()
    if status and status != "loaded":
        raise HarnessError(f"LM Studio did not load {model}: status={status}")
    applied_config = payload.get("load_config") or payload.get("config") or {}
    validate_applied_load_config(profile_name, applied_config)
    return {
        "model": model,
        "profile": normalize_profile_name(profile_name),
        "instance_id": str(payload.get("instance_id") or model).strip() or model,
        "status": status or "load_requested",
        "requested_load_config": load_config,
        "load_config_applied": applied_config,
        "load_time_seconds": payload.get("load_time_seconds"),
    }


def ensure_lm_studio_model_loaded(
    *,
    base_url: str,
    model: str,
    timeout_seconds: int,
    profile_name: str = DEFAULT_PROFILE,
    force_reload: bool = False,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    def log(message: str) -> None:
        if logger:
            logger(message)

    profile_name = normalize_profile_name(profile_name)
    requests_module = require_requests()
    native_models_url = f"{lm_studio_native_base_url(base_url)}/models"
    response = requests_module.get(native_models_url, headers=request_headers(), timeout=timeout_seconds)
    if response.status_code != 200:
        raise HarnessError(f"LM Studio model-state check failed: HTTP {response.status_code}: {response.text[:300]}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise HarnessError("LM Studio returned non-JSON model state.") from exc

    available = sorted({model_id_from_item(item) for item in native_model_items(payload) if model_id_from_item(item)})
    if model not in available:
        available_text = ", ".join(available) if available else "none"
        raise HarnessError(f"Model {model} was not found in LM Studio. Available models: {available_text}")

    loaded_rows = loaded_model_rows(payload)
    matching_rows = [row for row in loaded_rows if row["model"] == model or row["instance_id"] == model]
    if len(loaded_rows) == 1 and matching_rows and not force_reload:
        log(f"{model} is already loaded.")
        return {
            "model": model,
            "profile": profile_name,
            "instance_id": matching_rows[0]["instance_id"],
            "status": "loaded",
            "requested_load_config": build_load_config(profile_name, model),
            "load_config_applied": {},
            "load_time_seconds": None,
        }

    for row in loaded_rows:
        log(f"Offloading {row['model']}...")
        unload_lm_studio_instance(
            base_url=base_url,
            instance_id=row["instance_id"],
            timeout_seconds=timeout_seconds,
        )
    if loaded_rows:
        time.sleep(2)

    log(f"Loading {model} with {profile_name}...")
    load_info = load_lm_studio_model(
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
        profile_name=profile_name,
    )
    if load_info.get("load_config_applied"):
        log(f"Applied load config: {json.dumps(load_info['load_config_applied'], sort_keys=True)}")
    if load_info.get("load_time_seconds") is not None:
        log(f"Load time: {float(load_info['load_time_seconds']):.1f}s")
    instance_id = str(load_info.get("instance_id") or model)
    deadline = time.monotonic() + max(10, timeout_seconds)
    while time.monotonic() < deadline:
        response = requests_module.get(native_models_url, headers=request_headers(), timeout=timeout_seconds)
        if response.status_code == 200:
            try:
                rows = loaded_model_rows(response.json())
            except ValueError:
                rows = []
            for row in rows:
                if row["model"] == model or row["instance_id"] in {model, instance_id}:
                    log(f"{model} is loaded.")
                    load_info["instance_id"] = row["instance_id"]
                    load_info["status"] = "loaded"
                    return load_info
        time.sleep(1)
    raise HarnessError(f"Timed out waiting for LM Studio to load {model}.")


def call_lm_studio(
    *,
    model: str,
    prompt_text: str,
    image_path: Path,
    base_url: str,
    timeout_seconds: int,
    profile_name: str = DEFAULT_PROFILE,
) -> Dict[str, Any]:
    requests_module = require_requests()
    profile_name = normalize_profile_name(profile_name)
    payload = build_inference_payload(profile_name, model, build_vision_messages(prompt_text, image_path))

    endpoint = chat_endpoint(base_url)
    total_start = time.perf_counter()
    for attempt in (1, 2):
        request_start = time.perf_counter()
        try:
            response = requests_module.post(
                endpoint,
                json=payload,
                headers=request_headers(),
                timeout=timeout_seconds,
            )
        except requests_module.exceptions.Timeout as exc:
            if attempt == 1:
                time.sleep(5)
                continue
            return error_image_result(
                f"Timed out after retry: {exc}",
                response_time_sec=time.perf_counter() - total_start,
            )
        except requests_module.exceptions.RequestException as exc:
            return error_image_result(
                f"Could not reach LM Studio at {endpoint}: {exc}",
                response_time_sec=time.perf_counter() - request_start,
            )

        response_time_sec = time.perf_counter() - request_start
        if response.status_code != 200:
            return error_image_result(
                f"LM Studio returned HTTP {response.status_code}: {response.text[:500]}",
                response_time_sec=response_time_sec,
                raw_response=response.text,
            )

        try:
            payload_json = response.json()
        except ValueError:
            return error_image_result(
                "LM Studio returned a non-JSON response.",
                response_time_sec=response_time_sec,
                raw_response=response.text,
            )
        return successful_image_result(payload_json, response_time_sec)

    return error_image_result("Timed out before receiving a response.", response_time_sec=0.0)


def successful_image_result(payload: Dict[str, Any], response_time_sec: float) -> Dict[str, Any]:
    raw_response = extract_message_content(payload)
    parsed = parse_response(raw_response)
    usage = payload.get("usage") if isinstance(payload, dict) else {}
    if not isinstance(usage, dict):
        usage = {}

    completion_tokens = safe_int(usage.get("completion_tokens"))
    prompt_tokens = safe_int(usage.get("prompt_tokens"))
    tokens_per_sec = completion_tokens / response_time_sec if response_time_sec > 0 else 0.0

    return {
        "description": parsed["description"],
        "evidence": parsed["evidence"],
        "verdict": parsed["verdict"],
        "title": parsed["title"],
        "build_stage": parsed["build_stage"],
        "layout": parsed["layout"],
        "response_time_sec": round(response_time_sec, 3),
        "tokens_per_sec": round(tokens_per_sec, 3),
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "raw_response": raw_response,
    }


def error_image_result(
    error: str,
    *,
    response_time_sec: float,
    raw_response: str = "",
) -> Dict[str, Any]:
    return {
        "description": "",
        "evidence": f"ERROR: {error}",
        "verdict": "ERROR",
        "title": None,
        "build_stage": None,
        "layout": None,
        "response_time_sec": round(response_time_sec, 3),
        "tokens_per_sec": 0.0,
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "raw_response": raw_response,
        "error": error,
    }


def is_model_level_failure(result: Dict[str, Any]) -> bool:
    if str(result.get("verdict") or "") != "ERROR":
        return False
    text = " ".join(
        str(result.get(key) or "")
        for key in ("error", "raw_response", "evidence")
    ).lower()
    return any(marker in text for marker in MODEL_LEVEL_ERROR_MARKERS)


def extract_message_content(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return ""


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def run_codex_baseline(
    *,
    results: Dict[str, Any],
    images: Sequence[Path],
    codex_model: Optional[str],
    base_url: str,
    timeout_seconds: int,
    profile_name: str = DEFAULT_PROFILE,
    results_path: Path = RESULTS_PATH,
) -> None:
    existing = results.get("codex_verdicts")
    if isinstance(existing, dict) and existing:
        if codex_model:
            print("Existing Codex baseline found. Reusing it without overwriting.")
        return

    if not codex_model:
        print(
            "No Codex baseline found. Specify --codex-model to generate one, "
            "or continue without baseline (agreement comparisons will be skipped)."
        )
        results["codex_verdicts"] = {}
        return

    _, prompt_text, _ = load_prompt("classify_v1")
    print(f"Generating Codex baseline with {codex_model}...")
    ensure_lm_studio_model_loaded(
        base_url=base_url,
        model=codex_model,
        timeout_seconds=timeout_seconds,
        profile_name=profile_name,
        force_reload=True,
        logger=print,
    )
    verdicts: Dict[str, str] = {}
    for index, image_path in enumerate(images, start=1):
        result = call_lm_studio(
            model=codex_model,
            prompt_text=prompt_text,
            image_path=image_path,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            profile_name=profile_name,
        )
        verdict = str(result.get("verdict") or "UNCERTAIN")
        verdicts[image_path.name] = verdict
        print(progress_line(index, len(images), image_path.name, verdict, result))

    results["codex_model"] = codex_model
    results["codex_verdicts"] = verdicts
    save_results(results, results_path)
    print("Codex baseline saved.")


def build_run(
    *,
    results: Dict[str, Any],
    model: str,
    prompt_name: str,
    prompt_text: str,
    prompt_hash: str,
    images: Sequence[Path],
    base_url: str,
    timeout_seconds: int,
    profile_name: str = DEFAULT_PROFILE,
    load_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    profile_name = normalize_profile_name(profile_name)
    run_id = next_run_id(results.get("runs") or [])
    started = utc_now()
    image_results: List[Dict[str, Any]] = []
    run_start = time.perf_counter()

    for index, image_path in enumerate(images, start=1):
        result = call_lm_studio(
            model=model,
            prompt_text=prompt_text,
            image_path=image_path,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            profile_name=profile_name,
        )
        result["filename"] = image_path.name
        attach_codex_agreement(result, results.get("codex_verdicts"))
        image_results.append(result)
        print(progress_line(index, len(images), image_path.name, str(result["verdict"]), result))

    total_time_sec = time.perf_counter() - run_start
    return {
        "run_id": run_id,
        "model": model,
        "profile": profile_name,
        "prompt": prompt_name,
        "prompt_hash": prompt_hash,
        "inference_config": dict(PROFILES[profile_name]["inference"]),
        "requested_load_config": build_load_config(profile_name, model),
        "load_config_applied": dict((load_info or {}).get("load_config_applied") or {}),
        "load_time_seconds": (load_info or {}).get("load_time_seconds"),
        "run_timestamp": started,
        "batch_stats": calculate_batch_stats(image_results, total_time_sec),
        "images": image_results,
    }


def attach_codex_agreement(image_result: Dict[str, Any], codex_verdicts: Any) -> None:
    filename = str(image_result.get("filename") or "")
    codex_verdict = ""
    if isinstance(codex_verdicts, dict):
        codex_verdict = str(codex_verdicts.get(filename) or "")
    image_result["codex_verdict"] = codex_verdict

    verdict = str(image_result.get("verdict") or "")
    if verdict in KNOWN_VERDICTS and codex_verdict in KNOWN_VERDICTS:
        image_result["agreement"] = verdict == codex_verdict
    elif codex_verdict in KNOWN_VERDICTS:
        image_result["agreement"] = False
    else:
        image_result["agreement"] = None


def calculate_batch_stats(image_results: Sequence[Dict[str, Any]], total_time_sec: float) -> Dict[str, Any]:
    token_speeds = [
        float(item.get("tokens_per_sec") or 0.0)
        for item in image_results
        if float(item.get("tokens_per_sec") or 0.0) > 0
    ]
    response_times = [float(item.get("response_time_sec") or 0.0) for item in image_results]
    agreements = [item for item in image_results if item.get("agreement") is True]
    disagreements = [item for item in image_results if item.get("agreement") is False]
    comparable = len(agreements) + len(disagreements)

    return {
        "total_images": len(image_results),
        "total_time_sec": round(total_time_sec, 3),
        "avg_tokens_per_sec": round(sum(token_speeds) / len(token_speeds), 3) if token_speeds else 0.0,
        "min_tokens_per_sec": round(min(token_speeds), 3) if token_speeds else 0.0,
        "max_tokens_per_sec": round(max(token_speeds), 3) if token_speeds else 0.0,
        "avg_response_time_sec": round(sum(response_times) / len(response_times), 3)
        if response_times
        else 0.0,
        "slide_count": sum(1 for item in image_results if item.get("verdict") == "SLIDE"),
        "not_slide_count": sum(1 for item in image_results if item.get("verdict") == "NOT SLIDE"),
        "error_count": sum(1 for item in image_results if item.get("verdict") == "ERROR"),
        "uncertain_count": sum(1 for item in image_results if item.get("verdict") == "UNCERTAIN"),
        "codex_agreement_count": len(agreements),
        "codex_disagreement_count": len(disagreements),
        "codex_agreement_rate": round(len(agreements) / comparable, 3) if comparable else None,
    }


def progress_line(
    index: int,
    total: int,
    filename: str,
    verdict: str,
    result: Dict[str, Any],
) -> str:
    return (
        f"[{index:02d}/{total:02d}] {filename} -> {verdict:<9} "
        f"({float(result.get('response_time_sec') or 0.0):.1f}s | "
        f"{float(result.get('tokens_per_sec') or 0.0):.1f} tok/s)"
    )


def print_run_summary(run: Dict[str, Any]) -> None:
    stats = run.get("batch_stats") or {}
    total_images = int(stats.get("total_images") or 0)
    total_time = float(stats.get("total_time_sec") or 0.0)
    agreement_count = int(stats.get("codex_agreement_count") or 0)
    disagreement_count = int(stats.get("codex_disagreement_count") or 0)
    comparable = agreement_count + disagreement_count
    rate = stats.get("codex_agreement_rate")

    print("-" * 49)
    print(f"Run complete: {total_images} images in {total_time:.1f}s")
    print(
        f"Avg tok/s: {float(stats.get('avg_tokens_per_sec') or 0.0):.1f} | "
        f"Min: {float(stats.get('min_tokens_per_sec') or 0.0):.1f} | "
        f"Max: {float(stats.get('max_tokens_per_sec') or 0.0):.1f}"
    )
    print(
        f"SLIDEs: {int(stats.get('slide_count') or 0)} | "
        f"NOT SLIDEs: {int(stats.get('not_slide_count') or 0)} | "
        f"Errors: {int(stats.get('error_count') or 0)}"
    )
    if rate is None:
        print("Codex agreement: skipped")
    else:
        print(f"Codex agreement: {rate * 100:.1f}% ({agreement_count}/{comparable})")
    print("-" * 49)
    print(f"Report written: {REPORT_PATH.name}")
    print(f"Results saved: {RESULTS_PATH.name}")


def generate_report(results: Dict[str, Any], report_path: Path = REPORT_PATH) -> str:
    runs = [run for run in results.get("runs") or [] if isinstance(run, dict)]
    lines: List[str] = []
    lines.extend(report_header(results, runs))
    lines.extend(performance_matrix(runs))
    lines.extend(speed_accuracy_table(runs))
    lines.extend(per_image_results(results, runs))
    lines.extend(disagreement_detail(results, runs))
    lines.extend(per_run_detail(runs))
    report = "\n".join(lines).rstrip() + "\n"
    report_path.write_text(report, encoding="utf-8")
    return report


def report_header(results: Dict[str, Any], runs: Sequence[Dict[str, Any]]) -> List[str]:
    total_images = 0
    if runs:
        stats = runs[-1].get("batch_stats") or {}
        total_images = int(stats.get("total_images") or 0)
    if not total_images:
        codex = results.get("codex_verdicts")
        total_images = len(codex) if isinstance(codex, dict) else 0

    return [
        "# Vision Model Test Report",
        "",
        f"- Batch path: {results.get('batch_path') or 'n/a'}",
        f"- Total images: {total_images}",
        f"- Total runs recorded: {len(runs)}",
        f"- Last updated: {format_timestamp(utc_now())}",
        "",
    ]


def performance_matrix(runs: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "## Performance Matrix (ranked by Codex agreement)",
        "",
        "| Rank | Model | Prompt | Profile | Run Date | Avg tok/s | SLIDEs | NOT SLIDEs | Codex Agreement |",
        "|------|-------|--------|---------|----------|-----------|--------|------------|-----------------|",
    ]
    for rank, run in enumerate(sorted_runs_by_agreement(runs), start=1):
        stats = run.get("batch_stats") or {}
        lines.append(
            "| {rank} | {model} | {prompt} | {profile} | {date} | {avg:.1f} | {slides} | {not_slides} | {agreement} |".format(
                rank=rank,
                model=escape_md(str(run.get("model") or "")),
                prompt=escape_md(str(run.get("prompt") or "")),
                profile=escape_md(str(run.get("profile") or DEFAULT_PROFILE)),
                date=format_timestamp(str(run.get("run_timestamp") or "")),
                avg=float(stats.get("avg_tokens_per_sec") or 0.0),
                slides=int(stats.get("slide_count") or 0),
                not_slides=int(stats.get("not_slide_count") or 0),
                agreement=format_percent(stats.get("codex_agreement_rate")),
            )
        )
    if not runs:
        lines.append("| - | - | - | - | - | - | - | - | - |")
    return lines + [""]


def speed_accuracy_table(runs: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "## Speed vs Accuracy",
        "",
        "| Model | Prompt | Profile | Avg tok/s | Codex Agreement | Quality/Speed Score |",
        "|-------|--------|---------|-----------|-----------------|---------------------|",
    ]

    scored_runs = sorted(
        runs,
        key=lambda run: quality_speed_score(run),
        reverse=True,
    )
    for run in scored_runs:
        stats = run.get("batch_stats") or {}
        score = quality_speed_score(run)
        score_text = f"{score:.2f}" if score >= 0 else "n/a"
        lines.append(
            "| {model} | {prompt} | {profile} | {avg:.1f} | {agreement} | {score} |".format(
                model=escape_md(str(run.get("model") or "")),
                prompt=escape_md(str(run.get("prompt") or "")),
                profile=escape_md(str(run.get("profile") or DEFAULT_PROFILE)),
                avg=float(stats.get("avg_tokens_per_sec") or 0.0),
                agreement=format_percent(stats.get("codex_agreement_rate")),
                score=score_text,
            )
        )
    if not runs:
        lines.append("| - | - | - | - | - | - |")
    lines.extend(
        [
            "",
            "*Quality/Speed Score = agreement_rate / avg_response_time_sec - higher is better*",
            "",
        ]
    )
    return lines


def per_image_results(results: Dict[str, Any], runs: Sequence[Dict[str, Any]]) -> List[str]:
    labels = unique_run_labels(runs)
    lines = [
        "## Per-Image Results",
        "",
        "| Image | Codex | " + " | ".join(escape_md(label) for label in labels) + " |"
        if labels
        else "| Image | Codex |",
        "|-------|-------|" + "---|" * len(labels) if labels else "|-------|-------|",
    ]

    codex = known_codex_verdicts(results)
    image_names = sorted(set(codex.keys()) | set(iter_run_filenames(runs)))
    run_indexes = [images_by_filename(run) for run in runs]

    for image_name in image_names:
        row = [escape_md(image_name), escape_md(str(codex.get(image_name) or ""))]
        for run_index in run_indexes:
            image = run_index.get(image_name)
            if not image:
                row.append("")
                continue
            row.append(escape_md(verdict_with_marker(image)))
        lines.append("| " + " | ".join(row) + " |")

    if not image_names:
        lines.append("| - | - |" + " - |" * len(labels))
    return lines + [""]


def disagreement_detail(results: Dict[str, Any], runs: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "## Disagreement Detail",
        "",
    ]
    codex = known_codex_verdicts(results)
    if not codex:
        lines.extend(["Codex baseline is not available, so disagreement detail was skipped.", ""])
        return lines

    labels = unique_run_labels(runs)
    found = False
    image_names = sorted(set(codex.keys()) | set(iter_run_filenames(runs)))
    for image_name in image_names:
        disagreements: List[Tuple[str, Dict[str, Any]]] = []
        for label, run in zip(labels, runs):
            image = images_by_filename(run).get(image_name)
            if image and image.get("agreement") is False:
                disagreements.append((label, image))
        if not disagreements:
            continue

        found = True
        lines.extend([f"### {image_name}", "", f"- Codex: {codex.get(image_name)}"])
        for label, image in disagreements:
            lines.extend(
                [
                    f"- {label}: {image.get('verdict')}",
                    f"  - Description: {one_line(str(image.get('description') or ''))}",
                    f"  - Evidence: {one_line(str(image.get('evidence') or ''))}",
                ]
            )
        lines.append("")

    if not found:
        lines.extend(["No disagreements recorded.", ""])
    return lines


def per_run_detail(runs: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "## Per-Run Detail",
        "",
    ]
    if not runs:
        lines.extend(["No runs recorded.", ""])
        return lines

    for run in runs:
        stats = run.get("batch_stats") or {}
        lines.extend(
            [
                "### {run_id} - {model} / {prompt} / {profile}".format(
                    run_id=escape_md(str(run.get("run_id") or "")),
                    model=escape_md(str(run.get("model") or "")),
                    prompt=escape_md(str(run.get("prompt") or "")),
                    profile=escape_md(str(run.get("profile") or DEFAULT_PROFILE)),
                ),
                "",
                "Run date: {date} | Prompt hash: `{hash}` | Profile: `{profile}` | Agreement: {agreement}".format(
                    date=format_timestamp(str(run.get("run_timestamp") or "")),
                    hash=str(run.get("prompt_hash") or ""),
                    profile=str(run.get("profile") or DEFAULT_PROFILE),
                    agreement=format_percent(stats.get("codex_agreement_rate")),
                ),
                "",
                "| Image | Verdict | Title | Build Stage | Layout | Codex | Agree | Time (s) | tok/s |",
                "|-------|---------|-------|-------------|--------|-------|-------|----------|-------|",
            ]
        )
        for image in run.get("images") or []:
            if not isinstance(image, dict):
                continue
            lines.append(
                "| {image} | {verdict} | {title} | {build_stage} | {layout} | {codex} | {agree} | {time_sec:.1f} | {tok_sec:.1f} |".format(
                    image=escape_md(str(image.get("filename") or "")),
                    verdict=escape_md(str(image.get("verdict") or "")),
                    title=escape_md(per_run_attribute_text(image, "title")),
                    build_stage=escape_md(per_run_attribute_text(image, "build_stage")),
                    layout=escape_md(per_run_attribute_text(image, "layout")),
                    codex=escape_md(str(image.get("codex_verdict") or "")),
                    agree=agreement_text(image.get("agreement")),
                    time_sec=float(image.get("response_time_sec") or 0.0),
                    tok_sec=float(image.get("tokens_per_sec") or 0.0),
                )
            )
        lines.append("")
    return lines


def sorted_runs_by_agreement(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        runs,
        key=lambda run: (
            agreement_sort_value(run),
            quality_speed_score(run),
            str(run.get("run_timestamp") or ""),
        ),
        reverse=True,
    )


def agreement_sort_value(run: Dict[str, Any]) -> float:
    stats = run.get("batch_stats") or {}
    rate = stats.get("codex_agreement_rate")
    return float(rate) if isinstance(rate, (int, float)) else -1.0


def quality_speed_score(run: Dict[str, Any]) -> float:
    stats = run.get("batch_stats") or {}
    rate = stats.get("codex_agreement_rate")
    avg_response = float(stats.get("avg_response_time_sec") or 0.0)
    if not isinstance(rate, (int, float)) or avg_response <= 0:
        return -1.0
    return float(rate) / avg_response


def unique_run_labels(runs: Sequence[Dict[str, Any]]) -> List[str]:
    base_labels = [f"{run.get('model')}/{run.get('prompt')}/{run.get('profile') or DEFAULT_PROFILE}" for run in runs]
    counts: Dict[str, int] = {}
    for label in base_labels:
        counts[label] = counts.get(label, 0) + 1

    labels = []
    for label, run in zip(base_labels, runs):
        if counts[label] > 1:
            labels.append(f"{label} ({run.get('run_id')})")
        else:
            labels.append(label)
    return labels


def iter_run_filenames(runs: Sequence[Dict[str, Any]]) -> Iterable[str]:
    for run in runs:
        for image in run.get("images") or []:
            if isinstance(image, dict) and image.get("filename"):
                yield str(image["filename"])


def images_by_filename(run: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for image in run.get("images") or []:
        if isinstance(image, dict) and image.get("filename"):
            index[str(image["filename"])] = image
    return index


def known_codex_verdicts(results: Dict[str, Any]) -> Dict[str, str]:
    codex = results.get("codex_verdicts")
    if not isinstance(codex, dict):
        return {}
    return {
        str(filename): str(verdict)
        for filename, verdict in codex.items()
        if str(verdict) in KNOWN_VERDICTS
    }


def verdict_with_marker(image: Dict[str, Any]) -> str:
    verdict = str(image.get("verdict") or "")
    agreement = image.get("agreement")
    if agreement is True:
        return f"{verdict} ✅"
    if agreement is False:
        return f"{verdict} ❌"
    return verdict


def agreement_text(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return ""


def per_run_attribute_text(image: Dict[str, Any], key: str) -> str:
    if image.get("verdict") != "SLIDE":
        return "-"
    value = image.get(key)
    return str(value).strip() if value else "-"


def format_percent(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) * 100:.1f}%"
    return "n/a"


def format_timestamp(value: str) -> str:
    if not value:
        return "n/a"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M")


def escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def one_line(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip() or "n/a"


def print_runs(results: Dict[str, Any]) -> None:
    runs = [run for run in results.get("runs") or [] if isinstance(run, dict)]
    if not runs:
        print("No runs recorded yet.")
        return

    rows = [
        [
            str(run.get("run_id") or ""),
            format_timestamp(str(run.get("run_timestamp") or "")),
            str(run.get("model") or ""),
            str(run.get("prompt") or ""),
            str(run.get("profile") or DEFAULT_PROFILE),
            str((run.get("batch_stats") or {}).get("total_images") or 0),
            f"{float((run.get('batch_stats') or {}).get('avg_tokens_per_sec') or 0.0):.1f}",
            format_percent((run.get("batch_stats") or {}).get("codex_agreement_rate")),
        ]
        for run in runs
    ]
    headers = ["Run", "Date", "Model", "Prompt", "Profile", "Images", "Avg tok/s", "Agreement"]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    print("  ".join(headers[index].ljust(widths[index]) for index in range(len(headers))))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(row[index].ljust(widths[index]) for index in range(len(row))))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="LM Studio model name for the main run.")
    parser.add_argument("--prompt", help="Prompt filename from prompts/ without the .txt extension.")
    parser.add_argument("--batch", type=Path, help="Folder containing .jpg, .jpeg, or .png images.")
    parser.add_argument("--codex-model", help="Model used once to generate the reusable Codex baseline.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"LM Studio base URL. Default: {DEFAULT_BASE_URL}")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="Request timeout in seconds.")
    parser.add_argument(
        "--profile",
        choices=profile_names(),
        default=os.environ.get("VISION_PROFILE", DEFAULT_PROFILE),
        help="Inference profile. Env var: VISION_PROFILE",
    )
    parser.add_argument("--list-runs", action="store_true", help="List all recorded runs without API calls.")
    parser.add_argument("--report-only", action="store_true", help="Regenerate report.md without API calls.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.list_runs:
            print_runs(load_results() or fresh_results())
            return 0

        if args.report_only:
            results = load_results()
            if not results:
                raise HarnessError("No results.json found. Run a model first, then regenerate the report.")
            generate_report(results)
            print(f"Report written: {REPORT_PATH.name}")
            return 0

        missing = [name for name in ("model", "prompt", "batch") if not getattr(args, name)]
        if missing:
            raise HarnessError("Missing required arguments for a run: " + ", ".join(f"--{name}" for name in missing))

        require_requests()
        batch_path = args.batch.expanduser().resolve()
        if not batch_path.exists() or not batch_path.is_dir():
            raise HarnessError(f"Batch folder does not exist: {batch_path}")

        images = discover_images(batch_path)
        if not images:
            raise HarnessError(f"No supported images found in {batch_path}. Supported formats: .jpg, .jpeg, .png")

        prompt_name, prompt_text, prompt_hash = load_prompt(args.prompt)
        profile_name = normalize_profile_name(args.profile)
        results = load_results() or fresh_results(batch_path)
        ensure_same_batch(results, batch_path)

        run_codex_baseline(
            results=results,
            images=images,
            codex_model=args.codex_model,
            base_url=args.base_url,
            timeout_seconds=args.timeout,
            profile_name=DEFAULT_PROFILE,
        )

        load_info = ensure_lm_studio_model_loaded(
            base_url=args.base_url,
            model=args.model,
            timeout_seconds=args.timeout,
            profile_name=profile_name,
            force_reload=True,
            logger=print,
        )
        run = build_run(
            results=results,
            model=args.model,
            prompt_name=prompt_name,
            prompt_text=prompt_text,
            prompt_hash=prompt_hash,
            images=images,
            base_url=args.base_url,
            timeout_seconds=args.timeout,
            profile_name=profile_name,
            load_info=load_info,
        )
        results.setdefault("runs", []).append(run)
        save_results(results)
        generate_report(results)
        print_run_summary(run)
        return 0
    except HarnessError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
