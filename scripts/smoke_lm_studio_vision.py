#!/usr/bin/env python3
"""Send one lecture slide through the same LM Studio vision prompt used by the app."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lecture_processor.ai.providers.base import AnalyzeLectureRequest  # noqa: E402
from lecture_processor.ai.providers.mlx_openai import (  # noqa: E402
    DEFAULT_TIMEOUT_SECONDS,
    MLXVisionProvider,
    _slide_batch_prompt,
    _vision_input_items,
)


def _keychain_lm_studio_token() -> str:
    if os.environ.get("LM_STUDIO_API_KEY") or os.environ.get("LM_API_TOKEN"):
        return ""
    if sys.platform != "darwin":
        return ""
    try:
        token = subprocess.check_output(
            [
                "security",
                "find-generic-password",
                "-a",
                "lm_studio_api_token",
                "-s",
                "Lecture Processor",
                "-w",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""
    return token


def _load_artifact(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _batch_config_for(lecture_json: Path) -> dict:
    batch_path = lecture_json.parent.parent / "batch.json"
    if not batch_path.exists():
        return {}
    try:
        return _load_artifact(batch_path).get("config") or {}
    except Exception:
        return {}


def _default_base_url(lecture_json: Path) -> str:
    return str(_batch_config_for(lecture_json).get("mlx_vision_base_url") or "http://localhost:1234/v1")


def _default_model(lecture_json: Path) -> str:
    routing = _batch_config_for(lecture_json).get("ai_model_routing") or {}
    slides = routing.get("slides") or {}
    return str(slides.get("model") or "default")


def _slide_by_id(artifact: dict, slide_id: int) -> dict:
    for slide in artifact.get("slides") or []:
        if int(slide.get("id") or 0) == slide_id:
            return slide
    raise SystemExit(f"Slide id {slide_id} was not found in the lecture artifact.")


def _request_for_slide(lecture_json: Path, artifact: dict, slide: dict) -> AnalyzeLectureRequest:
    transcript = artifact.get("transcript") or {}
    duration_seconds = float((artifact.get("media") or {}).get("duration_seconds") or 0.0)
    return AnalyzeLectureRequest(
        lecture_id=str(artifact.get("lecture_id") or lecture_json.parent.name),
        transcript_text=str(transcript.get("text") or ""),
        segments=list(transcript.get("segments") or []),
        slides=[slide],
        duration_minutes=duration_seconds / 60 if duration_seconds else 0.0,
        lecture_dir=lecture_json.parent,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lecture_json", type=Path, help="Path to a processed lecture.json file.")
    parser.add_argument("--slide-id", type=int, default=3, help="Slide id to send. Defaults to 3.")
    parser.add_argument("--base-url", default=None, help="LM Studio base URL. Defaults to batch config.")
    parser.add_argument("--model", default=None, help="LM Studio model id. Defaults to batch slide model.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--show-prompt", action="store_true", help="Include the exact prompt in stdout.")
    args = parser.parse_args()

    lecture_json = args.lecture_json.expanduser().resolve()
    artifact = _load_artifact(lecture_json)
    slide = _slide_by_id(artifact, args.slide_id)
    request = _request_for_slide(lecture_json, artifact, slide)
    prompt = _slide_batch_prompt(request, [slide])
    base_url = args.base_url or _default_base_url(lecture_json)
    model = args.model or _default_model(lecture_json)

    token = _keychain_lm_studio_token()
    if token:
        os.environ["LM_STUDIO_API_KEY"] = token

    output = {
        "ok": False,
        "base_url": base_url,
        "model": model,
        "lecture_json": str(lecture_json),
        "slide_id": args.slide_id,
        "slide_image": str(lecture_json.parent / str(slide.get("relative_path") or "")),
    }
    if args.show_prompt:
        output["prompt"] = prompt

    provider = MLXVisionProvider(base_url=base_url, model=model, timeout_seconds=args.timeout)
    try:
        payload, usage = provider._client.chat_json_with_lm_studio_images(
            _vision_input_items(request, [slide], prompt),
            system="You analyze lecture slide images and return compact JSON.",
            max_tokens=4096,
            temperature=0.3,
            context="LM Studio vision smoke test",
        )
    except Exception as exc:
        output["error_type"] = type(exc).__name__
        output["error"] = str(exc)
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 2

    output["ok"] = True
    output["usage"] = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }
    output["response"] = payload
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
