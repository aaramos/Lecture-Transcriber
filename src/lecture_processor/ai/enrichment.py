import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from lecture_processor.artifacts import load_json, save_json, utc_now_iso
from lecture_processor.config import AIProviderName, BatchConfig

from .model_routing import (
    RouteAvailability,
    probe_lm_studio_model_availability,
    routed_analyze_lecture,
    routed_analyze_lectures_staged,
    uses_experimental_routing,
)
from .providers.registry import build_provider, provider_info
from .providers.base import AnalyzeLectureRequest, AnalyzeLectureResponse

ENRICHMENT_PARTIAL_NAME = ".enrichment_partial.json"


def enrich_lecture_artifact(
    lecture_json_path: Path,
    config: BatchConfig,
    *,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    if config.ai_provider is AIProviderName.NONE:
        return load_json(lecture_json_path)

    artifact = load_json(lecture_json_path)
    _emit(progress_callback, "enrichment_started", source=artifact["source"]["filename"])
    started = time.monotonic()
    started_at = utc_now_iso()
    routing_enabled = uses_experimental_routing(config)
    availability = _probe_availability(config) if routing_enabled else RouteAvailability()
    if availability.warnings:
        _append_processing_warnings(lecture_json_path, availability.warnings)
        artifact = load_json(lecture_json_path)
    info = provider_info(_provider_info_name(config))
    provider = _build_provider_for_config(config, info)
    request = _request_from_artifact(artifact, lecture_json_path.parent)
    if routing_enabled:
        response = routed_analyze_lecture(
            request,
            config,
            gemini_provider=provider,
            cache_path=lecture_json_path.parent / ENRICHMENT_PARTIAL_NAME,
            progress_callback=lambda payload: _emit(
                progress_callback,
                "enrichment_progress",
                source=artifact["source"]["filename"],
                **payload,
            ),
            availability=availability,
        )
    else:
        chunked_analyzer = getattr(provider, "analyze_lecture_chunked", None) if provider else None
        if provider and callable(chunked_analyzer):
            response = chunked_analyzer(
                request,
                cache_path=lecture_json_path.parent / ENRICHMENT_PARTIAL_NAME,
                progress_callback=lambda payload: _emit(
                    progress_callback,
                    "enrichment_progress",
                    source=artifact["source"]["filename"],
                    **payload,
                ),
            )
        elif provider:
            response = provider.analyze_lecture(request)
        else:
            provider = build_provider("local-stub", model="local-stub-v0")
            response = provider.analyze_lecture(request)
    finished_at = utc_now_iso()
    enrichment = _enrichment_payload(
        response,
        provider="experimental-routing" if routing_enabled else config.ai_provider.value,
        model=config.ai_model or info.default_model,
        model_routing=config.ai_model_routing if routing_enabled else None,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=time.monotonic() - started,
    )
    artifact["enrichment"] = enrichment
    save_json(lecture_json_path, artifact)
    _emit(
        progress_callback,
        "enrichment_finished",
        source=artifact["source"]["filename"],
        title=enrichment["title"],
        elapsed_seconds=round(enrichment["elapsed_seconds"], 1),
        input_tokens=enrichment["input_token_estimate"],
        output_tokens=enrichment["output_token_estimate"],
    )
    return artifact


def enrich_lecture_artifacts_staged(
    lecture_json_paths: List[Path],
    config: BatchConfig,
    *,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> Dict[Path, Dict]:
    if config.ai_provider is AIProviderName.NONE:
        return {path: load_json(path) for path in lecture_json_paths}

    if not lecture_json_paths:
        return {}

    routing_enabled = uses_experimental_routing(config)
    if not routing_enabled or config.ai_uses_gemini:
        return {
            path: enrich_lecture_artifact(path, config, progress_callback=progress_callback)
            for path in lecture_json_paths
        }

    availability = _probe_availability(config)
    artifacts = {path: load_json(path) for path in lecture_json_paths}
    if availability.warnings:
        for path in lecture_json_paths:
            _append_processing_warnings(path, availability.warnings)
        artifacts = {path: load_json(path) for path in lecture_json_paths}
    started_at_by_path = {path: utc_now_iso() for path in lecture_json_paths}
    started_monotonic_by_path = {path: time.monotonic() for path in lecture_json_paths}
    jobs = []
    key_to_path = {}
    key_to_source = {}

    for path, artifact in artifacts.items():
        source_name = artifact["source"]["filename"]
        _emit(progress_callback, "enrichment_started", source=source_name)
        key = str(path)
        key_to_path[key] = path
        key_to_source[key] = source_name
        jobs.append((key, _request_from_artifact(artifact, path.parent)))

    def emit_staged_progress(payload: Dict) -> None:
        key = str(payload.get("lecture_id") or "")
        source = key_to_source.get(key, "")
        clean_payload = {name: value for name, value in payload.items() if name != "lecture_id"}
        _emit(progress_callback, "enrichment_progress", source=source, **clean_payload)

    responses = routed_analyze_lectures_staged(
        jobs,
        config,
        progress_callback=emit_staged_progress,
        availability=availability,
    )
    enriched_artifacts: Dict[Path, Dict] = {}
    for key, response in responses.items():
        path = key_to_path[key]
        artifact = artifacts[path]
        finished_at = utc_now_iso()
        enrichment = _enrichment_payload(
            response,
            provider="experimental-routing",
            model=config.ai_model,
            model_routing=config.ai_model_routing,
            started_at=started_at_by_path[path],
            finished_at=finished_at,
            elapsed_seconds=time.monotonic() - started_monotonic_by_path[path],
        )
        artifact["enrichment"] = enrichment
        save_json(path, artifact)
        enriched_artifacts[path] = artifact
        _emit(
            progress_callback,
            "enrichment_finished",
            source=artifact["source"]["filename"],
            title=enrichment["title"],
            elapsed_seconds=round(enrichment["elapsed_seconds"], 1),
            input_tokens=enrichment["input_token_estimate"],
            output_tokens=enrichment["output_token_estimate"],
        )
    return enriched_artifacts


def _build_provider_for_config(config: BatchConfig, info):
    if config.ai_uses_gemini:
        return build_provider(
            "gemini",
            api_key=_api_key_for(AIProviderName.GEMINI),
            model=config.ai_model or info.default_model,
            max_concurrency=config.gemini_max_concurrency,
        )
    if config.ai_provider is AIProviderName.MOCK:
        return build_provider("mock", model=config.ai_model or info.default_model)
    return None


def _provider_info_name(config: BatchConfig) -> str:
    if config.ai_provider is AIProviderName.GEMINI:
        return "gemini"
    if config.ai_provider is AIProviderName.LM_STUDIO:
        return "mlx-text"
    return config.ai_provider.value


def _probe_availability(config: BatchConfig) -> RouteAvailability:
    if not config.ai_uses_mlx:
        return RouteAvailability()
    return probe_lm_studio_model_availability(config)


def _request_from_artifact(artifact: Dict, lecture_dir: Path) -> AnalyzeLectureRequest:
    duration = float(artifact.get("media", {}).get("duration_seconds") or 0.0) / 60.0
    return AnalyzeLectureRequest(
        lecture_id=artifact["lecture_id"],
        transcript_text=artifact.get("transcript", {}).get("text") or "",
        segments=list(artifact.get("transcript", {}).get("segments") or []),
        slides=list(artifact.get("slides") or []),
        duration_minutes=duration,
        lecture_dir=lecture_dir,
    )


def _enrichment_payload(
    response: AnalyzeLectureResponse,
    *,
    provider: str,
    model: str,
    model_routing: Optional[Dict] = None,
    started_at: str,
    finished_at: str,
    elapsed_seconds: float,
) -> Dict:
    payload = {
        "schema_version": "1.0.0",
        "provider": provider,
        "model": model,
        "prompt_version": "v1",
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "input_token_estimate": response.input_token_estimate,
        "output_token_estimate": response.output_token_estimate,
        "title": response.title,
        "executive_summary": response.executive_summary,
        "formatted_transcript": response.formatted_transcript,
        "outline": response.outline,
        "slide_analysis": response.slide_analysis,
        "resources": response.resources,
        "warnings": response.warnings,
    }
    if model_routing:
        payload["model_routing"] = model_routing
    return payload


def _append_processing_warnings(lecture_json_path: Path, warnings: List[str]) -> None:
    if not warnings:
        return
    artifact = load_json(lecture_json_path)
    processing = artifact.setdefault("processing", {})
    existing = list(processing.get("warnings") or [])
    seen = {str(item) for item in existing}
    for warning in warnings:
        text = str(warning or "").strip()
        if text and text not in seen:
            existing.append(text)
            seen.add(text)
    processing["warnings"] = existing
    save_json(lecture_json_path, artifact)


def _api_key_for(provider: AIProviderName) -> str:
    if provider is AIProviderName.MOCK:
        return ""
    if provider is AIProviderName.GEMINI:
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("LECTURE_PROCESSOR_GEMINI_API_KEY") or ""
    return ""


def _emit(progress_callback: Optional[Callable[[Dict], None]], kind: str, **payload) -> None:
    if not progress_callback:
        return
    try:
        progress_callback({"kind": kind, **payload})
    except Exception:
        pass
