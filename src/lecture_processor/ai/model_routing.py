from pathlib import Path
from typing import Callable, Dict, Optional

from lecture_processor.config import AIModelProvider, BatchConfig

from .providers.base import AnalyzeLectureRequest, AnalyzeLectureResponse
from .providers.local_stub import (
    disabled_formatted_transcript,
    disabled_overview,
    disabled_resources,
    disabled_slide_analysis,
    local_formatted_transcript,
    local_overview,
    local_resources,
    local_slide_analysis,
)
from .providers.mlx_openai import MLXTextProvider, MLXVisionProvider

AI_ROUTE_STEPS = ("overview", "transcript", "slides", "resources")
MLX_PROVIDERS = (AIModelProvider.MLX_TEXT, AIModelProvider.MLX_VISION)


def uses_experimental_routing(config: BatchConfig) -> bool:
    return any(config.ai_step_provider(step) is not AIModelProvider.GEMINI for step in AI_ROUTE_STEPS)


def routed_analyze_lecture(
    request: AnalyzeLectureRequest,
    config: BatchConfig,
    *,
    gemini_provider=None,
    cache_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> AnalyzeLectureResponse:
    gemini_response = None
    progress = _LocalProgress(progress_callback, _local_route_count(config))
    overview_override = None
    if config.ai_uses_gemini:
        overview_provider = config.ai_overview_provider
        needs_gemini_overview = overview_provider is AIModelProvider.GEMINI
        if not needs_gemini_overview and (
            config.ai_slides_provider is AIModelProvider.GEMINI
            or config.ai_resources_provider is AIModelProvider.GEMINI
        ):
            overview_override = _route_overview(request, config, None)
            progress.emit("overview", overview_override)
        gemini_response = gemini_provider.analyze_lecture_chunked(
            request,
            cache_path=cache_path,
            progress_callback=progress_callback,
            include_overview=needs_gemini_overview,
            include_transcript=config.ai_transcript_provider is AIModelProvider.GEMINI,
            include_slides=config.ai_slides_provider is AIModelProvider.GEMINI,
            include_resources=config.ai_resources_provider is AIModelProvider.GEMINI,
            overview_override=overview_override,
        )

    overview = overview_override or _route_overview(request, config, gemini_response)
    if overview_override is None and config.ai_overview_provider is not AIModelProvider.GEMINI:
        progress.emit("overview", overview)
    transcript = _route_transcript(request, config, gemini_response)
    if config.ai_transcript_provider is not AIModelProvider.GEMINI:
        progress.emit("transcript", transcript)
    slides = _route_slides(request, config, gemini_response)
    if config.ai_slides_provider is not AIModelProvider.GEMINI:
        progress.emit("slides", slides)
    resources = _route_resources(request, config, gemini_response, overview=overview)
    if config.ai_resources_provider is not AIModelProvider.GEMINI:
        progress.emit("resources", resources)

    warnings = []
    if gemini_response:
        warnings.extend(gemini_response.warnings)
    for payload in (overview, transcript, slides, resources):
        warnings.extend(payload.get("warnings") or [])
    warnings.append("Experimental model routing is active.")

    return AnalyzeLectureResponse(
        title=overview["title"],
        executive_summary=overview["executive_summary"],
        outline=overview["outline"],
        formatted_transcript=transcript["formatted_transcript"],
        slide_analysis=slides["slide_analysis"],
        resources=resources["resources"],
        input_token_estimate=(gemini_response.input_token_estimate if gemini_response else 0)
        + _sum_tokens("input_tokens", overview, transcript, slides, resources)
        or _local_input_estimate(request),
        output_token_estimate=(gemini_response.output_token_estimate if gemini_response else 0)
        + _sum_tokens("output_tokens", overview, transcript, slides, resources)
        or 350,
        raw_response_id="experimental-routing",
        warnings=_dedupe_warnings(warnings),
    )


def _route_overview(request: AnalyzeLectureRequest, config: BatchConfig, gemini_response) -> Dict:
    provider = config.ai_overview_provider
    if provider is AIModelProvider.GEMINI and gemini_response:
        return {
            "title": gemini_response.title,
            "executive_summary": gemini_response.executive_summary,
            "outline": gemini_response.outline,
            "warnings": [],
        }
    if provider is AIModelProvider.LOCAL_STUB:
        return local_overview(request, config.ai_step_model("overview"))
    if provider in MLX_PROVIDERS:
        return _mlx_provider(provider, config, "overview").analyze_overview(request)
    return disabled_overview(request)


def _route_transcript(request: AnalyzeLectureRequest, config: BatchConfig, gemini_response) -> Dict:
    provider = config.ai_transcript_provider
    if provider is AIModelProvider.GEMINI and gemini_response:
        return {"formatted_transcript": gemini_response.formatted_transcript, "warnings": []}
    if provider is AIModelProvider.LOCAL_STUB:
        return local_formatted_transcript(request, config.ai_step_model("transcript"))
    if provider in MLX_PROVIDERS:
        return _mlx_provider(provider, config, "transcript").analyze_transcript(request)
    return disabled_formatted_transcript(request)


def _route_slides(request: AnalyzeLectureRequest, config: BatchConfig, gemini_response) -> Dict:
    provider = config.ai_slides_provider
    if provider is AIModelProvider.GEMINI and gemini_response:
        return {"slide_analysis": gemini_response.slide_analysis, "warnings": []}
    if provider is AIModelProvider.LOCAL_STUB:
        return local_slide_analysis(request, config.ai_step_model("slides"))
    if provider in MLX_PROVIDERS:
        return _mlx_provider(provider, config, "slides").analyze_slides(request)
    return disabled_slide_analysis(request)


def _route_resources(
    request: AnalyzeLectureRequest,
    config: BatchConfig,
    gemini_response,
    *,
    overview: Optional[Dict] = None,
) -> Dict:
    provider = config.ai_resources_provider
    if provider is AIModelProvider.GEMINI and gemini_response:
        return {"resources": gemini_response.resources, "warnings": []}
    if provider is AIModelProvider.LOCAL_STUB:
        return local_resources(request, config.ai_step_model("resources"))
    if provider in MLX_PROVIDERS:
        return _mlx_provider(provider, config, "resources").analyze_resources(request, overview=overview)
    return disabled_resources()


def _mlx_provider(provider: AIModelProvider, config: BatchConfig, step: str):
    kwargs = {
        "model": config.ai_step_model(step),
        "timeout_seconds": config.mlx_request_timeout_seconds,
    }
    if provider is AIModelProvider.MLX_VISION:
        return MLXVisionProvider(base_url=config.mlx_vision_base_url, **kwargs)
    return MLXTextProvider(base_url=config.mlx_text_base_url, **kwargs)


class _LocalProgress:
    def __init__(self, progress_callback: Optional[Callable[[Dict], None]], total: int) -> None:
        self.progress_callback = progress_callback
        self.total = total
        self.completed = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def emit(self, step: str, payload: Dict) -> None:
        if not self.progress_callback or self.total <= 0:
            return
        self.completed += 1
        self.input_tokens += int(payload.get("input_tokens") or 0)
        self.output_tokens += int(payload.get("output_tokens") or 0)
        self.progress_callback(
            {
                "step": f"local {step}",
                "completed": self.completed,
                "total": self.total,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            }
        )


def _local_route_count(config: BatchConfig) -> int:
    return sum(
        1 for step in AI_ROUTE_STEPS if config.ai_step_provider(step) is not AIModelProvider.GEMINI
    )


def _sum_tokens(key: str, *payloads: Dict) -> int:
    return sum(int(payload.get(key) or 0) for payload in payloads)


def _local_input_estimate(request: AnalyzeLectureRequest) -> int:
    return max(1, len(request.transcript_text.split()) + (50 * len(request.slides or [])))


def _dedupe_warnings(warnings):
    seen = set()
    deduped = []
    for warning in warnings:
        text = str(warning or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped
