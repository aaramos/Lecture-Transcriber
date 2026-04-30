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

AI_ROUTE_STEPS = ("overview", "transcript", "slides", "resources")


def uses_experimental_routing(config: BatchConfig) -> bool:
    return config.ai_uses_local_stub or any(
        config.ai_step_provider(step) is AIModelProvider.OFF for step in AI_ROUTE_STEPS
    )


def routed_analyze_lecture(
    request: AnalyzeLectureRequest,
    config: BatchConfig,
    *,
    gemini_provider=None,
    cache_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> AnalyzeLectureResponse:
    gemini_response = None
    if config.ai_uses_gemini:
        overview_provider = config.ai_overview_provider
        needs_gemini_overview = overview_provider is AIModelProvider.GEMINI
        overview_override = None
        if not needs_gemini_overview and (
            config.ai_slides_provider is AIModelProvider.GEMINI
            or config.ai_resources_provider is AIModelProvider.GEMINI
        ):
            overview_override = _route_overview(request, config, None)
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
    else:
        _emit_local_progress(progress_callback)

    overview = _route_overview(request, config, gemini_response)
    transcript = _route_transcript(request, config, gemini_response)
    slides = _route_slides(request, config, gemini_response)
    resources = _route_resources(request, config, gemini_response)

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
        input_token_estimate=(gemini_response.input_token_estimate if gemini_response else _local_input_estimate(request)),
        output_token_estimate=(gemini_response.output_token_estimate if gemini_response else 350),
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
    return disabled_overview(request)


def _route_transcript(request: AnalyzeLectureRequest, config: BatchConfig, gemini_response) -> Dict:
    provider = config.ai_transcript_provider
    if provider is AIModelProvider.GEMINI and gemini_response:
        return {"formatted_transcript": gemini_response.formatted_transcript, "warnings": []}
    if provider is AIModelProvider.LOCAL_STUB:
        return local_formatted_transcript(request, config.ai_step_model("transcript"))
    return disabled_formatted_transcript(request)


def _route_slides(request: AnalyzeLectureRequest, config: BatchConfig, gemini_response) -> Dict:
    provider = config.ai_slides_provider
    if provider is AIModelProvider.GEMINI and gemini_response:
        return {"slide_analysis": gemini_response.slide_analysis, "warnings": []}
    if provider is AIModelProvider.LOCAL_STUB:
        return local_slide_analysis(request, config.ai_step_model("slides"))
    return disabled_slide_analysis(request)


def _route_resources(request: AnalyzeLectureRequest, config: BatchConfig, gemini_response) -> Dict:
    provider = config.ai_resources_provider
    if provider is AIModelProvider.GEMINI and gemini_response:
        return {"resources": gemini_response.resources, "warnings": []}
    if provider is AIModelProvider.LOCAL_STUB:
        return local_resources(request, config.ai_step_model("resources"))
    return disabled_resources()


def _emit_local_progress(progress_callback: Optional[Callable[[Dict], None]]) -> None:
    if not progress_callback:
        return
    for index, step in enumerate(AI_ROUTE_STEPS, start=1):
        progress_callback(
            {
                "step": f"local {step}",
                "completed": index,
                "total": len(AI_ROUTE_STEPS),
                "input_tokens": 0,
                "output_tokens": 0,
            }
        )


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
