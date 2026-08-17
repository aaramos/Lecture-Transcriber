import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from lecture_processor.config import AIModelProvider, AIProviderName, BatchConfig

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
from .providers.mlx_openai import (
    DEFAULT_LOCAL_MODEL,
    DEFAULT_ROLE_MAX_TOKENS,
    MLXTextProvider,
    MLXVisionProvider,
    _OpenAICompatibleClient,
    clear_lm_studio_model_verification_cache,
)

AI_ROUTE_STEPS = ("overview", "transcript", "slides", "resources")
AI_PHASES = (
    ("text", ("overview", "transcript")),
    ("vision", ("slides",)),
    ("resource", ("resources",)),
)
AI_STEP_SEQUENCE = tuple(step for _role, steps in AI_PHASES for step in steps)
AI_STEP_DEPENDENCIES = {
    "overview": set(),
    "transcript": {"overview"},
    "slides": set(),
    "resources": {"overview"},
}
AI_ROLE_BY_STEP = {
    step: role
    for role, steps in AI_PHASES
    for step in steps
}
MLX_PROVIDERS = (
    AIModelProvider.MLX_TEXT,
    AIModelProvider.MLX_VISION,
    AIModelProvider.OLLAMA_CLOUD,
)
AI_ROLE_MODEL_CONFIG = {
    role: {"max_tokens": max_tokens}
    for role, max_tokens in DEFAULT_ROLE_MAX_TOKENS.items()
}
AI_TIMING_KEY_BY_STEP = {
    "overview": "overview",
    "transcript": "transcript_cleanup",
    "slides": "slide_analysis",
    "resources": "resource_formatter",
}
AI_TOKEN_KEY_BY_STEP = AI_TIMING_KEY_BY_STEP


@dataclass(frozen=True)
class RouteAvailability:
    skipped_roles: Dict[str, str] = field(default_factory=dict)
    resolved_step_models: Dict[str, str] = field(default_factory=dict)

    def skip_reason_for_step(self, step: str) -> str:
        role = AI_ROLE_BY_STEP.get(step, step)
        return str(self.skipped_roles.get(role) or "").strip()

    def model_for_step(self, step: str, config: BatchConfig) -> str:
        return str(self.resolved_step_models.get(step) or config.ai_step_model(step)).strip()

    @property
    def warnings(self) -> List[str]:
        return [reason for _role, reason in self.skipped_roles.items() if reason]


@dataclass(frozen=True)
class _LocalModelRef:
    provider: AIModelProvider
    base_url: str
    model: str


def uses_experimental_routing(config: BatchConfig) -> bool:
    if config.ai_provider in (AIProviderName.NONE, AIProviderName.MOCK):
        return False
    if config.ai_provider is AIProviderName.LM_STUDIO:
        return True
    return any(config.ai_step_provider(step) is not AIModelProvider.GEMINI for step in AI_ROUTE_STEPS)


def probe_lm_studio_model_availability(config: BatchConfig) -> RouteAvailability:
    skipped_roles: Dict[str, str] = {}
    resolved_step_models: Dict[str, str] = {}
    models_by_base_url: Dict[str, List[str]] = {}
    errors_by_base_url: Dict[str, str] = {}

    for role, steps in AI_PHASES:
        local_steps = [
            step
            for step in steps
            if config.ai_step_provider(step) in MLX_PROVIDERS
        ]
        if not local_steps:
            continue

        missing: List[str] = []
        lookup_errors: List[str] = []
        for step in local_steps:
            provider = config.ai_step_provider(step)
            base_url = _base_url_for_provider(provider, config)
            if base_url not in models_by_base_url:
                try:
                    models_by_base_url[base_url] = _list_models_for_provider(provider, config)
                except Exception as exc:
                    models_by_base_url[base_url] = []
                    errors_by_base_url[base_url] = str(exc)
            if base_url in errors_by_base_url:
                lookup_errors.append(errors_by_base_url[base_url])
                continue
            available_models = models_by_base_url.get(base_url) or []
            selected_model = config.ai_step_model(step)
            if step == "slides" and selected_model == DEFAULT_LOCAL_MODEL:
                selected_model = _default_slide_analysis_model(config, resolved_step_models)
            if selected_model == DEFAULT_LOCAL_MODEL:
                if available_models:
                    resolved_step_models[step] = available_models[0]
                else:
                    missing.append("any loaded LM Studio model")
                continue
            if selected_model in available_models:
                resolved_step_models[step] = selected_model
            else:
                missing.append(selected_model)

        if lookup_errors:
            reason = f"LM Studio {role} model lookup failed: {'; '.join(_dedupe_warnings(lookup_errors))}"
            skipped_roles[role] = reason
            continue
        if missing:
            reason = (
                f"LM Studio {role} model unavailable: {', '.join(_dedupe_warnings(missing))}. "
                "Load the selected model in LM Studio or refresh model choices in Settings."
            )
            skipped_roles[role] = reason

    return RouteAvailability(skipped_roles=skipped_roles, resolved_step_models=resolved_step_models)


def _default_slide_analysis_model(config: BatchConfig, resolved_step_models: Dict[str, str]) -> str:
    for step in ("overview", "transcript"):
        resolved = str(resolved_step_models.get(step) or "").strip()
        if resolved:
            return resolved
        explicit = str(getattr(config, f"ai_{step}_model") or "").strip()
        if explicit:
            return explicit
    return DEFAULT_LOCAL_MODEL


def routed_analyze_lecture(
    request: AnalyzeLectureRequest,
    config: BatchConfig,
    *,
    gemini_provider=None,
    cache_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[Dict], None]] = None,
    availability: Optional[RouteAvailability] = None,
) -> AnalyzeLectureResponse:
    gemini_response = None
    progress = _LocalProgress(progress_callback, _local_route_count(config))
    overview_override = None
    transition_warnings: List[str] = []
    route_steps = _ordered_ai_steps(config, availability)
    if config.ai_uses_gemini:
        overview_provider = config.ai_overview_provider
        needs_gemini_overview = overview_provider is AIModelProvider.GEMINI
        if not needs_gemini_overview and (
            config.ai_slides_provider is AIModelProvider.GEMINI
            or config.ai_resources_provider is AIModelProvider.GEMINI
        ):
            clear_lm_studio_model_verification_cache()
            overview_override = _route_overview(request, config, None, availability=availability)
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

    step_results = {
        "overview": None,
        "transcript": None,
        "slides": None,
        "resources": None,
    }
    step_timings: Dict[str, Dict] = {}
    step_token_usage: Dict[str, Dict] = {}
    step_outcomes: Dict[str, Dict] = {}
    if gemini_response:
        _merge_step_token_usage(step_token_usage, gemini_response.step_token_usage)
    for index, (_role, step) in enumerate(route_steps):
        clear_lm_studio_model_verification_cache()
        timer = _start_step_timer()
        if step == "overview":
            payload = overview_override or _route_overview(request, config, gemini_response, availability=availability)
        elif step == "transcript":
            payload = _route_transcript(request, config, gemini_response, availability=availability)
        elif step == "slides":
            payload = _route_slides(request, config, gemini_response, availability=availability)
        else:
            payload = _route_resources(
                request,
                config,
                gemini_response,
                overview=step_results.get("overview"),
                availability=availability,
            )
        step_results[step] = payload
        timing = _finish_step_timer(timer, **_timing_metadata(step, payload))
        _record_step_timing(step_timings, step, timing, payload)
        _record_step_token_usage(step_token_usage, step, payload)
        outcome = _step_outcome_entry(step, payload)
        step_outcomes[AI_TOKEN_KEY_BY_STEP.get(step, step)] = outcome
        if config.ai_step_provider(step) is not AIModelProvider.GEMINI and not (
            step == "overview" and overview_override is not None
        ):
            progress_payload = dict(payload)
            progress_payload["step_elapsed_seconds"] = timing.get("elapsed_seconds")
            progress_payload["step_outcome"] = outcome["outcome"]
            progress_payload["step_message"] = outcome["message"]
            progress_payload.update(_slide_progress_counts(step, request, payload))
            progress.emit(step, progress_payload)

        next_step = route_steps[index + 1][1] if index + 1 < len(route_steps) else None
        _append_transition_warning(
            transition_warnings,
            _offload_model_between_steps(step, next_step, config, availability),
        )

    overview = step_results["overview"] or disabled_overview(request)
    transcript = step_results["transcript"] or disabled_formatted_transcript(request)
    slides = step_results["slides"] or disabled_slide_analysis(request)
    resources = step_results["resources"] or disabled_resources()

    warnings = []
    if gemini_response:
        warnings.extend(gemini_response.warnings)
    for payload in (overview, transcript, slides, resources):
        warnings.extend(payload.get("warnings") or [])
    warnings.extend(transition_warnings)
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
        step_timings=step_timings,
        step_token_usage=step_token_usage,
        step_outcomes=step_outcomes,
    )


def routed_analyze_lectures_staged(
    jobs: List[Tuple[str, AnalyzeLectureRequest]],
    config: BatchConfig,
    *,
    progress_callback: Optional[Callable[[Dict], None]] = None,
    availability: Optional[RouteAvailability] = None,
) -> Dict[str, AnalyzeLectureResponse]:
    """Run one AI role phase at a time across all lectures.

    LM Studio can unload models when multiple local models are active at once.
    This staged path keeps each configured role/model phase isolated before
    moving to the next one.
    """
    availability = availability or RouteAvailability()
    states = {
        key: {
            "request": request,
            "overview": None,
            "transcript": None,
            "slides": None,
            "resources": None,
            "warnings": [],
            "step_timings": {},
            "step_token_usage": {},
            "step_outcomes": {},
            "input_tokens": 0,
            "output_tokens": 0,
        }
        for key, request in jobs
    }
    total_steps = max(1, len(jobs) * len(AI_ROUTE_STEPS))
    completed_steps = 0
    input_tokens = 0
    output_tokens = 0

    ordered_steps = _ordered_ai_steps(config, availability)
    previous_stage_ref: Optional[_LocalModelRef] = None
    for index, (role, step) in enumerate(ordered_steps):
        current_stage_ref = _local_model_ref_for_step(step, config, availability)
        if index == 0 or not _same_model_ref(previous_stage_ref, current_stage_ref):
            clear_lm_studio_model_verification_cache()
        for key, request in jobs:
            state = states[key]
            timer = _start_step_timer()
            if step == "overview":
                payload = _route_overview(request, config, None, availability=availability)
            elif step == "transcript":
                payload = _route_transcript(request, config, None, availability=availability)
            elif step == "slides":
                payload = _route_slides(request, config, None, availability=availability)
            else:
                payload = _route_resources(
                    request,
                    config,
                    None,
                    overview=state.get("overview"),
                    availability=availability,
                )

            state[step] = payload
            timing = _finish_step_timer(timer, **_timing_metadata(step, payload))
            _record_step_timing(
                state["step_timings"],
                step,
                timing,
                payload,
            )
            state["warnings"].extend(payload.get("warnings") or [])
            step_usage = _token_usage_entry(
                payload.get("input_tokens"),
                payload.get("output_tokens"),
            )
            _record_step_token_usage(state["step_token_usage"], step, payload)
            outcome = _step_outcome_entry(step, payload)
            state["step_outcomes"][AI_TOKEN_KEY_BY_STEP.get(step, step)] = outcome
            state["input_tokens"] += step_usage["input_tokens"]
            state["output_tokens"] += step_usage["output_tokens"]
            input_tokens += step_usage["input_tokens"]
            output_tokens += step_usage["output_tokens"]
            completed_steps += 1
            if progress_callback:
                progress_callback(
                    {
                        "lecture_id": key,
                        "step": f"staged {role} {step}",
                        "completed": completed_steps,
                        "total": total_steps,
                        "input_tokens": state["input_tokens"],
                        "output_tokens": state["output_tokens"],
                        "step_input_tokens": step_usage["input_tokens"],
                        "step_output_tokens": step_usage["output_tokens"],
                        "step_elapsed_seconds": timing.get("elapsed_seconds"),
                        "step_outcome": outcome["outcome"],
                        "step_message": outcome["message"],
                        "batch_input_tokens": input_tokens,
                        "batch_output_tokens": output_tokens,
                        "provider": config.ai_step_provider(step).value,
                        "model": availability.model_for_step(step, config),
                        "role": role,
                        **_slide_progress_counts(step, request, payload),
                    }
                )
        next_step = ordered_steps[index + 1][1] if index + 1 < len(ordered_steps) else None
        if next_step:
            transition_warning = _offload_model_between_steps(step, next_step, config, availability)
            if transition_warning:
                for state in states.values():
                    state["warnings"].append(transition_warning)
        previous_stage_ref = current_stage_ref

    responses: Dict[str, AnalyzeLectureResponse] = {}
    for key, state in states.items():
        request = state["request"]
        overview = state.get("overview") or disabled_overview(request)
        transcript = state.get("transcript") or disabled_formatted_transcript(request)
        slides = state.get("slides") or disabled_slide_analysis(request)
        resources = state.get("resources") or disabled_resources()
        warnings = []
        for payload in (overview, transcript, slides, resources):
            warnings.extend(payload.get("warnings") or [])
        warnings.extend(state["warnings"])
        warnings.append("Experimental model routing is active.")
        warnings.append("Batch AI ran in staged single-model order.")
        responses[key] = AnalyzeLectureResponse(
            title=overview["title"],
            executive_summary=overview["executive_summary"],
            outline=overview["outline"],
            formatted_transcript=transcript["formatted_transcript"],
            slide_analysis=slides["slide_analysis"],
            resources=resources["resources"],
            input_token_estimate=_sum_tokens("input_tokens", overview, transcript, slides, resources)
            or _local_input_estimate(request),
            output_token_estimate=_sum_tokens("output_tokens", overview, transcript, slides, resources)
            or 350,
            raw_response_id="experimental-routing-staged",
            warnings=_dedupe_warnings(warnings),
            step_timings=state["step_timings"],
            step_token_usage=state["step_token_usage"],
            step_outcomes=state["step_outcomes"],
        )
    return responses


def _ordered_ai_steps(config: BatchConfig, availability: Optional[RouteAvailability]) -> List[Tuple[str, str]]:
    completed_steps: List[str] = []
    scheduled = set()

    while len(completed_steps) < len(AI_STEP_SEQUENCE):
        ready = [
            step
            for step in AI_STEP_SEQUENCE
            if step not in scheduled and AI_STEP_DEPENDENCIES[step].issubset(scheduled)
        ]
        if not ready:
            break

        previous_step = completed_steps[-1] if completed_steps else None
        previous_ref = _local_model_ref_for_step(previous_step, config, availability) if previous_step else None
        next_step = None

        if previous_ref:
            for candidate in ready:
                candidate_ref = _local_model_ref_for_step(candidate, config, availability)
                if _same_model_ref(previous_ref, candidate_ref):
                    next_step = candidate
                    break

        if next_step is None:
            next_step = ready[0]

        completed_steps.append(next_step)
        scheduled.add(next_step)

    if len(completed_steps) < len(AI_STEP_SEQUENCE):
        for step in AI_STEP_SEQUENCE:
            if step not in scheduled:
                completed_steps.append(step)

    return [(AI_ROLE_BY_STEP[step], step) for step in completed_steps]


def _route_overview(
    request: AnalyzeLectureRequest,
    config: BatchConfig,
    gemini_response,
    *,
    availability: Optional[RouteAvailability] = None,
) -> Dict:
    skipped = (availability or RouteAvailability()).skip_reason_for_step("overview")
    if skipped:
        return _disabled_payload_for_step("overview", request, skipped)
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
        return _mlx_provider(provider, config, "overview", availability=availability).analyze_overview(request)
    return disabled_overview(request)


def _route_transcript(
    request: AnalyzeLectureRequest,
    config: BatchConfig,
    gemini_response,
    *,
    availability: Optional[RouteAvailability] = None,
) -> Dict:
    skipped = (availability or RouteAvailability()).skip_reason_for_step("transcript")
    if skipped:
        return _disabled_payload_for_step("transcript", request, skipped)
    provider = config.ai_transcript_provider
    if provider is AIModelProvider.GEMINI and gemini_response:
        return {"formatted_transcript": gemini_response.formatted_transcript, "warnings": []}
    if provider is AIModelProvider.LOCAL_STUB:
        return local_formatted_transcript(request, config.ai_step_model("transcript"))
    if provider in MLX_PROVIDERS:
        return _mlx_provider(provider, config, "transcript", availability=availability).analyze_transcript(request)
    return disabled_formatted_transcript(request)


def _route_slides(
    request: AnalyzeLectureRequest,
    config: BatchConfig,
    gemini_response,
    *,
    availability: Optional[RouteAvailability] = None,
) -> Dict:
    skipped = (availability or RouteAvailability()).skip_reason_for_step("slides")
    if skipped:
        return _disabled_payload_for_step("slides", request, skipped)
    provider = config.ai_slides_provider
    if provider is AIModelProvider.GEMINI and gemini_response:
        return {"slide_analysis": gemini_response.slide_analysis, "warnings": []}
    if provider is AIModelProvider.LOCAL_STUB:
        return local_slide_analysis(request, config.ai_step_model("slides"))
    if provider in MLX_PROVIDERS:
        return _mlx_provider(provider, config, "slides", availability=availability).analyze_slides(request)
    return disabled_slide_analysis(request)


def _route_resources(
    request: AnalyzeLectureRequest,
    config: BatchConfig,
    gemini_response,
    *,
    overview: Optional[Dict] = None,
    availability: Optional[RouteAvailability] = None,
) -> Dict:
    skipped = (availability or RouteAvailability()).skip_reason_for_step("resources")
    if skipped:
        return _disabled_payload_for_step("resources", request, skipped)
    provider = config.ai_resources_provider
    if provider is AIModelProvider.GEMINI and gemini_response:
        return {"resources": gemini_response.resources, "warnings": []}
    if provider is AIModelProvider.LOCAL_STUB:
        return local_resources(request, config.ai_step_model("resources"))
    if provider in MLX_PROVIDERS:
        return _mlx_provider(provider, config, "resources", availability=availability).analyze_resources(
            request,
            overview=overview,
        )
    return disabled_resources()


def _mlx_provider(
    provider: AIModelProvider,
    config: BatchConfig,
    step: str,
    *,
    availability: Optional[RouteAvailability] = None,
):
    kwargs = {
        "model": (availability or RouteAvailability()).model_for_step(step, config),
        "timeout_seconds": config.mlx_request_timeout_seconds,
        "role_max_tokens": {
            role: int(role_config["max_tokens"])
            for role, role_config in AI_ROLE_MODEL_CONFIG.items()
        },
        "disable_thinking": config.mlx_disable_thinking,
    }
    if provider is AIModelProvider.MLX_VISION or (
        provider is AIModelProvider.OLLAMA_CLOUD and step == "slides"
    ):
        return MLXVisionProvider(base_url=_base_url_for_provider(provider, config), **kwargs)
    return MLXTextProvider(base_url=_base_url_for_provider(provider, config), **kwargs)


def _disabled_payload_for_step(step: str, request: AnalyzeLectureRequest, reason: str) -> Dict:
    if step == "overview":
        payload = disabled_overview(request)
    elif step == "transcript":
        payload = disabled_formatted_transcript(request)
    elif step == "slides":
        payload = disabled_slide_analysis(request)
    else:
        payload = disabled_resources()
    payload.setdefault("warnings", []).append(reason)
    payload.setdefault("warnings", []).append(f"{step.title()} skipped because its LM Studio role was unavailable.")
    return payload


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _start_step_timer() -> Dict[str, object]:
    return {"started_at": _utc_now_iso(), "monotonic": time.monotonic()}


def _finish_step_timer(timer: Dict[str, object], **metadata) -> Dict:
    payload = {
        "started_at": str(timer.get("started_at") or _utc_now_iso()),
        "finished_at": _utc_now_iso(),
        "elapsed_seconds": round(max(0.0, time.monotonic() - float(timer.get("monotonic") or time.monotonic())), 3),
    }
    for key, value in metadata.items():
        if value is not None:
            payload[key] = value
    return payload


def _record_step_timing(step_timings: Dict[str, Dict], step: str, timing: Dict, payload: Dict) -> None:
    nested = payload.get("step_timings") if isinstance(payload, dict) else None
    if isinstance(nested, dict):
        step_timings.update({str(key): value for key, value in nested.items() if isinstance(value, dict)})
    if step == "resources" and nested:
        return
    step_timings[AI_TIMING_KEY_BY_STEP.get(step, step)] = timing


def _record_step_token_usage(step_token_usage: Dict[str, Dict], step: str, payload: Dict) -> None:
    nested = payload.get("step_token_usage") if isinstance(payload, dict) else None
    if isinstance(nested, dict):
        _merge_step_token_usage(step_token_usage, nested)
    if step == "resources" and nested:
        return
    entry = _token_usage_entry(
        payload.get("input_tokens") if isinstance(payload, dict) else 0,
        payload.get("output_tokens") if isinstance(payload, dict) else 0,
    )
    if entry["total_tokens"] <= 0:
        return
    step_token_usage[AI_TOKEN_KEY_BY_STEP.get(step, step)] = entry


def _merge_step_token_usage(target: Dict[str, Dict], source: Dict) -> None:
    if not isinstance(source, dict):
        return
    for key, value in source.items():
        if not isinstance(value, dict):
            continue
        entry = _token_usage_entry(value.get("input_tokens"), value.get("output_tokens"))
        if entry["total_tokens"] <= 0:
            continue
        current = target.get(str(key)) or {}
        target[str(key)] = _token_usage_entry(
            int(current.get("input_tokens") or 0) + entry["input_tokens"],
            int(current.get("output_tokens") or 0) + entry["output_tokens"],
        )


def _token_usage_entry(input_tokens, output_tokens) -> Dict[str, int]:
    input_count = max(0, int(input_tokens or 0))
    output_count = max(0, int(output_tokens or 0))
    return {
        "input_tokens": input_count,
        "output_tokens": output_count,
        "total_tokens": input_count + output_count,
    }


def _timing_metadata(step: str, payload: Dict) -> Dict:
    if step == "transcript":
        return {"chunk_count": int(payload.get("chunk_count") or 0)}
    if step == "slides":
        return {
            "batch_count": int(payload.get("batch_count") or 0),
            "retry_count": int(payload.get("retry_count") or 0),
        }
    return {}


def _step_outcome_entry(step: str, payload: Dict) -> Dict[str, str]:
    warnings = " ".join(str(item or "") for item in (payload.get("warnings") or []))
    lowered = warnings.lower()
    if step == "slides" and "smart slide extraction metadata" in lowered:
        return {"outcome": "metadata_used", "message": "Metadata used"}
    if any(marker in lowered for marker in ("route is off", "skipped because", "unavailable")):
        return {"outcome": "skipped", "message": "No model call"}
    if step == "transcript" and ("raw fallback" in lowered or "used raw" in lowered):
        return {"outcome": "fallback", "message": "Raw transcript kept"}
    if "fallback" in lowered or "local stub" in lowered:
        return {"outcome": "fallback", "message": "Fallback used"}
    return {"outcome": "success", "message": "Generated"}


def _base_url_for_provider(provider: AIModelProvider, config: BatchConfig) -> str:
    if provider is AIModelProvider.OLLAMA_CLOUD:
        return config.ollama_cloud_base_url
    if provider is AIModelProvider.MLX_VISION:
        return config.mlx_vision_base_url
    return config.mlx_text_base_url


def _list_models_for_provider(provider: AIModelProvider, config: BatchConfig) -> List[str]:
    base_url = _base_url_for_provider(provider, config)
    return _OpenAICompatibleClient(base_url, DEFAULT_LOCAL_MODEL, config.mlx_request_timeout_seconds).list_models()


def _offload_model_between_steps(
    completed_step: str,
    next_step: Optional[str],
    config: BatchConfig,
    availability: Optional[RouteAvailability],
) -> Optional[str]:
    return None


def _local_model_ref_for_step(
    step: Optional[str],
    config: BatchConfig,
    availability: Optional[RouteAvailability],
) -> Optional[_LocalModelRef]:
    if not step:
        return None
    availability = availability or RouteAvailability()
    if availability.skip_reason_for_step(step):
        return None
    provider = config.ai_step_provider(step)
    if provider not in MLX_PROVIDERS:
        return None
    model = availability.model_for_step(step, config)
    if not model or model == DEFAULT_LOCAL_MODEL:
        return None
    return _LocalModelRef(
        provider=provider,
        base_url=_base_url_for_provider(provider, config),
        model=model,
    )


def _same_model_ref(left: Optional[_LocalModelRef], right: Optional[_LocalModelRef]) -> bool:
    if left is None or right is None:
        return False
    return left.base_url == right.base_url and left.model == right.model


def _append_transition_warning(warnings: List[str], warning: Optional[str]) -> None:
    if warning:
        warnings.append(warning)


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
        step_usage = _token_usage_entry(payload.get("input_tokens"), payload.get("output_tokens"))
        self.input_tokens += step_usage["input_tokens"]
        self.output_tokens += step_usage["output_tokens"]
        self.progress_callback(
            {
                "step": f"local {step}",
                "completed": self.completed,
                "total": self.total,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "step_input_tokens": step_usage["input_tokens"],
                "step_output_tokens": step_usage["output_tokens"],
                "step_elapsed_seconds": payload.get("step_elapsed_seconds"),
                "step_outcome": payload.get("step_outcome"),
                "step_message": payload.get("step_message"),
                **_progress_count_fields(payload),
            }
        )


def _local_route_count(config: BatchConfig) -> int:
    return sum(
        1 for step in AI_ROUTE_STEPS if config.ai_step_provider(step) is not AIModelProvider.GEMINI
    )


def _slide_progress_counts(step: str, request: AnalyzeLectureRequest, payload: Dict) -> Dict:
    if step != "slides":
        return {}
    input_count = len(request.slides or [])
    remaining_count = len(payload.get("slide_analysis") or [])
    return {
        "input_slide_count": input_count,
        "slide_count_remaining": remaining_count,
        "remaining_slide_count": remaining_count,
    }


def _progress_count_fields(payload: Dict) -> Dict:
    keys = ("input_slide_count", "slide_count_remaining", "remaining_slide_count")
    return {key: payload[key] for key in keys if key in payload}


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
