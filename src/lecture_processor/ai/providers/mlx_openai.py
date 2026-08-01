import base64
from contextlib import contextmanager
try:
    import fcntl
except ImportError:  # pragma: no cover - non-macOS fallback
    fcntl = None
import json
import os
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import (
    AnalyzeLectureRequest,
    AnalyzeLectureResponse,
    ProviderInfo,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTransientError,
)
from .gemini import _json_from_response_text, _slide_image_bytes, _slide_image_path
from .local_stub import (
    local_formatted_transcript,
    local_overview,
    local_slide_analysis,
)
from ...config import DEFAULT_LM_STUDIO_BASE_URL

DEFAULT_LOCAL_MODEL = "default"
DEFAULT_TEXT_BASE_URL = DEFAULT_LM_STUDIO_BASE_URL
DEFAULT_VISION_BASE_URL = DEFAULT_LM_STUDIO_BASE_URL
DEFAULT_TIMEOUT_SECONDS = 120
TRANSCRIPT_CHUNK_CHARS = 20000
OVERVIEW_TRANSCRIPT_CHARS = 100000
DEFAULT_SLIDE_BATCH_SIZE = 4
MAX_SLIDE_BATCH_SIZE = 8
VISION_HQ_PROFILE = "VISION_HQ"
SUMMARIZATION_HQ_PROFILE = "SUMMARIZATION_HQ"
LM_STUDIO_PROFILES = {
    VISION_HQ_PROFILE: {
        "load": {
            "context_length": 32768,
            "flash_attention": True,
            "offload_kv_cache_to_gpu": False,
            "echo_load_config": True,
        },
        "inference": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "seed": 42,
            "repeat_penalty": 1.0,
            "max_tokens": 250,
            "stop": ["\n\n"],
            "stream": False,
        },
    },
    SUMMARIZATION_HQ_PROFILE: {
        "load": {
            "context_length": 65536,
            "flash_attention": True,
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
}
JSON_ONLY_INSTRUCTION = (
    "Respond immediately with only the JSON object. Do not include planning, analysis, reasoning, "
    "or any text before or after the JSON."
)
DEFAULT_ROLE_MAX_TOKENS = {
    "overview": 8192,
    "transcript_cleanup": 8192,
    "slide_analysis": 4096,
    "resource_planner": 1024,
    "resource_formatter_direct": 2048,
    "resource_formatter_gathered": 2048,
    "resource_gather": 4096,
}
BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
DIRECT_RESOURCE_MAX_QUERIES = 4
DIRECT_RESOURCE_RESULTS_PER_QUERY = 6
DIRECT_RESOURCE_VERIFY_LIMIT = 10
DIRECT_RESOURCE_OUTPUT_LIMIT = 6
RESOURCE_PLANNER_QUERY_LIMIT = 6
LM_STUDIO_WEB_INTEGRATIONS = [
    {
        "type": "plugin",
        "id": "mcp/brave-search",
        "allowed_tools": ["brave_web_search", "brave_local_search"],
    },
    {
        "type": "plugin",
        "id": "mcp/fetch",
        "allowed_tools": ["fetch_url_content_tool"],
    },
]


class MLXTextProvider:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_TEXT_BASE_URL,
        model: str = DEFAULT_LOCAL_MODEL,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        role_max_tokens: Optional[Dict[str, int]] = None,
        disable_thinking: bool = False,
    ) -> None:
        self.model = model or DEFAULT_LOCAL_MODEL
        self.base_url = base_url or DEFAULT_TEXT_BASE_URL
        self.timeout_seconds = _bounded_timeout(timeout_seconds)
        self._role_max_tokens = {
            **DEFAULT_ROLE_MAX_TOKENS,
            **{key: int(value) for key, value in (role_max_tokens or {}).items()},
        }
        self._client = _OpenAICompatibleClient(
            self.base_url,
            self.model,
            self.timeout_seconds,
            profile_name=SUMMARIZATION_HQ_PROFILE,
            disable_thinking=disable_thinking,
        )

    def _max_tokens(self, role: str) -> int:
        return int(self._role_max_tokens.get(role) or DEFAULT_ROLE_MAX_TOKENS[role])

    @classmethod
    def info(cls) -> ProviderInfo:
        return ProviderInfo(
            name="mlx-text",
            display_name="LM Studio Text",
            available_models=[DEFAULT_LOCAL_MODEL],
            default_model=DEFAULT_LOCAL_MODEL,
            docs_url="",
        )

    def test_connection(self) -> None:
        self._client.chat_json(
            [{"role": "user", "content": "Reply with JSON: {\"ok\": true}"}],
            max_tokens=32,
            temperature=0.0,
            context="LM Studio text connection test",
        )

    def analyze_lecture(self, request: AnalyzeLectureRequest) -> AnalyzeLectureResponse:
        step_timings: Dict[str, Dict] = {}
        step_token_usage: Dict[str, Dict] = {}
        timer = _start_step_timer()
        overview = self.analyze_overview(request)
        step_timings["overview"] = _finish_step_timer(timer)
        step_token_usage["overview"] = _token_usage_entry_from_payload(overview)
        timer = _start_step_timer()
        transcript = self.analyze_transcript(request)
        step_timings["transcript_cleanup"] = _finish_step_timer(
            timer,
            chunk_count=int(transcript.get("chunk_count") or 0),
        )
        step_token_usage["transcript_cleanup"] = _token_usage_entry_from_payload(transcript)
        timer = _start_step_timer()
        slides = local_slide_analysis(request, "mlx-text")
        step_timings["slide_analysis"] = _finish_step_timer(timer, batch_count=0, retry_count=0)
        step_token_usage["slide_analysis"] = _token_usage_entry_from_payload(slides)
        timer = _start_step_timer()
        resources = self.analyze_resources(request, overview=overview)
        nested_timings = resources.get("step_timings") if isinstance(resources, dict) else None
        if isinstance(nested_timings, dict):
            step_timings.update(nested_timings)
        else:
            step_timings["resource_formatter"] = _finish_step_timer(timer)
        nested_token_usage = resources.get("step_token_usage") if isinstance(resources, dict) else None
        if isinstance(nested_token_usage, dict):
            step_token_usage.update(
                {str(key): value for key, value in nested_token_usage.items() if isinstance(value, dict)}
            )
        else:
            step_token_usage["resource_formatter"] = _token_usage_entry_from_payload(resources)
        warnings = []
        for payload in (overview, transcript, slides, resources):
            warnings.extend(payload.get("warnings") or [])
        return AnalyzeLectureResponse(
            title=overview["title"],
            executive_summary=overview["executive_summary"],
            outline=overview["outline"],
            formatted_transcript=transcript["formatted_transcript"],
            slide_analysis=slides["slide_analysis"],
            resources=resources["resources"],
            input_token_estimate=_sum_tokens("input_tokens", overview, transcript, slides, resources),
            output_token_estimate=_sum_tokens("output_tokens", overview, transcript, slides, resources),
            raw_response_id="lm-studio-text",
            warnings=_dedupe_warnings(warnings),
            step_timings=step_timings,
            step_token_usage={key: value for key, value in step_token_usage.items() if value.get("total_tokens", 0) > 0},
        )

    def analyze_overview(self, request: AnalyzeLectureRequest) -> Dict:
        prompt = f"""
{JSON_ONLY_INSTRUCTION}
Create a lecture overview with these fields:
- title: short, specific lecture title
- executive_summary: one concise paragraph
- outline: array of objects with id, heading, slide_ids
- warnings: array of strings

Lecture id: {request.lecture_id}
Duration minutes: {request.duration_minutes:.2f}

Transcript:
{_trim_text(request.transcript_text, OVERVIEW_TRANSCRIPT_CHARS)}

Slides:
{_slide_index_text(request.slides)}
""".strip()
        try:
            payload, usage = self._client.chat_json(
                _messages(prompt, system="You create concise study-note overviews from lecture transcripts."),
                max_tokens=self._max_tokens("overview"),
                temperature=0.2,
                context="MLX overview",
            )
            normalized = _normalize_overview(payload, request)
            _attach_usage(normalized, usage, prompt, payload)
            normalized.setdefault("warnings", []).append(f"Overview used local LM Studio text model: {self.model}")
            return normalized
        except Exception as exc:
            fallback = local_overview(request, self.model)
            fallback.setdefault("warnings", []).append(f"LM Studio overview fallback used: {exc}")
            return fallback

    def analyze_transcript(self, request: AnalyzeLectureRequest) -> Dict:
        chunks = _transcript_chunks(request.transcript_text)
        parts: List[str] = []
        warnings: List[str] = [f"Transcript editing used local LM Studio text model: {self.model}"]
        input_tokens = 0
        output_tokens = 0
        for index, chunk in enumerate(chunks, start=1):
            prompt = f"""
{JSON_ONLY_INSTRUCTION}
Return fields:
- formatted_transcript: polished transcript text with readable paragraph breaks
- warnings: array of strings, each noting an edit that changed wording beyond punctuation, capitalization, or spacing.

Format this lecture transcript chunk for a student reading it later.

Rules:
- Make light copy edits only: punctuation, capitalization, spacing, repeated words, and obvious transcription glitches.
- Add paragraph breaks every 2-5 sentences or whenever the topic shifts.
- Use blank lines between paragraphs so the result is not one large blob.
- Keep the instructor's voice and the original order of ideas.
- Preserve names, technical terms, examples, and substantive details.
- Do not summarize, add new ideas, or remove meaningful content.
- Never add headings. If the speaker introduces a section verbally, leave the title as plain text in the paragraph.

Chunk {index}/{len(chunks)}:
{chunk}
""".strip()
            try:
                payload, usage = self._client.chat_json(
                    _messages(prompt, system="You are a careful lecture transcript copy editor. Preserve meaning."),
                    max_tokens=self._max_tokens("transcript_cleanup"),
                    temperature=0.1,
                    context=f"MLX transcript chunk {index}",
                )
                formatted = str(payload.get("formatted_transcript") or chunk).strip()
                parts.append(formatted)
                warnings.extend(payload.get("warnings") or [])
                input_tokens += usage.input_tokens or _estimate_tokens(prompt)
                output_tokens += usage.output_tokens or _estimate_tokens(formatted)
            except Exception as exc:
                parts.append(chunk)
                warnings.append(f"Transcript chunk {index} used raw fallback: {exc}")
        return {
            "formatted_transcript": "\n\n".join(part for part in parts if part).strip()
            or local_formatted_transcript(request, self.model)["formatted_transcript"],
            "warnings": _dedupe_warnings(warnings),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "chunk_count": len(chunks),
        }

    def analyze_resources(self, request: AnalyzeLectureRequest, *, overview: Optional[Dict] = None) -> Dict:
        title = str((overview or {}).get("title") or request.lecture_id)
        summary = str((overview or {}).get("executive_summary") or "")
        outline_headings = _outline_heading_values(overview)
        step_timings: Dict[str, Dict] = {}
        step_token_usage: Dict[str, Dict] = {}
        timer = _start_step_timer()
        planned_queries, planner_warnings, planner_usage = self._plan_resource_queries(
            title=title,
            summary=summary,
            outline_headings=outline_headings,
        )
        step_timings["resource_planner"] = _finish_step_timer(
            timer,
            query_count=len(planned_queries),
        )
        step_token_usage["resource_planner"] = _token_usage_entry_from_usage(planner_usage)
        timer = _start_step_timer()
        direct_candidates, direct_warnings = _direct_resource_candidates(
            title=title,
            summary=summary,
            outline_headings=outline_headings,
            timeout_seconds=self.timeout_seconds,
            queries=planned_queries,
        )
        step_timings["resource_search"] = _finish_step_timer(
            timer,
            query_count=len(planned_queries or _resource_search_queries(title, summary, outline_headings)),
            candidate_count=len(direct_candidates),
        )
        step_token_usage["resource_search"] = _token_usage_entry(0, 0)
        resource_warnings = _dedupe_warnings(planner_warnings + direct_warnings)
        if direct_candidates:
            timer = _start_step_timer()
            result = self._format_direct_resource_candidates(
                title=title,
                summary=summary,
                outline_headings=outline_headings,
                candidates=direct_candidates,
                warnings=resource_warnings,
                planner_usage=planner_usage,
            )
            step_timings["resource_formatter"] = _finish_step_timer(
                timer,
                formatter="direct",
                candidate_count=len(direct_candidates),
            )
            result["step_timings"] = step_timings
            step_token_usage.update(result.get("step_token_usage") or {})
            result["step_token_usage"] = step_token_usage
            return result
        result = self._resources_via_lm_studio_tools(
            title=title,
            summary=summary,
            outline_headings=outline_headings,
            fallback_warnings=resource_warnings,
            queries=planned_queries,
        )
        if result.get("step_timings"):
            step_timings.update(result["step_timings"])
        result["step_timings"] = step_timings
        step_token_usage.update(result.get("step_token_usage") or {})
        result["step_token_usage"] = step_token_usage
        return result

    def _plan_resource_queries(
        self,
        *,
        title: str,
        summary: str,
        outline_headings: List[str],
    ) -> Tuple[List[str], List[str], "_Usage"]:
        prompt = f"""
{JSON_ONLY_INSTRUCTION}
Create focused web search queries for finding external study resources for this lecture.

Rules:
- Return 3 to 6 web-search-ready queries.
- Prefer queries that find official docs, university pages, peer-reviewed papers, reputable educational sources, or strong industry reports.
- Do not include URLs.
- Keep each query concise and specific.

Lecture title: {title}
Executive summary: {summary}
Outline headings:
{_outline_headings_text_from_values(outline_headings)}
""".strip()
        try:
            payload, usage = self._client.chat_structured_json(
                _messages(prompt, system="You plan high-quality resource searches for lecture study materials."),
                schema_name="resource_query_plan",
                schema=_resource_query_plan_schema(),
                max_tokens=self._max_tokens("resource_planner"),
                temperature=0.1,
                context="LM Studio resource query planner",
            )
            queries = _queries_from_plan(payload)[:RESOURCE_PLANNER_QUERY_LIMIT]
            warnings = list(payload.get("warnings") or [])
            if queries:
                warnings.append(f"Resource queries planned with LM Studio structured output using {self.model}.")
                return queries, _dedupe_warnings(warnings), usage
            warnings.append("Resource query planner returned no usable queries; used app fallback queries.")
            return [], _dedupe_warnings(warnings), usage
        except Exception as exc:
            return [], [f"Resource query planner failed; used app fallback queries: {exc}"], _Usage(
                input_tokens=_estimate_tokens(prompt),
                output_tokens=0,
            )

    def _format_direct_resource_candidates(
        self,
        *,
        title: str,
        summary: str,
        outline_headings: List[str],
        candidates: List[Dict],
        warnings: List[str],
        planner_usage: "_Usage",
    ) -> Dict:
        format_prompt = f"""
{JSON_ONLY_INSTRUCTION}
Return fields:
- resources: array of up to 6 objects with title, url, summary, source_quality
- warnings: array of strings

Rank and summarize these verified search candidates for a student.

Rules:
- Include only URLs from the candidate list.
- Keep URLs exactly as provided.
- Prefer official, university, peer-reviewed, or reputable educational sources.
- Each summary should explain why the resource helps with this lecture.
- source_quality:
  - "high" = official documentation, peer-reviewed papers, .edu/.gov domains, established textbooks
  - "medium" = reputable industry blogs, journalism from major outlets, well-cited tutorials

Lecture title: {title}
Executive summary: {summary}
Outline headings:
{_outline_headings_text_from_values(outline_headings)}

Verified resource candidates:
{json.dumps(candidates[:DIRECT_RESOURCE_VERIFY_LIMIT], indent=2)}
""".strip()
        try:
            payload, format_usage = self._client.chat_structured_json(
                _messages(format_prompt, system="You rank verified lecture study resources and return compact JSON only."),
                schema_name="lecture_resources",
                schema=_resource_formatter_schema(),
                max_tokens=self._max_tokens("resource_formatter_direct"),
                temperature=0.1,
                context="LM Studio direct resources JSON format",
            )
            normalized = _normalize_resources(payload)
            resources = _resources_matching_candidates(normalized.get("resources") or [], candidates)
            result_warnings = list(warnings) + list(normalized.get("warnings") or [])
            if len(resources) < min(DIRECT_RESOURCE_OUTPUT_LIMIT, len(candidates)):
                resources = _fill_resources_from_candidates(resources, candidates)
                result_warnings.append(
                    "Resource formatter returned too few usable links; filled remaining slots from verified search results."
                )
            result_warnings.append(
                f"Resources used direct Brave Search plus LM Studio structured formatting with {self.model}."
            )
            return {
                "resources": resources[:DIRECT_RESOURCE_OUTPUT_LIMIT],
                "warnings": _dedupe_warnings(result_warnings),
                "input_tokens": (planner_usage.input_tokens or 0)
                + (format_usage.input_tokens or _estimate_tokens(format_prompt)),
                "output_tokens": (planner_usage.output_tokens or 0)
                + (format_usage.output_tokens or _estimate_tokens(json.dumps(payload))),
                "step_token_usage": {
                    "resource_formatter": _token_usage_entry(
                        format_usage.input_tokens or _estimate_tokens(format_prompt),
                        format_usage.output_tokens or _estimate_tokens(json.dumps(payload)),
                    ),
                },
            }
        except Exception as exc:
            return {
                "resources": _resources_from_candidates(candidates),
                "warnings": _dedupe_warnings(
                    list(warnings)
                    + [
                        f"Resource formatting failed; saved verified search results instead: {exc}",
                        "Resources used direct Brave Search without LM Studio formatting.",
                    ]
                ),
                "input_tokens": (planner_usage.input_tokens or 0) + _estimate_tokens(format_prompt),
                "output_tokens": _estimate_tokens(json.dumps(candidates[:DIRECT_RESOURCE_OUTPUT_LIMIT])),
                "step_token_usage": {
                    "resource_formatter": _token_usage_entry(
                        _estimate_tokens(format_prompt),
                        _estimate_tokens(json.dumps(candidates[:DIRECT_RESOURCE_OUTPUT_LIMIT])),
                    ),
                },
            }

    def _resources_via_lm_studio_tools(
        self,
        *,
        title: str,
        summary: str,
        outline_headings: List[str],
        fallback_warnings: List[str],
        queries: Optional[List[str]] = None,
    ) -> Dict:
        outline_text = _outline_headings_text_from_values(outline_headings)
        planner_queries = _planner_queries_text(queries or _resource_search_queries(title, summary, outline_headings))
        step_timings: Dict[str, Dict] = {}
        gather_prompt = f"""
Use LM Studio's web-search tools to gather candidate study resources for this lecture.

Find 4-6 high-quality external resources that help a student go deeper on the lecture's main topics.

Rules:
- Prefer official documentation, university pages, peer-reviewed papers, reputable educational sources, or
  well-known industry reports.
- Use fetch when it helps confirm a search result.
- Only keep URLs you verified through the web tools. Do not invent or guess URLs.
- Return concise notes for each candidate with title, URL, why it is relevant, and source quality.
- Run these searches in order: {planner_queries}. Stop early once you have 4-6 verified candidates.
- This is a research-gathering pass. Do not return JSON yet.

Lecture title: {title}
Executive summary: {summary}
Outline headings:
{outline_text}
""".strip()
        try:
            gather_timer = _start_step_timer()
            gathered_context, gather_usage, tool_calls = self._client.chat_text_with_lm_studio_tools(
                gather_prompt,
                system=(
                    "You are a research assistant for lecture study resources. "
                    "Search the web, verify URLs, and gather concise source notes."
                ),
                max_tokens=self._max_tokens("resource_gather"),
                temperature=0.2,
                context="LM Studio web resources gather",
            )
            step_timings["resource_search"] = _finish_step_timer(
                gather_timer,
                query_count=len(_clean_resource_queries(queries)),
                tool_call_count=len(tool_calls),
            )
            if not tool_calls:
                extracted = _resources_from_gathered_context(gathered_context)
                if extracted:
                    return {
                        "resources": extracted,
                        "warnings": _dedupe_warnings(fallback_warnings + [
                            "LM Studio did not report web-search tool-call metadata; used returned resource links."
                        ]),
                        "input_tokens": gather_usage.input_tokens or _estimate_tokens(gather_prompt),
                        "output_tokens": gather_usage.output_tokens or _estimate_tokens(gathered_context),
                        "step_timings": step_timings,
                        "step_token_usage": {
                            "resource_search": _token_usage_entry(
                                gather_usage.input_tokens or _estimate_tokens(gather_prompt),
                                gather_usage.output_tokens or _estimate_tokens(gathered_context),
                            ),
                        },
                    }
                fallback = _empty_resources_payload(
                    fallback_warnings + [
                        "LM Studio resources skipped: no web-search tool call was reported. "
                        "Check LM Studio's MCP/web-search settings before trusting links."
                    ]
                )
                fallback["input_tokens"] = gather_usage.input_tokens or _estimate_tokens(gather_prompt)
                fallback["output_tokens"] = gather_usage.output_tokens or _estimate_tokens(gathered_context)
                fallback["step_timings"] = step_timings
                fallback["step_token_usage"] = {
                    "resource_search": _token_usage_entry(
                        gather_usage.input_tokens or _estimate_tokens(gather_prompt),
                        gather_usage.output_tokens or _estimate_tokens(gathered_context),
                    ),
                }
                return fallback

            format_prompt = f"""
{JSON_ONLY_INSTRUCTION}
Return fields:
- resources: array of up to 6 objects with title, url, summary, source_quality
- warnings: array of strings

Format the gathered web research into study resources.

Rules:
- Include only resources from the gathered context.
- Every resource must include title, url, summary, and source_quality.
- source_quality:
  - "high" = official documentation, peer-reviewed papers, .edu/.gov domains, established textbooks
  - "medium" = reputable industry blogs, journalism from major outlets, well-cited tutorials
- Do not invent or guess URLs.
- If the gathered context does not contain useful verified URLs, return an empty resources array and explain the problem in warnings.

Lecture title: {title}
Executive summary: {summary}
Outline headings:
{outline_text}

Gathered web context:
{_trim_text(gathered_context, 12000)}
""".strip()
            format_timer = _start_step_timer()
            payload, format_usage = self._client.chat_json_with_lm_studio_text(
                format_prompt,
                system="You format verified lecture resource notes as compact JSON only.",
                max_tokens=self._max_tokens("resource_formatter_gathered"),
                temperature=0.1,
                context="LM Studio web resources JSON format",
            )
            step_timings["resource_formatter"] = _finish_step_timer(
                format_timer,
                formatter="gathered",
            )
            normalized = _normalize_resources(payload)
            normalized["input_tokens"] = (
                gather_usage.input_tokens
                or _estimate_tokens(gather_prompt)
            ) + (
                format_usage.input_tokens
                or _estimate_tokens(format_prompt)
            )
            normalized["output_tokens"] = (
                gather_usage.output_tokens
                or _estimate_tokens(gathered_context)
            ) + (
                format_usage.output_tokens
                or _estimate_tokens(json.dumps(payload))
            )
            normalized["step_token_usage"] = {
                "resource_search": _token_usage_entry(
                    gather_usage.input_tokens or _estimate_tokens(gather_prompt),
                    gather_usage.output_tokens or _estimate_tokens(gathered_context),
                ),
                "resource_formatter": _token_usage_entry(
                    format_usage.input_tokens or _estimate_tokens(format_prompt),
                    format_usage.output_tokens or _estimate_tokens(json.dumps(payload)),
                ),
            }
            warnings = normalized.setdefault("warnings", [])
            warnings.append(
                "Resources used LM Studio web search with "
                f"{self.model}: {', '.join(_dedupe_warnings(tool_calls))}."
            )
            normalized["warnings"] = _dedupe_warnings(fallback_warnings + warnings)
            normalized["step_timings"] = step_timings
            return normalized
        except Exception as exc:
            if "resource_search" not in step_timings and "gather_timer" in locals():
                step_timings["resource_search"] = _finish_step_timer(
                    locals()["gather_timer"],
                    query_count=len(_clean_resource_queries(queries)),
                    tool_call_count=0,
                )
            extracted = _resources_from_gathered_context(locals().get("gathered_context", ""))
            if extracted:
                usage = locals().get("gather_usage") or _Usage()
                return {
                    "resources": extracted,
                    "warnings": _dedupe_warnings(fallback_warnings + [
                        "Resources JSON formatting failed; used verified LM Studio web-tool results instead.",
                        f"LM Studio resources fallback used: {exc}",
                    ]),
                    "input_tokens": usage.input_tokens or _estimate_tokens(gather_prompt),
                    "output_tokens": usage.output_tokens
                    or _estimate_tokens(str(locals().get("gathered_context", ""))),
                    "step_timings": step_timings,
                    "step_token_usage": {
                        "resource_search": _token_usage_entry(
                            usage.input_tokens or _estimate_tokens(gather_prompt),
                            usage.output_tokens
                            or _estimate_tokens(str(locals().get("gathered_context", ""))),
                        ),
                    },
                }
            fallback = _empty_resources_payload(fallback_warnings + [f"LM Studio resources fallback used: {exc}"])
            fallback["step_timings"] = step_timings
            fallback["step_token_usage"] = {}
            return fallback

    def analyze_slides(self, request: AnalyzeLectureRequest) -> Dict:
        fallback = local_slide_analysis(request, self.model)
        fallback.setdefault("warnings", []).append(
            "Slide images require the LM Studio Vision route; LM Studio Text used transcript-only slide notes."
        )
        return fallback


class MLXVisionProvider(MLXTextProvider):
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_VISION_BASE_URL,
        model: str = DEFAULT_LOCAL_MODEL,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        role_max_tokens: Optional[Dict[str, int]] = None,
        disable_thinking: bool = False,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            role_max_tokens=role_max_tokens,
            disable_thinking=disable_thinking,
        )
        self._client.profile_name = VISION_HQ_PROFILE

    @classmethod
    def info(cls) -> ProviderInfo:
        return ProviderInfo(
            name="mlx-vision",
            display_name="LM Studio Vision",
            available_models=[DEFAULT_LOCAL_MODEL],
            default_model=DEFAULT_LOCAL_MODEL,
            docs_url="",
        )

    def test_connection(self) -> None:
        self._client.chat_json(
            [{"role": "user", "content": "Reply with JSON: {\"ok\": true}"}],
            max_tokens=32,
            temperature=0.0,
            context="LM Studio vision connection test",
        )

    def analyze_slides(self, request: AnalyzeLectureRequest) -> Dict:
        if not request.slides:
            return local_slide_analysis(request, self.model)
        metadata_payload = _slide_analysis_from_extraction_metadata(request, self.model)
        if metadata_payload is not None:
            return metadata_payload
        batch_size = _configured_slide_batch_size()
        slide_analysis: List[Dict] = []
        warnings = [
            f"Slide analysis used local LM Studio vision model: {self.model}",
            f"Slide analysis used LM Studio batch size {batch_size}.",
        ]
        input_tokens = 0
        output_tokens = 0
        batch_count = 0
        retry_count = 0
        for batch in _slide_batches(request.slides, batch_size):
            batch_count += 1
            batch_ids = _slide_ids(batch)
            raw_items: List[Dict] = []
            try:
                batch_items, batch_warnings, batch_input, batch_output = self._request_slide_batch(
                    request,
                    batch,
                    context="MLX slide batch",
                    temperature=0.3,
                )
                raw_items.extend(batch_items)
                warnings.extend(batch_warnings)
                input_tokens += batch_input
                output_tokens += batch_output
                missing_ids = _missing_slide_ids(raw_items, batch)
                if missing_ids:
                    retry_count += 1
                    missing_slides = _slides_with_ids(batch, missing_ids)
                    warnings.append(
                        "Slide batch missed slide id(s) "
                        f"{', '.join(map(str, missing_ids))}; retrying those slides."
                    )
                    try:
                        retry_items, retry_warnings, retry_input, retry_output = self._request_slide_batch(
                            request,
                            missing_slides,
                            context="MLX slide missing retry",
                            temperature=0.2,
                        )
                        raw_items.extend(retry_items)
                        warnings.extend(retry_warnings)
                        input_tokens += retry_input
                        output_tokens += retry_output
                    except Exception as retry_exc:
                        warnings.append(
                            f"Slide retry skipped for missing slide id(s) {', '.join(map(str, missing_ids))}: {retry_exc}"
                        )
                still_missing = _missing_slide_ids(raw_items, batch)
                if still_missing and len(batch) > 1:
                    for slide in _slides_with_ids(batch, still_missing):
                        slide_id = _slide_id(slide, fallback=1)
                        try:
                            retry_count += 1
                            solo_items, solo_warnings, solo_input, solo_output = self._request_slide_batch(
                                request,
                                [slide],
                                context=f"MLX slide solo retry {slide_id}",
                                temperature=0.1,
                            )
                            raw_items.extend(solo_items)
                            warnings.extend(solo_warnings)
                            input_tokens += solo_input
                            output_tokens += solo_output
                        except Exception as solo_exc:
                            warnings.append(f"Slide {slide_id} solo retry skipped: {solo_exc}")
                    still_missing = _missing_slide_ids(raw_items, batch)
                if still_missing:
                    warnings.append(
                        "Slide analysis missing slide id(s) after retries: "
                        f"{', '.join(map(str, still_missing))}. Local fallback filled those slides."
                    )
                normalized = _normalize_slide_analysis(raw_items, request, batch)
                slide_analysis.extend(normalized)
            except Exception as exc:
                retry_count += len(batch)
                warnings.append(
                    f"Slide batch {', '.join(map(str, batch_ids))} failed; retrying each slide separately: {exc}"
                )
                raw_items, fallback_warnings, fallback_input, fallback_output = self._retry_or_fallback_slide_batch(
                    request,
                    batch,
                )
                slide_analysis.extend(_normalize_slide_analysis(raw_items, request, batch))
                warnings.extend(fallback_warnings)
                input_tokens += fallback_input
                output_tokens += fallback_output
        return {
            "slide_analysis": slide_analysis or local_slide_analysis(request, self.model)["slide_analysis"],
            "warnings": _dedupe_warnings(warnings),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "batch_count": batch_count,
            "retry_count": retry_count,
        }

    def _request_slide_batch(
        self,
        request: AnalyzeLectureRequest,
        batch: List[Dict],
        *,
        context: str,
        temperature: float,
    ) -> Tuple[List[Dict], List[str], int, int]:
        prompt = _slide_batch_prompt(request, batch)
        payload, usage = self._client.chat_json_with_lm_studio_images(
            _vision_input_items(request, batch, prompt),
            system="You analyze lecture slide images and return compact JSON.",
            max_tokens=self._max_tokens("slide_analysis"),
            temperature=temperature,
            context=context,
        )
        wanted_ids = set(_slide_ids(batch))
        raw_items = []
        for item in (payload.get("slide_analysis") or []):
            if not isinstance(item, dict):
                continue
            try:
                slide_id = int(item.get("slide_id") or 0)
            except (TypeError, ValueError):
                continue
            if slide_id in wanted_ids:
                raw_items.append(item)
        return (
            raw_items,
            list(payload.get("warnings") or []),
            usage.input_tokens or _estimate_tokens(prompt),
            usage.output_tokens or _estimate_tokens(json.dumps(payload)),
        )

    def _retry_or_fallback_slide_batch(
        self,
        request: AnalyzeLectureRequest,
        batch: List[Dict],
    ) -> Tuple[List[Dict], List[str], int, int]:
        raw_items: List[Dict] = []
        warnings: List[str] = []
        input_tokens = 0
        output_tokens = 0
        for index, slide in enumerate(batch, start=1):
            slide_id = _slide_id(slide, fallback=index)
            try:
                solo_items, solo_warnings, solo_input, solo_output = self._request_slide_batch(
                    request,
                    [slide],
                    context=f"MLX slide solo fallback {slide_id}",
                    temperature=0.1,
                )
                raw_items.extend(solo_items)
                warnings.extend(solo_warnings)
                input_tokens += solo_input
                output_tokens += solo_output
                if slide_id in _missing_slide_ids(solo_items, [slide]):
                    warnings.append(f"Slide {slide_id} solo retry returned no usable slide row; local fallback filled it.")
            except Exception as exc:
                warnings.append(f"Slide {slide_id} used local fallback after solo retry failed: {exc}")
        missing_ids = _missing_slide_ids(raw_items, batch)
        if missing_ids:
            warnings.append(
                "Slide analysis local fallback filled slide id(s): "
                f"{', '.join(map(str, missing_ids))}."
            )
        return raw_items, warnings, input_tokens, output_tokens


@dataclass(frozen=True)
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class _LoadedModelInstance:
    model_id: str
    instance_id: str


class _OpenAICompatibleClient:
    _loaded_instance_cache: Dict[str, Dict[str, str]] = {}
    _model_verification_cache: Dict[Tuple[str, str], Tuple[str, float]] = {}
    _model_verification_ttl_seconds = 30.0
    _lm_studio_lock = threading.RLock()
    _load_config_support_cache: Dict[str, bool] = {}
    _native_max_token_field = "max_tokens"
    _native_max_token_field_logged = False
    _chat_template_kwargs_warning_logged = False

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int,
        *,
        profile_name: str = SUMMARIZATION_HQ_PROFILE,
        disable_thinking: bool = False,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.model = model or DEFAULT_LOCAL_MODEL
        self.timeout_seconds = _bounded_timeout(timeout_seconds)
        self.profile_name = _normalize_lm_studio_profile(profile_name)
        self.disable_thinking = bool(disable_thinking)
        self._resolved_model: Optional[str] = None

    @classmethod
    @contextmanager
    def _exclusive_lm_studio_request(cls):
        with cls._lm_studio_lock:
            lock_handle = None
            try:
                if fcntl is not None:
                    lock_path = Path(tempfile.gettempdir()) / "lecture-processor-lm-studio.lock"
                    lock_handle = lock_path.open("a+")
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                if lock_handle is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                    lock_handle.close()

    @classmethod
    def clear_model_verification_cache(cls) -> None:
        with cls._lm_studio_lock:
            cls._model_verification_cache = {}

    @classmethod
    def _set_native_max_token_field(cls, field_name: str) -> None:
        cls._native_max_token_field = field_name
        cls._native_max_token_field_logged = False

    @classmethod
    def _log_native_max_token_field_once(cls) -> None:
        if cls._native_max_token_field_logged:
            return
        print(
            f"LM Studio native chat output-token field: {cls._native_max_token_field}",
            file=sys.stderr,
        )
        cls._native_max_token_field_logged = True

    @classmethod
    def _warn_chat_template_kwargs_once(cls, context: str) -> None:
        if cls._chat_template_kwargs_warning_logged:
            return
        print(
            f"{context}: LM Studio rejected chat_template_kwargs; continuing with prompt-level no-think instructions.",
            file=sys.stderr,
        )
        cls._chat_template_kwargs_warning_logged = True

    def _add_native_max_tokens(self, body: Dict, max_tokens: int) -> None:
        for field_name in _native_max_token_field_names():
            body.pop(field_name, None)
        body[self._native_max_token_field] = int(max_tokens)

    def _add_chat_template_kwargs(self, body: Dict) -> None:
        if self.disable_thinking and not _is_ollama_base_url(self.base_url):
            body["chat_template_kwargs"] = {"enable_thinking": False}

    def _add_profile_inference(
        self,
        body: Dict,
        max_tokens: int,
        temperature: float,
        *,
        native_tokens: bool = False,
    ) -> None:
        if _uses_openai_only_api(self.base_url):
            body["max_tokens"] = int(max_tokens)
            body["temperature"] = float(temperature)
            return
        profile = LM_STUDIO_PROFILES[_normalize_lm_studio_profile(self.profile_name)]
        inference = dict(profile["inference"])
        inference["max_tokens"] = int(max_tokens)
        inference["temperature"] = float(temperature)
        body.update(inference)
        if native_tokens and "max_tokens" in body:
            self._add_native_max_tokens(body, int(body.pop("max_tokens")))

    def _profile_load_config(self) -> Dict:
        profile = LM_STUDIO_PROFILES[_normalize_lm_studio_profile(self.profile_name)]
        return dict(profile["load"])

    def chat_json(self, messages: List[Dict], *, max_tokens: int, temperature: float, context: str) -> Tuple[Dict, _Usage]:
        text, usage = self.chat_text(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            context=context,
        )
        try:
            return _json_from_response_text(text), usage
        except ProviderResponseError as exc:
            raise ProviderResponseError(f"{context} returned invalid JSON from LM Studio: {exc}") from exc

    def chat_json_with_lm_studio_tools(
        self,
        prompt: str,
        *,
        system: str,
        max_tokens: int,
        temperature: float,
        context: str,
    ) -> Tuple[Dict, _Usage, List[str]]:
        text, usage, tool_calls = self.chat_text_with_lm_studio_tools(
            prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            context=context,
        )
        try:
            return _json_from_response_text(text), usage, tool_calls
        except ProviderResponseError as exc:
            raise ProviderResponseError(f"{context} returned invalid JSON from LM Studio: {exc}") from exc

    def chat_json_with_lm_studio_text(
        self,
        prompt: str,
        *,
        system: str,
        max_tokens: int,
        temperature: float,
        context: str,
    ) -> Tuple[Dict, _Usage]:
        text, usage, _tool_calls = self.chat_text_with_lm_studio_native(
            prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            context=context,
        )
        try:
            return _json_from_response_text(text), usage
        except ProviderResponseError as exc:
            raise ProviderResponseError(f"{context} returned invalid JSON from LM Studio: {exc}") from exc

    def chat_structured_json(
        self,
        messages: List[Dict],
        *,
        schema_name: str,
        schema: Dict,
        max_tokens: int,
        temperature: float,
        context: str,
    ) -> Tuple[Dict, _Usage]:
        if _is_ollama_base_url(self.base_url):
            raise ProviderRequestError(
                f"{context} requires LM Studio structured output. "
                "Update the LM Studio server URL in Settings."
            )
        with self._exclusive_lm_studio_request():
            body = {
                "model": self._chat_model(context),
                "messages": messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
                "stream": False,
            }
            self._add_profile_inference(body, max_tokens, temperature)
            self._add_chat_template_kwargs(body)
            return self._post_lm_studio_structured_chat(body, context=context)

    def chat_json_with_lm_studio_images(
        self,
        input_items: List[Dict],
        *,
        system: str,
        max_tokens: int,
        temperature: float,
        context: str,
    ) -> Tuple[Dict, _Usage]:
        text, usage = self.chat_text_with_lm_studio_images(
            input_items,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            context=context,
        )
        try:
            return _json_from_response_text(text), usage
        except ProviderResponseError as exc:
            raise ProviderResponseError(f"{context} returned invalid JSON from LM Studio: {exc}") from exc

    def chat_text(self, messages: List[Dict], *, max_tokens: int, temperature: float, context: str) -> Tuple[str, _Usage]:
        if _uses_lm_studio_native_api(self.base_url):
            with self._exclusive_lm_studio_request():
                system_prompt, input_text = _native_prompt_from_messages(messages)
                body = {
                    "model": self._chat_model(context),
                    "input": input_text,
                    "store": False,
                }
                self._add_profile_inference(body, max_tokens, temperature, native_tokens=True)
                self._add_chat_template_kwargs(body)
                if system_prompt:
                    body["system_prompt"] = system_prompt
                text, usage, _tool_calls = self._post_lm_studio_native_chat(body, context=context)
                return text, usage

        body = {
            "model": self._chat_model(context),
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if _is_ollama_base_url(self.base_url):
            # Ollama thinking models can spend the full token budget in hidden reasoning.
            # Disabling thinking gives the app the visible JSON payload it needs.
            body["think"] = False
            body["stream"] = False
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=_request_headers(base_url=self.base_url, content_type="application/json"),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            raise ProviderRequestError(f"{context} failed at {self.base_url}: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderRequestError(
                f"{context} could not reach LM Studio at {self.base_url}. "
                "Start the local server or update the LM Studio server URL in Settings."
            ) from exc
        except TimeoutError as exc:
            raise ProviderTransientError(f"{context} timed out after {self.timeout_seconds}s at {self.base_url}") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(f"{context} returned non-JSON HTTP response from LM Studio.") from exc
        choices = payload.get("choices") or []
        if not choices:
            raise ProviderResponseError(f"{context} returned no choices from LM Studio.")
        message = choices[0].get("message") or {}
        text = str(message.get("content") or "").strip()
        if not text:
            raise ProviderResponseError(f"{context} returned an empty response from LM Studio.")
        usage = payload.get("usage") or {}
        return text, _Usage(
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
        )

    def chat_text_with_lm_studio_tools(
        self,
        prompt: str,
        *,
        system: str,
        max_tokens: int,
        temperature: float,
        context: str,
    ) -> Tuple[str, _Usage, List[str]]:
        if _is_ollama_base_url(self.base_url):
            raise ProviderRequestError(
                f"{context} requires LM Studio's tool-enabled API. "
                "Update the LM Studio server URL in Settings."
            )
        with self._exclusive_lm_studio_request():
            body = {
                "model": self._chat_model(context),
                "system_prompt": _system_prompt_no_think(system),
                "input": prompt,
                "integrations": LM_STUDIO_WEB_INTEGRATIONS,
                "store": False,
            }
            self._add_profile_inference(body, max_tokens, temperature, native_tokens=True)
            self._add_chat_template_kwargs(body)
            return self._post_lm_studio_native_chat(body, context=context, allow_tool_output_text=True)

    def chat_text_with_lm_studio_native(
        self,
        prompt: str,
        *,
        system: str,
        max_tokens: int,
        temperature: float,
        context: str,
    ) -> Tuple[str, _Usage, List[str]]:
        if _is_ollama_base_url(self.base_url):
            raise ProviderRequestError(
                f"{context} requires LM Studio's native chat API. "
                "Update the LM Studio server URL in Settings."
            )
        with self._exclusive_lm_studio_request():
            body = {
                "model": self._chat_model(context),
                "system_prompt": _system_prompt_no_think(system),
                "input": prompt,
                "store": False,
            }
            self._add_profile_inference(body, max_tokens, temperature, native_tokens=True)
            self._add_chat_template_kwargs(body)
            return self._post_lm_studio_native_chat(body, context=context)

    def chat_text_with_lm_studio_images(
        self,
        input_items: List[Dict],
        *,
        system: str,
        max_tokens: int,
        temperature: float,
        context: str,
    ) -> Tuple[str, _Usage]:
        if _is_ollama_base_url(self.base_url):
            raise ProviderRequestError(
                f"{context} requires LM Studio's native image API. "
                "Update the LM Studio server URL in Settings."
            )
        if _uses_openai_only_api(self.base_url):
            return self.chat_text_with_openai_images(
                _openai_image_messages(input_items, system=system),
                max_tokens=max_tokens,
                temperature=temperature,
                context=context,
            )
        with self._exclusive_lm_studio_request():
            body = {
                "model": self._chat_model(context),
                "system_prompt": _system_prompt_no_think(system),
                "input": input_items,
                "store": False,
            }
            self._add_profile_inference(body, max_tokens, temperature, native_tokens=True)
            self._add_chat_template_kwargs(body)
            text, usage, _tool_calls = self._post_lm_studio_native_chat(body, context=context)
            return text, usage

    def chat_text_with_openai_images(
        self,
        messages: List[Dict],
        *,
        max_tokens: int,
        temperature: float,
        context: str,
    ) -> Tuple[str, _Usage]:
        if _is_ollama_base_url(self.base_url):
            raise ProviderRequestError(
                f"{context} requires LM Studio's OpenAI-compatible image API. "
                "Update the LM Studio server URL in Settings."
            )
        with self._exclusive_lm_studio_request():
            self._chat_model(context)
            model_for_request = self.model if self.model != DEFAULT_LOCAL_MODEL else (self._resolved_model or self.model)
            body = {
                "model": model_for_request,
                "messages": messages,
            }
            self._add_profile_inference(body, max_tokens, temperature)
            self._add_chat_template_kwargs(body)
            request = urllib.request.Request(
                f"{_lm_studio_openai_base_url(self.base_url)}/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers=_request_headers(base_url=self.base_url, content_type="application/json"),
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                self.clear_model_verification_cache()
                detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
                if body.get("chat_template_kwargs") and _is_chat_template_kwargs_error(detail):
                    self._warn_chat_template_kwargs_once(context)
                    retry_body = dict(body)
                    retry_body.pop("chat_template_kwargs", None)
                    return self._post_openai_chat_completions(retry_body, context=context)
                if "invalid_api_key" in detail or "API token is required" in detail:
                    raise ProviderRequestError(
                        f"{context} needs a valid LM Studio API token. "
                        "Save the token in Settings, then refresh models and rerun."
                    ) from exc
                raise ProviderRequestError(f"{context} failed at {self.base_url}: HTTP {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                self.clear_model_verification_cache()
                raise ProviderRequestError(
                    f"{context} could not reach LM Studio at {_lm_studio_openai_base_url(self.base_url)}. "
                    "Start the local server or update the LM Studio server URL in Settings."
                ) from exc
            except TimeoutError as exc:
                self.clear_model_verification_cache()
                raise ProviderTransientError(
                    f"{context} timed out after {self.timeout_seconds}s at {_lm_studio_openai_base_url(self.base_url)}"
                ) from exc
        return _openai_chat_text_usage(raw, context=context)

    def _post_openai_chat_completions(self, body: Dict, *, context: str) -> Tuple[str, _Usage]:
        request = urllib.request.Request(
            f"{_lm_studio_openai_base_url(self.base_url)}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=_request_headers(base_url=self.base_url, content_type="application/json"),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            self.clear_model_verification_cache()
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            raise ProviderRequestError(f"{context} failed at {self.base_url}: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            self.clear_model_verification_cache()
            raise ProviderRequestError(
                f"{context} could not reach LM Studio at {_lm_studio_openai_base_url(self.base_url)}. "
                "Start the local server or update the LM Studio server URL in Settings."
            ) from exc
        except TimeoutError as exc:
            self.clear_model_verification_cache()
            raise ProviderTransientError(
                f"{context} timed out after {self.timeout_seconds}s at {_lm_studio_openai_base_url(self.base_url)}"
            ) from exc
        return _openai_chat_text_usage(raw, context=context)

    def _chat_model(self, context: str) -> str:
        if not _uses_lm_studio_native_api(self.base_url):
            if self.model and self.model != DEFAULT_LOCAL_MODEL:
                return self.model
            if self._resolved_model:
                return self._resolved_model
            models = _model_ids_from_payload(self._models_payload(context))
            if not models:
                raise ProviderRequestError(
                    f"{context} could not find any models at {self.base_url}. "
                    "Load a model and refresh the model list in Settings."
                )
            self._resolved_model = models[0]
            return self._resolved_model
        if self.model and self.model != DEFAULT_LOCAL_MODEL:
            return self._loaded_chat_model(self.model, context)
        if self._resolved_model:
            return self._loaded_chat_model(self._resolved_model, context)
        payload = self._models_payload(context)
        self._remember_loaded_instances(payload)
        models = _model_ids_from_payload(payload)
        if not models:
            raise ProviderRequestError(
                f"{context} could not find any LM Studio models at {self.base_url}. "
                "Load a model in LM Studio and refresh the model list in Settings."
            )
        self._resolved_model = models[0]
        return self._loaded_chat_model(self._resolved_model, context, known_payload=payload)

    def list_models(self, context: str = "LM Studio model lookup") -> List[str]:
        payload = self._models_payload(context)
        self._remember_loaded_instances(payload)
        return _model_ids_from_payload(payload)

    def _models_payload(self, context: str) -> Dict:
        request = urllib.request.Request(
            _model_list_url(self.base_url),
            headers=_request_headers(base_url=self.base_url),
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            raise ProviderRequestError(f"{context} failed at {self.base_url}: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderRequestError(
                f"{context} could not reach LM Studio at {self.base_url}. "
                "Start LM Studio's local server or update the server URL in Settings."
            ) from exc
        except TimeoutError as exc:
            raise ProviderTransientError(f"{context} timed out after {self.timeout_seconds}s at {self.base_url}") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(f"{context} returned non-JSON model list from LM Studio.") from exc
        return payload

    def _loaded_chat_model(self, model: str, context: str, *, known_payload: Optional[Dict] = None) -> str:
        if not _uses_lm_studio_native_api(self.base_url):
            return model

        if known_payload is None:
            cached_instance_id = self._cached_verified_model_instance(model)
            if cached_instance_id:
                return cached_instance_id

        payload = known_payload or self._models_payload(f"{context} model load check")
        self._remember_loaded_instances(payload)
        if not _payload_has_loaded_instance_state(payload):
            raise ProviderRequestError(
                f"{context} blocked because LM Studio did not report loaded model instances. "
                "Update LM Studio or enable the native /api/v1/models endpoint so the app can enforce one loaded model."
            )

        already_exclusive_instance_id = _single_loaded_matching_instance_id(payload, model)
        if already_exclusive_instance_id:
            self._remember_verified_model_instance(model, already_exclusive_instance_id)
            return already_exclusive_instance_id

        loaded_instance_id = self._unload_other_models(model, context, known_payload=payload)
        if not loaded_instance_id:
            loaded_instance_id = self._load_model(model, context, clear_existing=False)
        verified_instance_id = self._verify_single_loaded_model(model, loaded_instance_id, context)
        self._remember_verified_model_instance(model, verified_instance_id)
        return verified_instance_id

    def unload_model(self, model: str, context: str = "LM Studio model unload") -> bool:
        if _is_ollama_base_url(self.base_url):
            return False
        with self._exclusive_lm_studio_request():
            instance_id = self._loaded_instance_id(model, context)
            if not instance_id:
                return False
            return self._unload_instance(instance_id, model, context)

    def _unload_instance(self, instance_id: str, model_label: str, context: str) -> bool:
        if _is_ollama_base_url(self.base_url):
            return False
        self.clear_model_verification_cache()
        native_base_url = _lm_studio_native_base_url(self.base_url)
        request = urllib.request.Request(
            f"{native_base_url}/models/unload",
            data=json.dumps({"instance_id": instance_id}).encode("utf-8"),
            headers=_request_headers(base_url=self.base_url, content_type="application/json"),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            raise ProviderRequestError(
                f"{context} could not unload LM Studio model {model_label}: HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderRequestError(
                f"{context} could not reach LM Studio at {native_base_url} to unload {model_label}."
            ) from exc
        except TimeoutError as exc:
            raise ProviderTransientError(
                f"{context} timed out after {self.timeout_seconds}s unloading LM Studio model {model_label}"
            ) from exc
        self._forget_loaded_instance(native_base_url, model_label, instance_id)
        self.clear_model_verification_cache()
        return True

    def _loaded_instance_id(self, model: str, context: str) -> str:
        native_base_url = _lm_studio_native_base_url(self.base_url)
        cached = self._loaded_instance_cache.get(native_base_url, {})
        instance_id = str(cached.get(model) or "").strip()
        if instance_id:
            return instance_id
        payload = self._models_payload(f"{context} model unload check")
        self._remember_loaded_instances(payload)
        cached = self._loaded_instance_cache.get(native_base_url, {})
        return str(cached.get(model) or "").strip()

    def _forget_loaded_instance(self, native_base_url: str, model: str, instance_id: str) -> None:
        cached = dict(self._loaded_instance_cache.get(native_base_url) or {})
        for key, value in list(cached.items()):
            if key == model or key == instance_id or value == instance_id:
                cached.pop(key, None)
        self._loaded_instance_cache[native_base_url] = cached

    def _remember_loaded_instances(self, payload: Dict) -> None:
        if not _payload_has_loaded_instance_state(payload):
            return
        native_base_url = _lm_studio_native_base_url(self.base_url)
        loaded = _loaded_model_instances_from_payload(payload)
        cached = dict(self._loaded_instance_cache.get(native_base_url) or {})
        if loaded:
            cached.update(loaded)
        else:
            cached = {}
        self._loaded_instance_cache[native_base_url] = cached

    def _load_model(
        self,
        model: str,
        context: str,
        *,
        known_payload: Optional[Dict] = None,
        clear_existing: bool = True,
    ) -> str:
        if clear_existing:
            self._unload_other_models(model, context, known_payload=known_payload)
        self.clear_model_verification_cache()
        native_base_url = _lm_studio_native_base_url(self.base_url)
        load_config = self._profile_load_config()
        include_config = self._load_config_support_cache.get(native_base_url, True)
        raw, used_load_config = self._post_model_load(
            native_base_url,
            model,
            context,
            load_config=load_config,
            include_config=include_config,
        )

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(f"{context} returned non-JSON model load response from LM Studio.") from exc
        if used_load_config:
            _validate_lm_studio_load_config(self.profile_name, payload, context)
        instance_id = str(payload.get("instance_id") or model).strip() or model
        cached = dict(self._loaded_instance_cache.get(native_base_url) or {})
        cached[model] = instance_id
        cached[instance_id] = instance_id
        self._loaded_instance_cache[native_base_url] = cached
        return instance_id

    def _post_model_load(
        self,
        native_base_url: str,
        model: str,
        context: str,
        *,
        load_config: Dict,
        include_config: bool,
    ) -> Tuple[str, bool]:
        body = {"model": model}
        if include_config and load_config:
            body["config"] = load_config
        request = urllib.request.Request(
            f"{native_base_url}/models/load",
            data=json.dumps(body).encode("utf-8"),
            headers=_request_headers(base_url=self.base_url, content_type="application/json"),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            if include_config and _lm_studio_rejected_load_config(detail):
                # Older/newer LM Studio builds differ on whether /models/load accepts
                # a config object. Keep the run moving and apply profile settings at
                # inference time when load-time config is not supported.
                self._load_config_support_cache[native_base_url] = False
                return self._post_model_load(
                    native_base_url,
                    model,
                    context,
                    load_config=load_config,
                    include_config=False,
                )
            raise ProviderRequestError(f"{context} could not load LM Studio model {model}: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderRequestError(
                f"{context} could not reach LM Studio at {native_base_url} to load {model}."
            ) from exc
        except TimeoutError as exc:
            raise ProviderTransientError(
                f"{context} timed out after {self.timeout_seconds}s loading LM Studio model {model}"
            ) from exc
        self._load_config_support_cache[native_base_url] = include_config
        return raw, include_config

    def _verify_single_loaded_model(self, model: str, instance_id: str, context: str) -> str:
        payload = self._models_payload(f"{context} exclusive-model verify")
        self._remember_loaded_instances(payload)
        if not _payload_has_loaded_instance_state(payload):
            return instance_id or model

        rows = _loaded_model_instance_rows_from_payload(payload)
        if not rows:
            instance_id = self._load_model(model, context, clear_existing=False)
            payload = self._models_payload(f"{context} exclusive-model load verify")
            self._remember_loaded_instances(payload)
            rows = _loaded_model_instance_rows_from_payload(payload)

        if len(rows) != 1 or not _loaded_instance_matches_model(rows[0], model):
            self._unload_other_models(model, f"{context} exclusive-model repair", known_payload=payload)
            payload = self._models_payload(f"{context} exclusive-model repair verify")
            self._remember_loaded_instances(payload)
            rows = _loaded_model_instance_rows_from_payload(payload)

        if len(rows) == 1 and _loaded_instance_matches_model(rows[0], model):
            self._remember_verified_model_instance(model, rows[0].instance_id)
            return rows[0].instance_id

        loaded_labels = ", ".join(row.instance_id for row in rows) or "none"
        raise ProviderRequestError(
            f"{context} blocked because LM Studio does not have exactly one loaded model. "
            f"Loaded instances: {loaded_labels}. Unload extra models in LM Studio and rerun."
        )

    def _cached_verified_model_instance(self, model: str) -> str:
        key = self._model_verification_cache_key(model)
        instance_id, verified_at = self._model_verification_cache.get(key, ("", 0.0))
        if not instance_id:
            return ""
        if time.monotonic() - verified_at <= self._model_verification_ttl_seconds:
            return instance_id
        self._model_verification_cache.pop(key, None)
        return ""

    def _remember_verified_model_instance(self, model: str, instance_id: str) -> None:
        normalized_instance_id = str(instance_id or "").strip()
        if not normalized_instance_id:
            return
        self._model_verification_cache[self._model_verification_cache_key(model)] = (
            normalized_instance_id,
            time.monotonic(),
        )

    def _model_verification_cache_key(self, model: str) -> Tuple[str, str]:
        return (_lm_studio_native_base_url(self.base_url), str(model or "").strip())

    def _unload_other_models(
        self,
        keep_model: str,
        context: str,
        *,
        known_payload: Optional[Dict] = None,
    ) -> str:
        if known_payload is not None and _payload_has_loaded_instance_state(known_payload):
            payload = known_payload
        else:
            payload = self._models_payload(f"{context} competing-model scan")
        self._remember_loaded_instances(payload)
        loaded_rows = _loaded_model_instance_rows_from_payload(payload)
        kept_instance_id = _preferred_loaded_instance_id(loaded_rows, keep_model)
        unloaded_any = False
        for loaded in loaded_rows:
            matches_keep = _loaded_instance_matches_model(loaded, keep_model)
            if matches_keep and loaded.instance_id == kept_instance_id:
                continue
            try:
                unloaded = self._unload_instance(
                    loaded.instance_id,
                    loaded.model_id or loaded.instance_id,
                    f"{context} competing model unload",
                )
            except Exception as exc:
                payload = self._models_payload(f"{context} competing-model refresh")
                self._remember_loaded_instances(payload)
                remaining = _extra_loaded_instances(payload, keep_model)
                if not any(instance.instance_id == loaded.instance_id for instance in remaining):
                    continue
                raise ProviderRequestError(
                    f"{context} could not unload competing LM Studio model "
                    f"{loaded.model_id or loaded.instance_id} before loading {keep_model}: {exc}"
                ) from exc
            if unloaded:
                unloaded_any = True
                continue
            payload = self._models_payload(f"{context} competing-model refresh")
            self._remember_loaded_instances(payload)
            remaining = _extra_loaded_instances(payload, keep_model)
            if any(instance.instance_id == loaded.instance_id for instance in remaining):
                raise ProviderRequestError(
                    f"{context} could not clear competing LM Studio model "
                    f"{loaded.model_id or loaded.instance_id} before loading {keep_model}."
                )
        if unloaded_any:
            time.sleep(2)
        return kept_instance_id

    def _post_lm_studio_native_chat(
        self,
        body: Dict,
        *,
        context: str,
        allow_tool_output_text: bool = False,
        retry_without_reasoning: bool = True,
        unsupported_field_retries: int = 3,
    ) -> Tuple[str, _Usage, List[str]]:
        native_base_url = _lm_studio_native_base_url(self.base_url)
        request = urllib.request.Request(
            f"{native_base_url}/chat",
            data=json.dumps(body).encode("utf-8"),
            headers=_request_headers(base_url=self.base_url, content_type="application/json"),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            self.clear_model_verification_cache()
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            if retry_without_reasoning and body.get("reasoning") and _is_reasoning_setting_error(detail):
                retry_body = dict(body)
                retry_body.pop("reasoning", None)
                return self._post_lm_studio_native_chat(
                    retry_body,
                    context=context,
                    allow_tool_output_text=allow_tool_output_text,
                    retry_without_reasoning=False,
                    unsupported_field_retries=unsupported_field_retries,
                )
            if body.get("chat_template_kwargs") and _is_chat_template_kwargs_error(detail):
                self._warn_chat_template_kwargs_once(context)
                retry_body = dict(body)
                retry_body.pop("chat_template_kwargs", None)
                return self._post_lm_studio_native_chat(
                    retry_body,
                    context=context,
                    allow_tool_output_text=allow_tool_output_text,
                    retry_without_reasoning=retry_without_reasoning,
                    unsupported_field_retries=unsupported_field_retries,
                )
            retry_body = self._native_max_tokens_retry_body(body, detail)
            if retry_body is not None:
                return self._post_lm_studio_native_chat(
                    retry_body,
                    context=context,
                    allow_tool_output_text=allow_tool_output_text,
                    retry_without_reasoning=retry_without_reasoning,
                    unsupported_field_retries=unsupported_field_retries,
                )
            retry_body = _unsupported_fields_retry_body(body, detail)
            if retry_body is not None and unsupported_field_retries > 0:
                return self._post_lm_studio_native_chat(
                    retry_body,
                    context=context,
                    allow_tool_output_text=allow_tool_output_text,
                    retry_without_reasoning=retry_without_reasoning,
                    unsupported_field_retries=unsupported_field_retries - 1,
                )
            if "invalid_api_key" in detail or "API token is required" in detail:
                raise ProviderRequestError(
                    f"{context} needs a valid LM Studio API token. "
                    "Save the token in Settings, then refresh models and rerun."
                ) from exc
            if "Permission denied to use plugin" in detail:
                raise ProviderRequestError(
                    f"{context} was blocked by LM Studio server permissions. "
                    "In LM Studio, open Developer > Local Server > Server Settings and enable "
                    "'Allow calling servers from mcp.json' so API requests can use Brave Search/fetch."
                ) from exc
            raise ProviderRequestError(f"{context} failed at {native_base_url}: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            self.clear_model_verification_cache()
            raise ProviderRequestError(
                f"{context} could not reach LM Studio at {native_base_url}. "
                "Start the local server or update the LM Studio server URL in Settings."
            ) from exc
        except TimeoutError as exc:
            self.clear_model_verification_cache()
            raise ProviderTransientError(f"{context} timed out after {self.timeout_seconds}s at {native_base_url}") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(f"{context} returned non-JSON HTTP response from LM Studio.") from exc
        self._log_native_max_token_field_once()
        return _native_chat_text_usage(payload, context=context, allow_tool_output_text=allow_tool_output_text)

    def _native_max_tokens_retry_body(self, body: Dict, detail: str) -> Optional[Dict]:
        current_field = _native_max_token_field_from_body(body)
        if not current_field or not _is_max_token_field_error(detail, current_field):
            return None
        next_field = _next_native_max_token_field(current_field)
        if not next_field:
            return None
        max_tokens = int(body.get(current_field) or 0)
        retry_body = dict(body)
        for field_name in _native_max_token_field_names():
            retry_body.pop(field_name, None)
        retry_body[next_field] = max_tokens
        self._set_native_max_token_field(next_field)
        return retry_body

    def _post_lm_studio_structured_chat(
        self,
        body: Dict,
        *,
        context: str,
        retry_without_chat_template_kwargs: bool = True,
        unsupported_field_retries: int = 3,
    ) -> Tuple[Dict, _Usage]:
        openai_base_url = _lm_studio_openai_base_url(self.base_url)
        request = urllib.request.Request(
            f"{openai_base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=_request_headers(base_url=self.base_url, content_type="application/json"),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            if (
                retry_without_chat_template_kwargs
                and body.get("chat_template_kwargs")
                and _is_chat_template_kwargs_error(detail)
            ):
                self._warn_chat_template_kwargs_once(context)
                retry_body = dict(body)
                retry_body.pop("chat_template_kwargs", None)
                return self._post_lm_studio_structured_chat(
                    retry_body,
                    context=context,
                    retry_without_chat_template_kwargs=False,
                    unsupported_field_retries=unsupported_field_retries,
                )
            retry_body = _unsupported_fields_retry_body(body, detail)
            if retry_body is not None and unsupported_field_retries > 0:
                return self._post_lm_studio_structured_chat(
                    retry_body,
                    context=context,
                    retry_without_chat_template_kwargs=retry_without_chat_template_kwargs,
                    unsupported_field_retries=unsupported_field_retries - 1,
                )
            if "invalid_api_key" in detail or "API token is required" in detail:
                raise ProviderRequestError(
                    f"{context} needs a valid LM Studio API token. "
                    "Save the token in Settings, then refresh models and rerun."
                ) from exc
            raise ProviderRequestError(f"{context} failed at {self.base_url}: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderRequestError(
                f"{context} could not reach LM Studio at {openai_base_url}. "
                "Start the local server or update the LM Studio server URL in Settings."
            ) from exc
        except TimeoutError as exc:
            raise ProviderTransientError(f"{context} timed out after {self.timeout_seconds}s at {openai_base_url}") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(f"{context} returned non-JSON HTTP response from LM Studio.") from exc
        choices = payload.get("choices") or []
        if not choices:
            raise ProviderResponseError(f"{context} returned no choices from LM Studio.")
        message = choices[0].get("message") or {}
        parsed = _structured_json_from_message(message, context=context)
        usage = payload.get("usage") or {}
        return parsed, _Usage(
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
        )


def clear_lm_studio_model_verification_cache() -> None:
    _OpenAICompatibleClient.clear_model_verification_cache()


def _messages(prompt: str, *, system: str) -> List[Dict]:
    return [
        {"role": "system", "content": _system_prompt_no_think(system)},
        {"role": "user", "content": prompt},
    ]


def _system_prompt_no_think(system: str) -> str:
    clean = str(system or "").strip()
    if not clean:
        return "/no_think"
    if "/no_think" in clean:
        return clean
    return f"{clean} /no_think"


def _native_prompt_from_messages(messages: List[Dict]) -> Tuple[str, str]:
    system_parts: List[str] = []
    input_parts: List[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").strip().lower() or "user"
        content = _message_content_text(message.get("content"))
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        elif role == "user" and not input_parts:
            input_parts.append(content)
        else:
            input_parts.append(f"{role}: {content}")
    return "\n\n".join(system_parts).strip(), "\n\n".join(input_parts).strip()


def _message_content_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "\n".join(part.strip() for part in parts if part and part.strip())
    return str(content or "").strip()


def _is_ollama_base_url(base_url: str) -> bool:
    normalized = str(base_url or "").lower()
    return (
        "localhost:11434" in normalized
        or "127.0.0.1:11434" in normalized
        or "[::1]:11434" in normalized
    )


def _model_ids_from_payload(payload: Dict) -> List[str]:
    data = payload.get("data") if isinstance(payload, dict) else []
    if not isinstance(data, list):
        data = payload.get("models") if isinstance(payload, dict) else []
    model_ids: List[str] = []
    for item in (data if isinstance(data, list) else []):
        if isinstance(item, str):
            model_id = item.strip()
        elif isinstance(item, dict):
            if str(item.get("type") or "").strip().lower() == "embedding":
                continue
            model_id = str(item.get("id") or item.get("key") or item.get("name") or "").strip()
        else:
            model_id = ""
        if model_id and model_id not in model_ids:
            model_ids.append(model_id)
    return model_ids


def _payload_has_loaded_instance_state(payload: Dict) -> bool:
    data = payload.get("models") if isinstance(payload, dict) else []
    return isinstance(data, list) and any(isinstance(item, dict) and "loaded_instances" in item for item in data)


def _loaded_model_instances_from_payload(payload: Dict) -> Dict[str, str]:
    loaded: Dict[str, str] = {}
    for instance in _loaded_model_instance_rows_from_payload(payload):
        loaded[instance.instance_id] = instance.instance_id
        if instance.model_id and instance.model_id not in loaded:
            loaded[instance.model_id] = instance.instance_id
    return loaded


def _loaded_model_instance_rows_from_payload(payload: Dict) -> List[_LoadedModelInstance]:
    data = payload.get("models") if isinstance(payload, dict) else []
    loaded: List[_LoadedModelInstance] = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "").strip().lower() == "embedding":
            continue
        model_id = str(item.get("key") or item.get("id") or item.get("name") or "").strip()
        instances = item.get("loaded_instances") or []
        if not isinstance(instances, list) or not instances:
            continue
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            instance_id = str(instance.get("id") or "").strip()
            if not instance_id:
                continue
            loaded.append(_LoadedModelInstance(model_id=model_id, instance_id=instance_id))
    return loaded


def _loaded_instance_matches_model(instance: _LoadedModelInstance, model: str) -> bool:
    normalized = str(model or "").strip()
    return bool(normalized) and normalized in {instance.model_id, instance.instance_id}


def _preferred_loaded_instance_id(instances: List[_LoadedModelInstance], model: str) -> str:
    matching = [instance for instance in instances if _loaded_instance_matches_model(instance, model)]
    if not matching:
        return ""
    normalized = str(model or "").strip()
    for instance in matching:
        if instance.instance_id == normalized:
            return instance.instance_id
    for instance in matching:
        if not re.search(r":\d+$", instance.instance_id):
            return instance.instance_id
    return matching[0].instance_id


def _single_loaded_matching_instance_id(payload: Dict, model: str) -> str:
    loaded_rows = _loaded_model_instance_rows_from_payload(payload)
    if len(loaded_rows) == 1 and _loaded_instance_matches_model(loaded_rows[0], model):
        return loaded_rows[0].instance_id
    return ""


def _extra_loaded_instances(payload: Dict, keep_model: str) -> List[_LoadedModelInstance]:
    extras: List[_LoadedModelInstance] = []
    loaded_rows = _loaded_model_instance_rows_from_payload(payload)
    kept_instance_id = _preferred_loaded_instance_id(loaded_rows, keep_model)
    for instance in loaded_rows:
        matches_keep = _loaded_instance_matches_model(instance, keep_model)
        if matches_keep and instance.instance_id == kept_instance_id:
            continue
        extras.append(instance)
    return extras


def _model_list_url(base_url: str) -> str:
    if _is_ollama_base_url(base_url) or _uses_openai_only_api(base_url):
        return f"{_normalize_base_url(base_url)}/models"
    return f"{_lm_studio_native_base_url(base_url)}/models"


def _request_headers(*, base_url: str = "", content_type: Optional[str] = None) -> Dict[str, str]:
    token = "" if _is_ollama_base_url(base_url) else _lm_studio_api_token()
    headers: Dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _lm_studio_api_token() -> str:
    return str(os.environ.get("LM_STUDIO_API_KEY") or os.environ.get("LM_API_TOKEN") or "").strip()


def _is_reasoning_setting_error(detail: str) -> bool:
    lowered = str(detail or "").lower()
    return "reasoning" in lowered and (
        "unsupported" in lowered
        or "not supported" in lowered
        or "does not support" in lowered
        or "invalid" in lowered
    )


def _uses_lm_studio_native_api(base_url: str) -> bool:
    return not _is_ollama_base_url(base_url) and not _uses_openai_only_api(base_url)


def _uses_openai_only_api(base_url: str) -> bool:
    server_kind = str(os.environ.get("LECTURE_LOCAL_AI_SERVER_KIND") or "").strip().lower()
    if server_kind in {"openai", "openai-compatible", "omlx"}:
        return True
    try:
        parsed = urllib.parse.urlparse(_normalize_base_url(base_url))
    except Exception:
        return False
    return parsed.hostname == "192.168.86.22"


def _is_chat_template_kwargs_error(detail: str) -> bool:
    lowered = str(detail or "").lower()
    return ("chat_template_kwargs" in lowered or "enable_thinking" in lowered) and (
        "unsupported" in lowered
        or "not supported" in lowered
        or "unrecognized" in lowered
        or "unknown" in lowered
        or "invalid" in lowered
        or "extra" in lowered
    )


def _unsupported_fields_retry_body(body: Dict, detail: str) -> Optional[Dict]:
    fields = _unsupported_request_fields(detail)
    if not fields:
        return None
    retry_body = dict(body)
    removed = False
    for field in fields:
        if field in retry_body:
            retry_body.pop(field, None)
            removed = True
    return retry_body if removed else None


def _unsupported_request_fields(detail: str) -> List[str]:
    text = str(detail or "")
    lowered = text.lower()
    if not any(marker in lowered for marker in ("unrecognized", "unknown", "unsupported", "not supported", "extra")):
        return []
    if not any(marker in lowered for marker in ("key", "field", "parameter")):
        return []
    fields = []
    quoted_values = re.findall(r"'([^']+)'", text) + re.findall(r"\"([A-Za-z_][A-Za-z0-9_]*)\"", text)
    for value in quoted_values:
        field = value.strip()
        if field and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field):
            fields.append(field)
    ignored = {"error", "message", "type", "code"}
    return [field for field in fields if field not in ignored]


def _native_max_token_field_names() -> Tuple[str, ...]:
    return ("max_tokens", "max_completion_tokens", "max_output_tokens")


def _native_max_token_field_from_body(body: Dict) -> str:
    for field_name in _native_max_token_field_names():
        if field_name in body:
            return field_name
    return ""


def _next_native_max_token_field(field_name: str) -> str:
    fields = _native_max_token_field_names()
    try:
        index = fields.index(field_name)
    except ValueError:
        return ""
    next_index = index + 1
    if next_index >= len(fields):
        return ""
    return fields[next_index]


def _is_max_token_field_error(detail: str, field_name: str) -> bool:
    lowered = str(detail or "").lower()
    if field_name.lower() not in lowered:
        return False
    return (
        "unsupported" in lowered
        or "not supported" in lowered
        or "unrecognized" in lowered
        or "unknown" in lowered
        or "invalid" in lowered
        or "extra" in lowered
        or "not permitted" in lowered
    )


def _vision_input_items(request: AnalyzeLectureRequest, slides: List[Dict], prompt: str) -> List[Dict]:
    content = [{"type": "text", "content": prompt}]
    if request.lecture_dir:
        for slide in slides:
            path = _slide_image_path(request.lecture_dir, slide)
            if not path:
                continue
            data, mime_type = _slide_image_bytes(path)
            if not data or not mime_type:
                continue
            encoded = base64.standard_b64encode(data).decode("utf-8")
            content.append({"type": "text", "content": f"Attached image is slide_id={slide.get('id')} ({path.name})."})
            content.append({"type": "image", "data_url": f"data:{mime_type};base64,{encoded}"})
    return content


def _openai_image_messages(input_items: List[Dict], *, system: str) -> List[Dict]:
    content: List[Dict] = []
    for item in input_items:
        item_type = str(item.get("type") or "").strip()
        if item_type == "image":
            data_url = str(item.get("data_url") or "").strip()
            if data_url:
                content.append({"type": "image_url", "image_url": {"url": data_url}})
            continue
        text = str(item.get("content") or item.get("text") or "").strip()
        if text:
            content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": _system_prompt_no_think(system)},
        {"role": "user", "content": content or [{"type": "text", "text": ""}]},
    ]


def _native_chat_text_usage(
    payload: Dict,
    *,
    context: str,
    allow_tool_output_text: bool = False,
) -> Tuple[str, _Usage, List[str]]:
    output_items = payload.get("output") or []
    message_texts: List[str] = []
    tool_calls: List[str] = []
    tool_output_texts: List[str] = []
    invalid_tool_calls: List[str] = []
    for item in output_items if isinstance(output_items, list) else []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            content = str(item.get("content") or "").strip()
            if content:
                message_texts.append(content)
        elif item_type == "tool_call":
            tool_name = str(item.get("tool") or "").strip()
            if tool_name:
                tool_calls.append(tool_name)
            tool_output = str(item.get("output") or "").strip()
            if tool_output:
                label = tool_name or "tool"
                tool_output_texts.append(f"{label} output:\n{tool_output}")
        elif item_type == "invalid_tool_call":
            reason = str(item.get("reason") or "invalid tool call").strip()
            invalid_tool_calls.append(reason)

    if invalid_tool_calls:
        raise ProviderResponseError(
            f"{context} returned invalid LM Studio tool call(s): {'; '.join(invalid_tool_calls)}"
        )
    if not message_texts and allow_tool_output_text and tool_output_texts:
        message_texts.append("\n\n".join(tool_output_texts))
    if not message_texts:
        raise ProviderResponseError(f"{context} returned no message from LM Studio.")
    text = message_texts[-1].strip()
    if not text:
        raise ProviderResponseError(f"{context} returned an empty response from LM Studio.")
    stats = payload.get("stats") or {}
    return text, _Usage(
        input_tokens=int(stats.get("input_tokens") or 0),
        output_tokens=int(stats.get("total_output_tokens") or 0),
    ), tool_calls


def _openai_chat_text_usage(raw: str, *, context: str) -> Tuple[str, _Usage]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderResponseError(f"{context} returned non-JSON HTTP response from LM Studio.") from exc
    choices = payload.get("choices") or []
    if not choices:
        raise ProviderResponseError(f"{context} returned no choices from LM Studio.")
    message = choices[0].get("message") or {}
    text = str(message.get("content") or "").strip()
    if not text:
        raise ProviderResponseError(f"{context} returned an empty response from LM Studio.")
    usage = payload.get("usage") or {}
    return text, _Usage(
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
    )


def _normalize_lm_studio_profile(profile_name: str) -> str:
    normalized = str(profile_name or SUMMARIZATION_HQ_PROFILE).strip().upper()
    if normalized not in LM_STUDIO_PROFILES:
        return SUMMARIZATION_HQ_PROFILE
    return normalized


def _validate_lm_studio_load_config(profile_name: str, payload: Dict, context: str) -> None:
    applied = payload.get("load_config") or payload.get("config") or {}
    if not isinstance(applied, dict):
        return
    expected_context = LM_STUDIO_PROFILES[_normalize_lm_studio_profile(profile_name)]["load"].get("context_length")
    if expected_context is None or applied.get("context_length") is None:
        return
    try:
        applied_context = int(applied.get("context_length"))
    except (TypeError, ValueError):
        return
    if applied_context != int(expected_context):
        raise ProviderResponseError(
            f"{context} loaded LM Studio with context_length={applied_context}, expected {expected_context}. "
            "Check LM Studio JIT loading and model profile settings."
        )


def _lm_studio_rejected_load_config(detail: str) -> bool:
    lowered = str(detail or "").lower()
    return "config" in lowered and (
        "unrecognized key" in lowered
        or "unrecognized_keys" in lowered
        or "unknown field" in lowered
        or "extra inputs" in lowered
    )


def _structured_json_from_message(message: Dict, *, context: str) -> Dict:
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    content_text = str(content).strip() if content is not None else ""
    reasoning_text = str(reasoning).strip() if reasoning is not None else ""
    content_error: Optional[ProviderResponseError] = None
    if content_text:
        try:
            return _json_from_response_text(content_text)
        except ProviderResponseError as exc:
            content_error = exc
    if reasoning_text:
        try:
            return _json_from_response_text(reasoning_text)
        except ProviderResponseError as exc:
            if content_error:
                raise ProviderResponseError(
                    f"{context} returned invalid structured JSON in both content and reasoning_content: "
                    f"content={content_error}; reasoning_content={exc}"
                ) from exc
            raise ProviderResponseError(
                f"{context} returned invalid structured JSON in reasoning_content: {exc}"
            ) from exc
    if content_error:
        raise ProviderResponseError(
            f"{context} returned invalid structured JSON in content and no reasoning_content fallback: {content_error}"
        ) from content_error
    raise ProviderResponseError(f"{context} returned no structured JSON content from LM Studio.")


def _normalize_overview(payload: Dict, request: AnalyzeLectureRequest) -> Dict:
    fallback = local_overview(request, "mlx")
    outline = payload.get("outline")
    return {
        "title": str(payload.get("title") or fallback["title"]).strip(),
        "executive_summary": str(payload.get("executive_summary") or fallback["executive_summary"]).strip(),
        "outline": outline if isinstance(outline, list) and outline else fallback["outline"],
        "warnings": list(payload.get("warnings") or []),
    }


def _normalize_slide_analysis(items: List[Dict], request: AnalyzeLectureRequest, slides: List[Dict]) -> List[Dict]:
    fallback_by_id = {
        int(item.get("slide_id") or index): item
        for index, item in enumerate(local_slide_analysis(request, "mlx")["slide_analysis"], start=1)
    }
    normalized = []
    for index, slide in enumerate(slides, start=1):
        slide_id = int(slide.get("id") or index)
        item = next((candidate for candidate in items if int(candidate.get("slide_id") or 0) == slide_id), None)
        fallback = fallback_by_id.get(slide_id) or {}
        normalized.append(
            {
                "slide_id": slide_id,
                "descriptive_filename": str(
                    (item or {}).get("descriptive_filename")
                    or fallback.get("descriptive_filename")
                    or f"slide_{slide_id:04d}_mlx.png"
                ),
                "caption": (item or {}).get("caption"),
                "summary": str((item or {}).get("summary") or fallback.get("summary") or "").strip(),
                "tags": list((item or {}).get("tags") or fallback.get("tags") or ["lecture", f"slide-{slide_id}"]),
                "instructor_commentary": str(
                    (item or {}).get("instructor_commentary")
                    or fallback.get("instructor_commentary")
                    or ""
                ).strip(),
            }
        )
    return normalized


def _normalize_resources(payload: Dict) -> Dict:
    resources = []
    for item in payload.get("resources") or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        summary = str(item.get("summary") or "").strip()
        if not title or not url or not summary:
            continue
        resources.append(
            {
                "title": title,
                "url": url,
                "summary": summary,
                "source_quality": str(item.get("source_quality") or "medium").strip() or "medium",
            }
        )
    return {
        "resources": resources[:DIRECT_RESOURCE_OUTPUT_LIMIT],
        "warnings": list(payload.get("warnings") or []),
    }


def _empty_resources_payload(warnings: List[str]) -> Dict:
    return {
        "resources": [],
        "warnings": _dedupe_warnings(warnings or ["Resource search produced no usable results."]),
    }


def _resource_query_plan_schema() -> Dict:
    return {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "minItems": 3,
                "maxItems": RESOURCE_PLANNER_QUERY_LIMIT,
                "items": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "intent": {"type": "string"},
                    },
                    "required": ["query", "intent"],
                    "additionalProperties": False,
                },
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["queries", "warnings"],
        "additionalProperties": False,
    }


def _resource_formatter_schema() -> Dict:
    return {
        "type": "object",
        "properties": {
            "resources": {
                "type": "array",
                "maxItems": DIRECT_RESOURCE_OUTPUT_LIMIT,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "summary": {"type": "string"},
                        "source_quality": {"type": "string", "enum": ["high", "medium"]},
                    },
                    "required": ["title", "url", "summary", "source_quality"],
                    "additionalProperties": False,
                },
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["resources", "warnings"],
        "additionalProperties": False,
    }


def _queries_from_plan(payload: Dict) -> List[str]:
    queries: List[str] = []
    for item in (payload or {}).get("queries") or []:
        if isinstance(item, dict):
            query = str(item.get("query") or "").strip()
        else:
            query = str(item or "").strip()
        cleaned = _trim_text(_clean_resource_text(query), 160)
        if cleaned and cleaned.lower() not in {existing.lower() for existing in queries}:
            queries.append(cleaned)
        if len(queries) >= RESOURCE_PLANNER_QUERY_LIMIT:
            break
    return queries


def _direct_resource_candidates(
    *,
    title: str,
    summary: str,
    outline_headings: List[str],
    timeout_seconds: int,
    queries: Optional[List[str]] = None,
) -> Tuple[List[Dict], List[str]]:
    token = _brave_api_token()
    if not token:
        return [], [
            "Direct Brave resource search skipped: no BRAVE_API_KEY was found in the environment or LM Studio mcp.json."
        ]

    warnings: List[str] = []
    gathered: List[Dict] = []
    seen_urls: set = set()
    query_timeout = _resource_query_timeout(timeout_seconds)
    search_queries = _clean_resource_queries(queries) or _resource_search_queries(title, summary, outline_headings)
    for query in search_queries[:RESOURCE_PLANNER_QUERY_LIMIT]:
        try:
            payload = _brave_web_search(query, token, timeout_seconds=query_timeout)
        except Exception as exc:
            warnings.append(f"Brave resource search failed for query '{query}': {exc}")
            continue
        for candidate in _resource_candidates_from_brave_payload(payload):
            _append_resource(
                gathered,
                seen_urls,
                title=str(candidate.get("title") or ""),
                url=str(candidate.get("url") or ""),
                summary=str(candidate.get("summary") or ""),
            )
            if len(gathered) >= DIRECT_RESOURCE_VERIFY_LIMIT:
                break
        if len(gathered) >= DIRECT_RESOURCE_VERIFY_LIMIT:
            break

    if not gathered:
        return [], warnings or ["Direct Brave resource search returned no candidate links."]

    verified: List[Dict] = []
    for candidate in gathered[:DIRECT_RESOURCE_VERIFY_LIMIT]:
        if len(verified) >= DIRECT_RESOURCE_OUTPUT_LIMIT:
            break
        if _resource_url_is_reachable(str(candidate.get("url") or ""), timeout_seconds=_resource_fetch_timeout(timeout_seconds)):
            verified.append(candidate)

    if verified:
        return verified, warnings

    warnings.append(
        "Direct URL verification did not confirm any candidates; saved top Brave Search results instead."
    )
    return gathered[:DIRECT_RESOURCE_OUTPUT_LIMIT], warnings


def _brave_api_token() -> str:
    token = str(os.environ.get("BRAVE_API_KEY") or os.environ.get("BRAVE_SEARCH_API_KEY") or "").strip()
    if token:
        return token
    mcp_path = Path.home() / ".lmstudio" / "mcp.json"
    if not mcp_path.exists():
        return ""
    try:
        data = json.loads(mcp_path.read_text())
    except Exception:
        return ""
    servers = data.get("mcpServers") or data.get("servers") or {}
    if not isinstance(servers, dict):
        return ""
    for name, config in servers.items():
        if "brave" not in str(name).lower() or not isinstance(config, dict):
            continue
        env = config.get("env") or {}
        if not isinstance(env, dict):
            continue
        token = str(env.get("BRAVE_API_KEY") or env.get("BRAVE_SEARCH_API_KEY") or "").strip()
        if token:
            return token
    return ""


def _brave_web_search(query: str, token: str, *, timeout_seconds: int) -> Dict:
    params = urllib.parse.urlencode(
        {
            "q": query,
            "count": DIRECT_RESOURCE_RESULTS_PER_QUERY,
            "country": "us",
            "search_lang": "en",
        }
    )
    request = urllib.request.Request(
        f"{BRAVE_SEARCH_ENDPOINT}?{params}",
        headers={
            "Accept": "application/json",
            "User-Agent": "LectureProcessor/0.6 resource-search",
            "X-Subscription-Token": token,
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise ProviderRequestError(f"HTTP {exc.code}: {_trim_text(detail, 300)}") from exc
    except urllib.error.URLError as exc:
        raise ProviderRequestError(str(exc.reason or exc)) from exc
    except TimeoutError as exc:
        raise ProviderTransientError(f"timed out after {timeout_seconds}s") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderResponseError("Brave Search returned non-JSON response.") from exc


def _resource_candidates_from_brave_payload(payload: Dict) -> List[Dict]:
    results = ((payload or {}).get("web") or {}).get("results") or []
    candidates: List[Dict] = []
    seen_urls: set = set()
    for item in results if isinstance(results, list) else []:
        if not isinstance(item, dict):
            continue
        title = _clean_resource_text(item.get("title") or "")
        url = _clean_resource_url(item.get("url") or "")
        description_parts = [str(item.get("description") or "")]
        snippets = item.get("extra_snippets") or []
        if isinstance(snippets, list):
            description_parts.extend(str(snippet or "") for snippet in snippets[:2])
        summary = _trim_text(_clean_resource_text(" ".join(description_parts)), 420)
        if _is_low_value_resource_url(url):
            continue
        _append_resource(candidates, seen_urls, title=title, url=url, summary=summary)
    return candidates


def _resource_search_queries(title: str, summary: str, outline_headings: List[str]) -> List[str]:
    base = _clean_resource_text(title) or "lecture topic"
    heading_terms = [
        _clean_resource_text(heading)
        for heading in outline_headings
        if _clean_resource_text(heading)
    ]
    keyword_text = " ".join([base, " ".join(heading_terms[:3]), summary])
    keywords = _resource_keywords(keyword_text)
    keyword_query = " ".join(keywords[:6])
    queries = [
        f"{base} study resources",
        f"{base} research paper university",
    ]
    if keyword_query and keyword_query.lower() not in base.lower():
        queries.append(f"{keyword_query} overview")
    for heading in heading_terms[:3]:
        queries.append(f"{base} {heading}")
    deduped: List[str] = []
    for query in queries:
        cleaned = _trim_text(_clean_resource_text(query), 140)
        if cleaned and cleaned.lower() not in {item.lower() for item in deduped}:
            deduped.append(cleaned)
        if len(deduped) >= DIRECT_RESOURCE_MAX_QUERIES:
            break
    return deduped or [base]


def _clean_resource_queries(queries: Optional[List[str]]) -> List[str]:
    cleaned_queries: List[str] = []
    for query in queries or []:
        cleaned = _trim_text(_clean_resource_text(query), 160)
        if cleaned and cleaned.lower() not in {existing.lower() for existing in cleaned_queries}:
            cleaned_queries.append(cleaned)
        if len(cleaned_queries) >= RESOURCE_PLANNER_QUERY_LIMIT:
            break
    return cleaned_queries


def _resource_keywords(text: str) -> List[str]:
    stop_words = {
        "about",
        "across",
        "after",
        "also",
        "and",
        "are",
        "based",
        "between",
        "can",
        "creative",
        "from",
        "how",
        "into",
        "lecture",
        "life",
        "main",
        "model",
        "over",
        "stage",
        "stages",
        "that",
        "the",
        "this",
        "through",
        "use",
        "uses",
        "with",
    }
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", str(text or "").lower())
    scored: Dict[str, int] = {}
    for word in words:
        if word in stop_words:
            continue
        scored[word] = scored.get(word, 0) + 1
    return [
        word
        for word, _count in sorted(scored.items(), key=lambda item: (-item[1], item[0]))
    ]


def _resource_url_is_reachable(url: str, *, timeout_seconds: int) -> bool:
    if not _clean_resource_url(url):
        return False
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/html,application/pdf,*/*",
            "User-Agent": "LectureProcessor/0.6 resource-verify",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(getattr(response, "status", 0) or response.getcode() or 0)
            response.read(2048)
            return 200 <= status < 400
    except urllib.error.HTTPError as exc:
        return int(exc.code or 0) in {401, 403, 405, 429}
    except Exception:
        return False


def _resource_query_timeout(timeout_seconds: int) -> int:
    return max(5, min(15, int(timeout_seconds or DEFAULT_TIMEOUT_SECONDS) // 4))


def _resource_fetch_timeout(timeout_seconds: int) -> int:
    return max(4, min(8, int(timeout_seconds or DEFAULT_TIMEOUT_SECONDS) // 8))


def _is_low_value_resource_url(url: str) -> bool:
    lowered = str(url or "").lower()
    blocked_domains = (
        "facebook.com",
        "instagram.com",
        "linkedin.com/feed",
        "pinterest.com",
        "reddit.com",
        "tiktok.com",
        "x.com/",
        "youtube.com/shorts",
    )
    return any(domain in lowered for domain in blocked_domains)


def _resources_matching_candidates(resources: List[Dict], candidates: List[Dict]) -> List[Dict]:
    candidates_by_url = {
        _clean_resource_url(str(candidate.get("url") or "")): candidate
        for candidate in candidates
    }
    matched: List[Dict] = []
    seen_urls: set = set()
    for item in resources:
        url = _clean_resource_url(item.get("url") or "")
        candidate = candidates_by_url.get(url)
        if not candidate:
            continue
        _append_resource(
            matched,
            seen_urls,
            title=str(item.get("title") or candidate.get("title") or ""),
            url=str(candidate.get("url") or url),
            summary=str(item.get("summary") or candidate.get("summary") or ""),
        )
        if matched:
            matched[-1]["source_quality"] = str(
                item.get("source_quality") or candidate.get("source_quality") or "medium"
            ).strip() or "medium"
    return matched[:DIRECT_RESOURCE_OUTPUT_LIMIT]


def _fill_resources_from_candidates(resources: List[Dict], candidates: List[Dict]) -> List[Dict]:
    filled = list(resources)
    seen_urls = {_resource_url_key(str(item.get("url") or "")) for item in filled}
    for candidate in candidates:
        if len(filled) >= DIRECT_RESOURCE_OUTPUT_LIMIT:
            break
        key = _resource_url_key(str(candidate.get("url") or ""))
        if not key or key in seen_urls:
            continue
        filled.append(
            {
                "title": str(candidate.get("title") or "").strip() or _resource_title_from_url(candidate.get("url") or ""),
                "url": str(candidate.get("url") or "").strip(),
                "summary": str(candidate.get("summary") or "").strip()
                or "External resource found by Brave Search for this lecture topic.",
                "source_quality": str(candidate.get("source_quality") or "medium").strip() or "medium",
            }
        )
        seen_urls.add(key)
    return filled[:DIRECT_RESOURCE_OUTPUT_LIMIT]


def _resources_from_candidates(candidates: List[Dict]) -> List[Dict]:
    return _fill_resources_from_candidates([], candidates)


def _resources_from_gathered_context(text: str) -> List[Dict]:
    resources: List[Dict] = []
    seen_urls = set()
    clean = _clean_gathered_context(text)
    title_description_pattern = re.compile(
        r"Title:\s*(?P<title>.*?)\s+Description:\s*(?P<summary>.*?)\s+URL:\s*(?P<url>https?://[^\s\\\"<>]+)",
        re.IGNORECASE | re.DOTALL,
    )
    for match in title_description_pattern.finditer(clean):
        title = _clean_resource_text(match.group("title"))
        summary = _clean_resource_text(match.group("summary"))
        url = _clean_resource_url(match.group("url"))
        _append_resource(resources, seen_urls, title=title, url=url, summary=summary)
        if len(resources) >= DIRECT_RESOURCE_OUTPUT_LIMIT:
            break
    if len(resources) >= DIRECT_RESOURCE_OUTPUT_LIMIT:
        return resources[:DIRECT_RESOURCE_OUTPUT_LIMIT]

    blocks = re.split(r"(?=\n\s*(?:#{2,3}\s*)?\d+[\).\s-])", f"\n{clean}")
    for block in blocks:
        if len(resources) >= DIRECT_RESOURCE_OUTPUT_LIMIT:
            break
        url_match = re.search(r"https?://[^\s\\\"<>]+", block)
        if not url_match:
            continue
        url = _clean_resource_url(url_match.group(0))
        if not url or url in seen_urls:
            continue
        title = _resource_title_from_block(block)
        summary = _resource_summary_from_block(block)
        _append_resource(resources, seen_urls, title=title, url=url, summary=summary)

    if len(resources) >= DIRECT_RESOURCE_OUTPUT_LIMIT:
        return resources[:DIRECT_RESOURCE_OUTPUT_LIMIT]

    for match in re.finditer(r"https?://[^\s\\\"<>]+", clean):
        if len(resources) >= DIRECT_RESOURCE_OUTPUT_LIMIT:
            break
        url = _clean_resource_url(match.group(0))
        if not url or url in seen_urls:
            continue
        start = max(0, match.start() - 450)
        end = min(len(clean), match.end() + 700)
        block = clean[start:end]
        title = _resource_title_from_block(block)
        summary = _resource_summary_from_block(block)
        _append_resource(resources, seen_urls, title=title, url=url, summary=summary)
    return resources


def _append_resource(resources: List[Dict], seen_urls: set, *, title: str, url: str, summary: str) -> None:
    title = _clean_resource_text(title)
    url = _clean_resource_url(url)
    summary = _clean_resource_text(summary)
    url_key = _resource_url_key(url)
    if not title:
        title = _resource_title_from_url(url)
    if not summary:
        summary = "External resource found for this lecture topic."
    if not title or not url or not url_key or url_key in seen_urls:
        return
    seen_urls.add(url_key)
    resources.append(
        {
            "title": title,
            "url": url,
            "summary": summary,
            "source_quality": _source_quality_for_url(url),
        }
    )


def _clean_gathered_context(text: str) -> str:
    clean = str(text or "").replace("\\n", "\n")
    clean = clean.replace('\\"', '"')
    clean = clean.replace("\\/", "/")
    return clean


def _resource_title_from_block(block: str) -> str:
    patterns = [
        r"\*\*Title:\*\*\s*(?P<value>[^\n]+)",
        r"Title:\s*(?P<value>[^\n]+)",
        r"^\s*(?:#{2,3}\s*)?\d+[\).\s-]+\*\*(?P<value>[^*]+)\*\*",
        r"^\s*(?:#{2,3}\s*)?\d+[\).\s-]+(?P<value>[^\n]+)",
        r"\*\*(?P<value>[^*\n]{8,120})\*\*",
        r"(?P<value>[A-Z][^:\n]{3,120}):\s*https?://",
    ]
    for pattern in patterns:
        match = re.search(pattern, block, re.IGNORECASE | re.MULTILINE)
        if match:
            value = re.sub(r"https?://\S+", "", match.group("value"))
            return _clean_resource_text(value)
    return ""


def _resource_summary_from_block(block: str) -> str:
    patterns = [
        r"\*\*Relevance:\*\*\s*(?P<value>.*?)(?:\n\s*\*\*|\n\s*#{2,3}\s*\d+|$)",
        r"Relevance:\s*(?P<value>.*?)(?:\n\s*\*\*|\n\s*#{2,3}\s*\d+|$)",
        r"\*\*Description:\*\*\s*(?P<value>.*?)(?:\n\s*\*\*|\n\s*#{2,3}\s*\d+|$)",
        r"Description:\s*(?P<value>.*?)(?:\n\s*\*\*|\n\s*#{2,3}\s*\d+|$)",
        r"\*\*Summary:\*\*\s*(?P<value>.*?)(?:\n\s*\*\*|\n\s*#{2,3}\s*\d+|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, block, re.IGNORECASE | re.DOTALL)
        if match:
            return _clean_resource_text(match.group("value"))
    without_url = re.sub(r"https?://\S+", "", block)
    without_markers = re.sub(r"(?i)\b(URL|Source Quality|Title)\s*:\s*", "", without_url)
    return _trim_text(_clean_resource_text(without_markers), 280)


def _clean_resource_text(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", "", str(text or ""))
    cleaned = re.sub(r"[*_`#]+", "", cleaned)
    cleaned = cleaned.replace("\\n", " ").replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" -:;,.")


def _clean_resource_url(url: str) -> str:
    cleaned = str(url or "").strip()
    cleaned = cleaned.rstrip(".,;)]}'\"")
    return cleaned if cleaned.startswith(("http://", "https://")) else ""


def _resource_url_key(url: str) -> str:
    cleaned = _clean_resource_url(url)
    if not cleaned:
        return ""
    parsed = urllib.parse.urlsplit(cleaned)
    if not parsed.netloc:
        return cleaned.lower().rstrip("/")
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=False)
    kept_query = [
        (key, value)
        for key, value in query
        if not key.lower().startswith("utm_") and key.lower() not in {"fbclid", "gclid"}
    ]
    normalized_query = urllib.parse.urlencode(sorted(kept_query))
    return urllib.parse.urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower().removeprefix("www."),
            parsed.path.rstrip("/") or "/",
            normalized_query,
            "",
        )
    )


def _source_quality_for_url(url: str) -> str:
    lowered = url.lower()
    high_markers = (
        ".edu",
        ".gov",
        "acm.org",
        "arxiv.org",
        "doi.org",
        "harvard.edu",
        "ibm.com",
        "ieee.org",
        "mit.edu",
        "microsoft.com",
        "nature.com",
        "oecd.org",
        "science.org",
        "stanford.edu",
        "unesco.org",
    )
    return "high" if any(marker in lowered for marker in high_markers) else "medium"


def _resource_title_from_url(url: str) -> str:
    cleaned = re.sub(r"^https?://", "", str(url or ""))
    host = cleaned.split("/", 1)[0].replace("www.", "")
    slug = cleaned.rsplit("/", 1)[-1]
    slug = re.sub(r"[-_]+", " ", slug).strip()
    if slug and "." not in slug:
        return slug.title()
    return host or "External resource"


def _outline_heading_values(overview: Optional[Dict]) -> List[str]:
    headings: List[str] = []
    for item in (overview or {}).get("outline") or []:
        if not isinstance(item, dict):
            continue
        heading = str(item.get("heading") or "").strip()
        if heading:
            headings.append(heading)
    return headings


def _outline_headings_text(overview: Optional[Dict]) -> str:
    return _outline_headings_text_from_values(_outline_heading_values(overview))


def _outline_headings_text_from_values(headings: List[str]) -> str:
    cleaned = [_clean_resource_text(heading) for heading in headings if _clean_resource_text(heading)]
    return "\n".join(f"- {heading}" for heading in cleaned) if cleaned else "- No outline headings available."


def _planner_queries_text(queries: List[str]) -> str:
    cleaned = _clean_resource_queries(queries)
    if not cleaned:
        return "no planner queries were available"
    return "; ".join(f"{index}. {query}" for index, query in enumerate(cleaned, start=1))


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


def _transcript_chunks(text: str) -> List[str]:
    clean = str(text or "").strip()
    if not clean:
        return [""]
    chunks = []
    current = []
    current_len = 0
    for paragraph in clean.splitlines() or [clean]:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > TRANSCRIPT_CHUNK_CHARS:
            if current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            chunks.extend(_split_long_text(paragraph, TRANSCRIPT_CHUNK_CHARS))
            continue
        if current and current_len + len(paragraph) > TRANSCRIPT_CHUNK_CHARS:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += len(paragraph)
    if current:
        chunks.append("\n".join(current))
    if not chunks:
        chunks = [clean[:TRANSCRIPT_CHUNK_CHARS]]
    return chunks


def _split_long_text(text: str, max_chars: int) -> List[str]:
    chunks = []
    remaining = text
    while remaining:
        chunk = remaining[:max_chars]
        if len(remaining) > max_chars and " " in chunk:
            chunk = chunk.rsplit(" ", 1)[0]
        chunks.append(chunk.strip())
        remaining = remaining[len(chunk) :].strip()
    return chunks


def _configured_slide_batch_size() -> int:
    raw = str(os.environ.get("LECTURE_SLIDE_BATCH_SIZE") or "").strip()
    if not raw:
        return DEFAULT_SLIDE_BATCH_SIZE
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_SLIDE_BATCH_SIZE
    return max(1, min(value, MAX_SLIDE_BATCH_SIZE))


def _slide_batches(slides: List[Dict], batch_size: Optional[int] = None) -> List[List[Dict]]:
    size = max(1, int(batch_size or _configured_slide_batch_size()))
    return [slides[index : index + size] for index in range(0, len(slides), size)]


def _slide_batch_max_tokens(slides: List[Dict]) -> int:
    return DEFAULT_ROLE_MAX_TOKENS["slide_analysis"]


def _slide_analysis_from_extraction_metadata(request: AnalyzeLectureRequest, model: str) -> Optional[Dict]:
    slides = request.slides or []
    if not slides or not all(str(slide.get("description") or "").strip() for slide in slides):
        return None
    analysis = []
    for index, slide in enumerate(slides, start=1):
        slide_id = _slide_id(slide, fallback=index)
        description = str(slide.get("description") or "").strip()
        title = str(slide.get("title") or "").strip()
        tags = ["smart-slide-extraction", f"slide-{slide_id}"]
        for key in ("build_stage", "layout"):
            value = str(slide.get(key) or "").strip()
            if value:
                tags.append(value)
        summary = " ".join(part for part in [title, description] if part).strip() or description
        analysis.append(
            {
                "slide_id": slide_id,
                "descriptive_filename": slide.get("filename") or f"slide_{slide_id:04d}.png",
                "caption": description,
                "summary": _trim_text(summary, 420),
                "tags": tags,
                "instructor_commentary": _batch_transcript_context(request, [slide]),
            }
        )
    return {
        "slide_analysis": analysis,
        "warnings": [
            f"Slide analysis used smart slide extraction metadata from local LM Studio vision model: {model}",
            "Slide analysis skipped a second raw-image pass because slide title and description were already extracted.",
        ],
        "input_tokens": max(1, sum(_estimate_tokens(str(slide.get("description") or "")) for slide in slides)),
        "output_tokens": max(1, len(analysis) * 80),
        "batch_count": 0,
        "retry_count": 0,
    }


def _slide_batch_prompt(request: AnalyzeLectureRequest, slides: List[Dict]) -> str:
    return f"""
{JSON_ONLY_INSTRUCTION}
Return field slide_analysis.
Return one slide_analysis item for every slide listed.
Each item must include: slide_id, descriptive_filename, caption, summary, tags, instructor_commentary.

Analyze the attached image for each listed slide_id. For caption and summary, describe only what is visible in
the image. Do not use transcript context to guess visual content.
For each attached image, use only the visual content for that same slide_id. Do not mix visual details between slide_ids.

If the image is a speaker-only frame, transition screen, video-player screen, or otherwise not an actual slide,
say that directly in caption and summary. Do not borrow the topic from nearby transcript.
Treat low-value frames as discard candidates: use tags such as "low-value", "speaker-only", "transition",
"video-player", "duplicate", or "not-slide" when they apply, and keep their summary brief instead of inventing
educational value.

=== VISUAL ANALYSIS INPUT (use for caption, summary, tags) ===
Slides:
{_slide_index_text(slides)}

=== TRANSCRIPT CONTEXT (use ONLY for instructor_commentary, never for caption/summary/tags) ===
{_batch_transcript_context(request, slides)}
""".strip()


def _slide_index_text(slides: List[Dict]) -> str:
    if not slides:
        return "No slides."
    lines = []
    for index, slide in enumerate(slides, start=1):
        metadata = []
        for label, key in (("title", "title"), ("stage", "build_stage"), ("layout", "layout")):
            value = str(slide.get(key) or "").strip()
            if value:
                metadata.append(f"{label}={value}")
        description = str(slide.get("description") or "").strip()
        if description:
            metadata.append(f"description={_trim_text(description, 220)}")
        metadata_text = f" {' | '.join(metadata)}" if metadata else ""
        lines.append(
            f"- slide_id={int(slide.get('id') or index)} timestamp={float(slide.get('timestamp_seconds') or 0.0):.1f}s "
            f"segments={slide.get('linked_segment_ids') or []}{metadata_text}"
        )
    return "\n".join(lines)


def _missing_slide_ids(items: List[Dict], slides: List[Dict]) -> List[int]:
    expected = [_slide_id(slide, fallback=index) for index, slide in enumerate(slides, start=1)]
    returned = {
        int(item.get("slide_id") or 0)
        for item in items
        if isinstance(item, dict)
    }
    return [slide_id for slide_id in expected if slide_id not in returned]


def _slides_with_ids(slides: List[Dict], slide_ids: List[int]) -> List[Dict]:
    wanted = set(slide_ids)
    return [
        slide
        for index, slide in enumerate(slides, start=1)
        if _slide_id(slide, fallback=index) in wanted
    ]


def _slide_ids(slides: List[Dict]) -> List[int]:
    return [_slide_id(slide, fallback=index) for index, slide in enumerate(slides, start=1)]


def _slide_id(slide: Dict, *, fallback: int) -> int:
    try:
        return int(slide.get("id") or fallback)
    except (TypeError, ValueError):
        return fallback


def _batch_transcript_context(request: AnalyzeLectureRequest, slides: List[Dict]) -> str:
    segment_ids = set()
    for slide in slides:
        segment_ids.update(slide.get("linked_segment_ids") or [])
    parts = [
        f"[{segment.get('id')}] {segment.get('text')}"
        for segment in request.segments
        if segment.get("id") in segment_ids and str(segment.get("text") or "").strip()
    ]
    if not parts:
        return _trim_text(request.transcript_text, 1800)
    return _trim_text("\n".join(parts), 1800)


def _trim_text(text: str, max_chars: int) -> str:
    clean = str(text or "").strip()
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rsplit(" ", 1)[0] + "\n[trimmed]"


def _attach_usage(payload: Dict, usage: _Usage, prompt: str, response_payload: Dict) -> None:
    payload["input_tokens"] = usage.input_tokens or _estimate_tokens(prompt)
    payload["output_tokens"] = usage.output_tokens or _estimate_tokens(json.dumps(response_payload))


def _estimate_tokens(text: str) -> int:
    return max(1, len(str(text or "")) // 4)


def _sum_tokens(key: str, *payloads: Dict) -> int:
    return sum(int(payload.get(key) or 0) for payload in payloads)


def _token_usage_entry_from_payload(payload: Dict) -> Dict[str, int]:
    return _token_usage_entry(
        payload.get("input_tokens") if isinstance(payload, dict) else 0,
        payload.get("output_tokens") if isinstance(payload, dict) else 0,
    )


def _token_usage_entry_from_usage(usage: "_Usage") -> Dict[str, int]:
    return _token_usage_entry(usage.input_tokens, usage.output_tokens)


def _token_usage_entry(input_tokens, output_tokens) -> Dict[str, int]:
    input_count = max(0, int(input_tokens or 0))
    output_count = max(0, int(output_tokens or 0))
    return {
        "input_tokens": input_count,
        "output_tokens": output_count,
        "total_tokens": input_count + output_count,
    }


def _normalize_base_url(base_url: str) -> str:
    return str(base_url or "").strip().rstrip("/") or DEFAULT_TEXT_BASE_URL


def _lm_studio_native_base_url(base_url: str) -> str:
    normalized = _normalize_base_url(base_url)
    if normalized.endswith("/api/v1"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized[:-3]}/api/v1"
    return f"{normalized}/api/v1"


def _lm_studio_openai_base_url(base_url: str) -> str:
    normalized = _normalize_base_url(base_url)
    if normalized.endswith("/api/v1"):
        return f"{normalized[:-7]}/v1"
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def _bounded_timeout(value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_TIMEOUT_SECONDS
    return max(10, min(600, parsed))


def _dedupe_warnings(warnings: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for warning in warnings:
        text = str(warning or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped
