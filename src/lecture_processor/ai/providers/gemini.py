import importlib
import io
import json
import math
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from lecture_processor.artifacts import load_json, save_json, utc_now_iso
from lecture_processor.errors import DependencyMissingError

from .base import (
    AnalyzeLectureRequest,
    AnalyzeLectureResponse,
    ProviderAuthError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTransientError,
    ProviderInfo,
)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
TRANSCRIPT_CHUNK_MAX_CHARS = 6000
SLIDE_BATCH_SIZE = 10
DEFAULT_MAX_CONCURRENCY = 6
MAX_RETRY_ATTEMPTS = 4
SLIDE_IMAGE_MAX_EDGE = 1024
SLIDE_IMAGE_WEBP_QUALITY = 80
SLIDE_AI_CACHE_DIR = ".ai-cache"
SLIDE_ENTROPY_MIN_BITS = 1.0
SLIDE_PHASH_DISTANCE_THRESHOLD = 5
SLIDE_PHASH_SIZE = 8
SLIDE_PHASH_SAMPLE_SIZE = 32
MIN_RESOURCE_COUNT = 3
MAX_RESOURCE_COUNT = 4


class GeminiProvider:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        if not api_key:
            raise ProviderAuthError("Gemini API key is required.")
        self.model = model or DEFAULT_GEMINI_MODEL
        self.max_concurrency = _bounded_concurrency(max_concurrency)
        try:
            genai = importlib.import_module("google.genai")
            types = importlib.import_module("google.genai.types")
        except ImportError as exc:
            raise DependencyMissingError(
                "Gemini support is not installed. Install with: python3 -m pip install -e '.[ai]'"
            ) from exc
        self._client = genai.Client(api_key=api_key)
        self._types = types

    @classmethod
    def info(cls) -> ProviderInfo:
        return ProviderInfo(
            name="gemini",
            display_name="Google Gemini",
            available_models=[
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
            ],
            default_model=DEFAULT_GEMINI_MODEL,
            docs_url="https://ai.google.dev/gemini-api/docs/models/gemini",
        )

    def test_connection(self) -> None:
        try:
            self._client.models.generate_content(
                model=self.model,
                contents="Reply with OK.",
                config={"max_output_tokens": 4},
            )
        except Exception as exc:
            raise _map_gemini_error(exc) from exc

    def analyze_lecture(self, request: AnalyzeLectureRequest) -> AnalyzeLectureResponse:
        return self.analyze_lecture_chunked(request)

    def analyze_lecture_chunked(
        self,
        request: AnalyzeLectureRequest,
        *,
        cache_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[Dict], None]] = None,
        include_overview: bool = True,
        include_transcript: bool = True,
        include_slides: bool = True,
        include_resources: bool = True,
        overview_override: Optional[Dict] = None,
    ) -> AnalyzeLectureResponse:
        cache = _load_chunk_cache(cache_path, request, self.model)
        warnings = list(cache.get("warnings") or [])
        usage = {"input": 0, "output": 0}
        transcript_chunks = _transcript_chunks(request) if include_transcript else []
        slides_for_gemini, filtered_slides = (
            _filter_slides_for_gemini(request) if include_slides else ([], {})
        )
        slide_filter_warning = _slide_filter_warning(filtered_slides)
        if slide_filter_warning:
            warnings.append(slide_filter_warning)
        slide_batches = _slide_batches(slides_for_gemini)
        needs_overview = include_overview or include_slides or include_resources
        should_generate_overview = needs_overview and overview_override is None
        total_steps = (
            (1 if should_generate_overview else 0)
            + len(transcript_chunks)
            + len(slide_batches)
            + (1 if include_resources else 0)
        )
        completed_steps = 0

        def emit(step: str) -> None:
            _emit_progress(
                progress_callback,
                step=step,
                completed=completed_steps,
                total=total_steps,
                input_tokens=usage["input"],
                output_tokens=usage["output"],
            )

        chunks = cache.setdefault("chunks", {})
        overview = overview_override
        if should_generate_overview:
            overview = chunks.get("overview")
        if should_generate_overview and not overview:
            try:
                overview, response = self._generate_json(
                    _overview_prompt(request),
                    request,
                    use_google_search=False,
                    include_images=False,
                    context="Gemini overview",
                    response_schema=_overview_response_schema(),
                )
                _add_usage(usage, response)
            except ProviderResponseError as exc:
                overview = {
                    "title": _fallback_title(request),
                    "executive_summary": _fallback_summary(request.transcript_text),
                    "outline": _fallback_outline(request.slides),
                    "warnings": [f"Overview used transcript fallback: {exc}"],
                }
            chunks["overview"] = overview
            _save_chunk_cache(cache_path, cache)
        if not overview:
            overview = {
                "title": _fallback_title(request),
                "executive_summary": _fallback_summary(request.transcript_text),
                "outline": _fallback_outline(request.slides),
                "warnings": [],
            }
        warnings.extend(overview.get("warnings") or [])
        if should_generate_overview:
            completed_steps += 1
            emit("overview")

        transcript_cache = chunks.setdefault("transcript", {})
        slide_cache = chunks.setdefault("slides", {})
        cache_lock = threading.Lock()
        jobs = []
        if include_transcript:
            for chunk in transcript_chunks:
                jobs.append(
                    (
                        "transcript",
                        chunk["index"],
                        lambda chunk=chunk: self._analyze_transcript_chunk(
                            request,
                            chunk,
                            transcript_cache,
                            cache,
                            cache_path,
                            cache_lock,
                        ),
                    )
                )
        if include_slides:
            for batch_index, batch in enumerate(slide_batches, start=1):
                jobs.append(
                    (
                        "slides",
                        batch_index,
                        lambda batch=batch, batch_index=batch_index: self._analyze_slide_batch(
                            request,
                            batch,
                            batch_index,
                            overview,
                            slide_cache,
                            cache,
                            cache_path,
                            cache_lock,
                        ),
                    )
                )
        if include_resources:
            jobs.append(
                (
                    "resources",
                    1,
                    lambda: self._analyze_resources(
                        request,
                        overview,
                        chunks,
                        cache,
                        cache_path,
                        cache_lock,
                    ),
                )
            )

        transcript_results: Dict[int, Dict] = {}
        slide_results: Dict[int, Dict] = {}
        resources = {"resources": [], "warnings": []}
        if jobs:
            max_workers = min(self.max_concurrency, len(jobs))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(job): (kind, index) for kind, index, job in jobs}
                for future in as_completed(futures):
                    kind, index = futures[future]
                    payload, response = future.result()
                    _add_usage(usage, response)
                    warnings.extend(payload.get("warnings") or [])
                    if kind == "transcript":
                        transcript_results[index] = payload
                        step = f"transcript {index}/{len(transcript_chunks)}"
                    elif kind == "slides":
                        slide_results[index] = payload
                        step = f"slides {index}/{len(slide_batches)}"
                    else:
                        resources = payload
                        step = "resources"
                    completed_steps += 1
                    emit(step)

        formatted_parts = [
            str(transcript_results.get(chunk["index"], {}).get("formatted_transcript") or chunk["text"]).strip()
            for chunk in transcript_chunks
        ]
        slide_analysis = []
        for index in range(1, len(slide_batches) + 1):
            slide_analysis.extend(slide_results.get(index, {}).get("slide_analysis") or [])
        slide_analysis = _backfill_filtered_slide_analysis(request, slide_analysis, filtered_slides)

        formatted_transcript = "\n\n".join(part for part in formatted_parts if part).strip()
        if not include_transcript:
            formatted_transcript = request.transcript_text.strip()
        if not formatted_transcript:
            formatted_transcript = request.transcript_text.strip()
        if not include_slides:
            slide_analysis = _fallback_slide_analysis(request, request.slides)
        return AnalyzeLectureResponse(
            title=str(overview.get("title") or _fallback_title(request)).strip(),
            executive_summary=str(overview.get("executive_summary") or _fallback_summary(request.transcript_text)),
            formatted_transcript=formatted_transcript,
            outline=_normalize_outline(overview.get("outline") or _fallback_outline(request.slides)),
            slide_analysis=slide_analysis or _fallback_slide_analysis(request, request.slides),
            resources=list(resources.get("resources") or []),
            input_token_estimate=usage["input"],
            output_token_estimate=usage["output"],
            raw_response_id="chunked",
            warnings=_dedupe_warnings(warnings),
        )

    def _analyze_transcript_chunk(
        self,
        request: AnalyzeLectureRequest,
        chunk: Dict,
        transcript_cache: Dict,
        cache: Dict,
        cache_path: Optional[Path],
        cache_lock: threading.Lock,
    ) -> Tuple[Dict, Optional[object]]:
        key = str(chunk["index"])
        with cache_lock:
            payload = transcript_cache.get(key)
        if payload:
            return payload, None

        response = None
        try:
            payload, response = self._generate_json(
                _transcript_chunk_prompt(request, chunk),
                request,
                use_google_search=False,
                include_images=False,
                context=f"Gemini transcript chunk {key}",
                max_output_tokens=4096,
                response_schema=_transcript_chunk_response_schema(),
            )
        except ProviderResponseError as exc:
            payload = {
                "formatted_transcript": chunk["text"],
                "warnings": [f"Transcript chunk {key} used raw transcript fallback: {exc}"],
            }
        with cache_lock:
            transcript_cache[key] = payload
            _save_chunk_cache(cache_path, cache)
        return payload, response

    def _analyze_slide_batch(
        self,
        request: AnalyzeLectureRequest,
        batch: List[Dict],
        batch_index: int,
        overview: Dict,
        slide_cache: Dict,
        cache: Dict,
        cache_path: Optional[Path],
        cache_lock: threading.Lock,
    ) -> Tuple[Dict, Optional[object]]:
        key = _slide_batch_key(batch)
        with cache_lock:
            payload = slide_cache.get(key)
        if payload:
            return payload, None

        response = None
        try:
            payload, response = self._generate_json(
                _slide_batch_prompt(request, batch, overview),
                request,
                slides=batch,
                use_google_search=False,
                include_images=True,
                context=f"Gemini slide batch {batch_index}",
                max_output_tokens=8192,
                response_schema=_slide_batch_response_schema(),
            )
            payload["slide_analysis"] = _normalize_slide_analysis(
                payload.get("slide_analysis") or [],
                request,
                batch,
            )
        except ProviderResponseError as exc:
            payload = {
                "slide_analysis": _fallback_slide_analysis(request, batch),
                "warnings": [f"Slides {key} used transcript fallback: {exc}"],
            }
        with cache_lock:
            slide_cache[key] = payload
            _save_chunk_cache(cache_path, cache)
        return payload, response

    def _analyze_resources(
        self,
        request: AnalyzeLectureRequest,
        overview: Dict,
        chunks: Dict,
        cache: Dict,
        cache_path: Optional[Path],
        cache_lock: threading.Lock,
    ) -> Tuple[Dict, Optional[object]]:
        with cache_lock:
            resources = chunks.get("resources")
        if resources:
            return resources, None

        response = None
        try:
            resources, response = self._generate_json(
                _resources_prompt(request, overview),
                request,
                use_google_search=True,
                include_images=False,
                context="Gemini resources",
                max_output_tokens=4096,
                response_schema=_resources_response_schema(),
            )
            resources = _normalize_resources_payload(resources)
            if _complete_resource_count(resources) < MIN_RESOURCE_COUNT:
                retry_resources, retry_response = self._generate_json(
                    _resources_retry_prompt(request, overview, resources),
                    request,
                    use_google_search=True,
                    include_images=False,
                    context="Gemini resources retry",
                    max_output_tokens=4096,
                    response_schema=_resources_response_schema(),
                )
                retry_resources = _normalize_resources_payload(retry_resources)
                response = _combined_response(response, retry_response)
                if _complete_resource_count(retry_resources) >= _complete_resource_count(resources):
                    resources = retry_resources
            if _complete_resource_count(resources) < MIN_RESOURCE_COUNT:
                resources.setdefault("warnings", []).append(
                    f"Resources returned only {_complete_resource_count(resources)} complete item(s); expected 3-4."
                )
        except ProviderResponseError as exc:
            resources = {
                "resources": [],
                "warnings": [f"External resources skipped: {exc}"],
            }
        with cache_lock:
            chunks["resources"] = resources
            _save_chunk_cache(cache_path, cache)
        return resources, response

    def analyze_lecture_single_call(self, request: AnalyzeLectureRequest) -> AnalyzeLectureResponse:
        prompt = _build_prompt(request)
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=_contents_for_request(request, prompt, self._types),
                config=_generate_content_config(self._types),
            )
        except Exception as exc:
            raise _map_gemini_error(exc) from exc

        text = str(getattr(response, "text", "") or "").strip()
        if not text:
            raise _empty_response_error(response, "Gemini single-call enrichment")
        payload = _json_from_response_text(text)

        return _response_from_payload(payload, response)

    def _generate_json(
        self,
        prompt: str,
        request: AnalyzeLectureRequest,
        *,
        slides: Optional[List[Dict]] = None,
        use_google_search: bool,
        include_images: bool,
        context: str,
        max_output_tokens: Optional[int] = None,
        response_schema: Optional[Dict] = None,
    ) -> Tuple[Dict, object]:
        attempts = [True, False] if include_images else [False]
        last_error: Optional[Exception] = None
        for attempt_index, attempt_images in enumerate(attempts, start=1):
            contents = _contents_for_request(
                request,
                prompt,
                self._types,
                slides=slides,
                include_images=attempt_images,
            )
            try:
                response = _generate_content_with_retry(
                    self._client.models,
                    model=self.model,
                    contents=contents,
                    config=_generate_content_config(
                        self._types,
                        use_google_search=use_google_search,
                        max_output_tokens=max_output_tokens,
                        response_schema=response_schema,
                    ),
                )
            except ProviderRequestError:
                if use_google_search or not response_schema:
                    raise
                response = _generate_content_with_retry(
                    self._client.models,
                    model=self.model,
                    contents=contents,
                    config=_generate_content_config(
                        self._types,
                        use_google_search=use_google_search,
                        max_output_tokens=max_output_tokens,
                    ),
                )

            text = _response_text(response)
            if not text:
                last_error = _empty_response_error(
                    response,
                    f"{context}{' without images' if include_images and not attempt_images else ''}",
                )
                continue
            try:
                return _json_payload_from_response(response, text), response
            except ProviderResponseError as exc:
                if response_schema:
                    try:
                        repaired_payload, repair_response = self._repair_json_response(
                            text,
                            response_schema,
                            context=context,
                            max_output_tokens=max_output_tokens,
                        )
                        return repaired_payload, _combined_response(response, repair_response)
                    except ProviderResponseError as repair_exc:
                        last_error = ProviderResponseError(f"{exc}; repair failed: {repair_exc}")
                    except Exception as repair_exc:
                        last_error = ProviderResponseError(f"{exc}; repair failed: {repair_exc}")
                else:
                    last_error = exc
                if attempt_index >= len(attempts):
                    break
        if last_error:
            raise last_error
        raise ProviderResponseError(f"{context} did not return usable JSON.")

    def _repair_json_response(
        self,
        malformed_text: str,
        response_schema: Dict,
        *,
        context: str,
        max_output_tokens: Optional[int],
    ) -> Tuple[Dict, object]:
        prompt = _json_repair_prompt(malformed_text, response_schema, context=context)
        response = _generate_content_with_retry(
            self._client.models,
            model=self.model,
            contents=prompt,
            config=_generate_content_config(
                self._types,
                use_google_search=False,
                max_output_tokens=max_output_tokens,
                response_schema=response_schema,
            ),
        )
        text = _response_text(response)
        if not text:
            raise _empty_response_error(response, f"{context} JSON repair")
        return _json_payload_from_response(response, text), response


def build_manual_test_prompt(artifact: Dict, *, selected_slides: list = None) -> str:
    transcript = artifact.get("transcript", {})
    media = artifact.get("media", {})
    request = AnalyzeLectureRequest(
        lecture_id=artifact.get("lecture_id", "lecture"),
        transcript_text=transcript.get("text") or "",
        segments=list(transcript.get("segments") or []),
        slides=list(selected_slides if selected_slides is not None else artifact.get("slides") or []),
        duration_minutes=float(media.get("duration_seconds") or 0.0) / 60.0,
    )
    return _build_prompt(request, manual=True)


def response_schema_for_prompt() -> Dict:
    return _response_schema()


def _build_prompt(request: AnalyzeLectureRequest, manual: bool = False) -> str:
    segments = "\n".join(
        f"[{segment.get('id')}] {segment.get('start', 0):.1f}-{segment.get('end', 0):.1f}: {segment.get('text', '')}"
        for segment in request.segments
    )
    slides = "\n".join(
        f"[{slide.get('id')}] @{slide.get('timestamp_seconds', 0):.1f}s segments={slide.get('linked_segment_ids', [])}"
        for slide in request.slides
    )
    caption_note = _slide_caption_instructions()
    return f"""
You are creating study notes for a student from a processed lecture artifact.

Return JSON only. Do not include markdown fences.

Create:
- A clear lecture title.
- A concise executive summary.
- A lightly edited, human-readable transcript.
- A slide-aware outline.
- One slide analysis item for each slide listed.

## Resources
Use Google Search grounding to find 3-4 high-quality external resources relevant to the lecture's main topics.
Prefer peer-reviewed papers, WEF/McKinsey/industry reports, or reputable educational sources.
For each resource include: title, url, summary (one sentence), and source_quality ("high" or "medium").
Only include URLs confirmed by grounding. Do not invent or guess URLs.
{caption_note}

## Formatted transcript
Create formatted_transcript from the transcript text.
Make it human readable with paragraph breaks and very light editing only: fix obvious grammar errors,
typos, misspellings, capitalization, punctuation, and spacing.
Do not rewrite the instructor's meaning, add new content, summarize, or remove substantive details.

Expected top-level JSON keys:
- title
- executive_summary
- formatted_transcript
- outline
- slide_analysis
- resources
- warnings

Each slide_analysis item must include:
- slide_id
- descriptive_filename
- caption
- summary
- tags
- instructor_commentary

Lecture id: {request.lecture_id}
Duration minutes: {request.duration_minutes:.2f}

Transcript:
{request.transcript_text}

Transcript segments:
{segments}

Slides:
{slides}
""".strip()


def _slide_caption_instructions() -> str:
    return """
## Slide captions
For each slide, inspect the uploaded image and write a factual caption (1-2 sentences) describing exactly
what is visible: text on screen, diagram type, icons, layout, and visual hierarchy.
If the image is blank, a pre-roll frame, or a mid-animation transition with no instructional content, set caption to null.
Prefer the image over the transcript for what the slide shows.
Use the transcript for what the instructor says about it.
""".strip()


def _generate_content_config(
    types_module,
    *,
    use_google_search: bool = True,
    max_output_tokens: Optional[int] = None,
    response_schema: Optional[Dict] = None,
):
    kwargs = {}
    if max_output_tokens:
        kwargs["max_output_tokens"] = int(max_output_tokens)
    if use_google_search:
        kwargs["tools"] = [types_module.Tool(google_search=types_module.GoogleSearch())]
    elif response_schema:
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_json_schema"] = response_schema
    return types_module.GenerateContentConfig(**kwargs)


def _contents_for_request(
    request: AnalyzeLectureRequest,
    prompt: str,
    types_module,
    *,
    slides: Optional[List[Dict]] = None,
    include_images: bool = True,
):
    parts = [types_module.Part.from_text(text=prompt)]
    if include_images:
        parts.extend(_slide_image_parts(request, types_module, slides=slides))
    if len(parts) == 1:
        return prompt
    return [types_module.Content(role="user", parts=parts)]


def _slide_image_parts(request: AnalyzeLectureRequest, types_module, *, slides: Optional[List[Dict]] = None) -> List:
    if not request.lecture_dir:
        return []

    parts = []
    for slide in slides if slides is not None else request.slides:
        image_path = _slide_image_path(request.lecture_dir, slide)
        if not image_path:
            continue
        data, mime_type = _slide_image_bytes(image_path)
        if not mime_type:
            continue
        parts.append(
            types_module.Part.from_text(
                text=f"Slide {slide.get('id')}: {slide.get('filename') or image_path.name}"
            )
        )
        parts.append(types_module.Part.from_bytes(data=data, mime_type=mime_type))
    return parts


def _slide_image_path(lecture_dir: Path, slide: Dict) -> Optional[Path]:
    relative = slide.get("relative_path") or slide.get("filename")
    if not relative:
        return None
    path = lecture_dir / str(relative)
    return path if path.exists() and path.is_file() else None


def _mime_type_for_image(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return None


def _slide_image_bytes(path: Path) -> Tuple[bytes, Optional[str]]:
    # Keep the extracted PNG/JPEG untouched for the rendered study page, but
    # submit a smaller WebP copy to vision models to cut upload size and image tokens.
    cached = _cached_webp_path(path)
    if cached and cached.exists():
        try:
            return cached.read_bytes(), "image/webp"
        except OSError:
            pass

    try:
        image_module = importlib.import_module("PIL.Image")
        with image_module.open(path) as image:
            image.load()
            width, height = image.size
            if max(width, height) > SLIDE_IMAGE_MAX_EDGE:
                scale = SLIDE_IMAGE_MAX_EDGE / max(width, height)
                next_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                resampling = getattr(getattr(image_module, "Resampling", image_module), "LANCZOS")
                image = image.resize(next_size, resampling)
            buffer = io.BytesIO()
            image.convert("RGB").save(
                buffer,
                format="WEBP",
                quality=SLIDE_IMAGE_WEBP_QUALITY,
                method=4,
            )
            data = buffer.getvalue()
            if cached:
                try:
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    cached.write_bytes(data)
                except OSError:
                    pass
            return data, "image/webp"
    except Exception:
        mime_type = _mime_type_for_image(path)
        if not mime_type:
            return b"", None
        try:
            return path.read_bytes(), mime_type
        except OSError:
            return b"", None


def _cached_webp_path(path: Path) -> Optional[Path]:
    try:
        return path.parent / SLIDE_AI_CACHE_DIR / f"{path.stem}.webp"
    except Exception:
        return None


def _filter_slides_for_gemini(request: AnalyzeLectureRequest) -> Tuple[List[Dict], Dict[int, Dict]]:
    if not request.slides or not request.lecture_dir:
        return list(request.slides), {}

    kept: List[Tuple[Dict, int, Optional[Dict]]] = []
    filtered: Dict[int, Dict] = {}
    precursors_by_parent: Dict[int, List[int]] = {}
    slides_by_id: Dict[int, Dict] = {}

    for index, slide in enumerate(request.slides, start=1):
        slide_id = _slide_id(slide, fallback=index)
        slides_by_id[slide_id] = slide
        image_path = _slide_image_path(request.lecture_dir, slide)
        signature = _slide_signature(image_path) if image_path else None
        if signature is None:
            kept.append((slide, slide_id, None))
            continue

        if float(signature["entropy"]) < SLIDE_ENTROPY_MIN_BITS:
            filtered[slide_id] = {
                "status": "blank",
                "reason": "low_entropy",
                "parent_slide_id": None,
            }
            _set_slide_filter_status(slide, "skipped", reason="blank")
            continue

        if kept:
            _previous_slide, previous_id, previous_signature = kept[-1]
            if previous_signature is not None:
                distance = _hamming_distance(int(previous_signature["phash"]), int(signature["phash"]))
                if distance <= SLIDE_PHASH_DISTANCE_THRESHOLD:
                    kept[-1] = (slide, slide_id, signature)
                    earlier_precursors = precursors_by_parent.pop(previous_id, [])
                    for precursor_id in earlier_precursors:
                        filtered[precursor_id]["parent_slide_id"] = slide_id
                        _set_slide_filter_status(
                            slides_by_id[precursor_id],
                            "merged",
                            reason="near_duplicate",
                            parent_slide_id=slide_id,
                        )
                    filtered[previous_id] = {
                        "status": "build_precursor",
                        "reason": "near_duplicate",
                        "parent_slide_id": slide_id,
                    }
                    _set_slide_filter_status(
                        slides_by_id[previous_id],
                        "merged",
                        reason="near_duplicate",
                        parent_slide_id=slide_id,
                    )
                    precursors_by_parent[slide_id] = earlier_precursors + [previous_id]
                    continue

        kept.append((slide, slide_id, signature))

    for slide, _slide_id_value, _signature in kept:
        _set_slide_filter_status(slide, "sent_to_gemini")
    return [slide for slide, _slide_id_value, _signature in kept], filtered


def _set_slide_filter_status(
    slide: Dict,
    status: str,
    *,
    reason: Optional[str] = None,
    parent_slide_id: Optional[int] = None,
) -> None:
    payload = {"status": status}
    if reason:
        payload["reason"] = reason
    if parent_slide_id is not None:
        payload["parent_slide_id"] = parent_slide_id
    slide["filter_status"] = payload


def _slide_signature(path: Path) -> Optional[Dict]:
    try:
        image_module = importlib.import_module("PIL.Image")
        with image_module.open(path) as image:
            image.load()
            grayscale = image.convert("L")
            resampling = getattr(getattr(image_module, "Resampling", image_module), "LANCZOS")
            sampled = grayscale.resize((SLIDE_PHASH_SAMPLE_SIZE, SLIDE_PHASH_SAMPLE_SIZE), resampling)
            pixels = list(sampled.getdata())
            entropy = _image_entropy(pixels)
            phash = _perceptual_hash(pixels, SLIDE_PHASH_SAMPLE_SIZE)
            return {"entropy": entropy, "phash": phash}
    except Exception:
        return None


def _image_entropy(pixels: List[int]) -> float:
    if not pixels:
        return 0.0
    histogram = [0] * 256
    for pixel in pixels:
        histogram[int(pixel)] += 1
    total = float(len(pixels))
    entropy = 0.0
    for count in histogram:
        if count:
            probability = count / total
            entropy -= probability * math.log2(probability)
    return entropy


def _perceptual_hash(pixels: List[int], size: int) -> int:
    coefficients = []
    for u in range(SLIDE_PHASH_SIZE):
        for v in range(SLIDE_PHASH_SIZE):
            total = 0.0
            for y in range(size):
                row_offset = y * size
                cos_y = math.cos(((2 * y + 1) * u * math.pi) / (2 * size))
                for x in range(size):
                    cos_x = math.cos(((2 * x + 1) * v * math.pi) / (2 * size))
                    total += float(pixels[row_offset + x]) * cos_x * cos_y
            coefficients.append(total)

    comparable = coefficients[1:] or coefficients
    median = sorted(comparable)[len(comparable) // 2]
    value = 0
    for coefficient in comparable:
        value = (value << 1) | int(coefficient > median)
    return value


def _hamming_distance(left: int, right: int) -> int:
    return bin(left ^ right).count("1")


def _slide_filter_warning(filtered_slides: Dict[int, Dict]) -> Optional[str]:
    if not filtered_slides:
        return None
    blank_count = sum(1 for item in filtered_slides.values() if item.get("status") == "blank")
    duplicate_count = sum(1 for item in filtered_slides.values() if item.get("status") == "build_precursor")
    parts = []
    if blank_count:
        parts.append(f"{blank_count} blank/transition")
    if duplicate_count:
        parts.append(f"{duplicate_count} near-duplicate build frame")
    detail = ", ".join(parts) if parts else f"{len(filtered_slides)} filtered"
    return f"Slide pre-filter skipped {detail} before Gemini."


def _backfill_filtered_slide_analysis(
    request: AnalyzeLectureRequest,
    analyzed_items: List[Dict],
    filtered_slides: Dict[int, Dict],
) -> List[Dict]:
    if not filtered_slides:
        return analyzed_items

    by_id = {}
    for item in analyzed_items:
        slide_id = _coerce_int(item.get("slide_id"))
        if slide_id is not None:
            by_id[slide_id] = item

    fallback = {item["slide_id"]: item for item in _fallback_slide_analysis(request, request.slides)}
    ordered = []
    for index, slide in enumerate(request.slides, start=1):
        slide_id = _slide_id(slide, fallback=index)
        if slide_id in by_id:
            ordered.append(by_id[slide_id])
            continue

        filter_info = filtered_slides.get(slide_id)
        if not filter_info:
            if slide_id in fallback:
                ordered.append(fallback[slide_id])
            continue

        if filter_info.get("status") == "build_precursor":
            ordered.append(
                _build_precursor_slide_analysis(
                    slide_id,
                    int(filter_info.get("parent_slide_id") or slide_id),
                    by_id,
                    fallback,
                )
            )
        else:
            ordered.append(_blank_slide_analysis(slide_id, fallback))
    return ordered


def _build_precursor_slide_analysis(
    slide_id: int,
    parent_slide_id: int,
    analyzed_by_id: Dict[int, Dict],
    fallback_by_id: Dict[int, Dict],
) -> Dict:
    fallback_item = fallback_by_id[slide_id]
    parent_item = analyzed_by_id.get(parent_slide_id) or fallback_by_id.get(parent_slide_id) or fallback_item
    parent_summary = str(parent_item.get("summary") or "").strip()
    summary = f"Build precursor merged into slide {parent_slide_id} before Gemini analysis."
    if parent_summary:
        summary = f"{summary} {parent_summary}"
    tags = _dedupe_tags(list(parent_item.get("tags") or fallback_item["tags"]) + ["filtered", "build-precursor"])
    return {
        "slide_id": slide_id,
        "descriptive_filename": fallback_item["descriptive_filename"],
        "caption": parent_item.get("caption"),
        "summary": summary,
        "tags": tags,
        "instructor_commentary": str(parent_item.get("instructor_commentary") or fallback_item["instructor_commentary"]),
    }


def _blank_slide_analysis(slide_id: int, fallback_by_id: Dict[int, Dict]) -> Dict:
    fallback_item = fallback_by_id[slide_id]
    tags = _dedupe_tags(list(fallback_item["tags"]) + ["filtered", "blank"])
    return {
        "slide_id": slide_id,
        "descriptive_filename": fallback_item["descriptive_filename"],
        "caption": None,
        "summary": "Blank or transition slide filtered before Gemini.",
        "tags": tags,
        "instructor_commentary": "",
    }


def _dedupe_tags(tags: List[str]) -> List[str]:
    result = []
    seen = set()
    for tag in tags:
        clean = str(tag or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def _overview_prompt(request: AnalyzeLectureRequest) -> str:
    slides = "\n".join(
        f"- Slide {slide.get('id')}: @{slide.get('timestamp_seconds', 0):.1f}s segments={slide.get('linked_segment_ids', [])}"
        for slide in request.slides
    )
    return f"""
You are creating study notes for a student from a processed lecture transcript.

Return JSON only. Do not include markdown fences.

Create the high-level study structure only:
- title
- executive_summary
- outline
- warnings

The outline should be slide-aware. Each item must include id, heading, and slide_ids.
Use the transcript for concepts and the slide list for the slide_ids.

Expected JSON keys:
- title
- executive_summary
- outline
- warnings

Lecture id: {request.lecture_id}
Duration minutes: {request.duration_minutes:.2f}

Transcript:
{request.transcript_text}

Slides:
{slides}
""".strip()


def _transcript_chunk_prompt(request: AnalyzeLectureRequest, chunk: Dict) -> str:
    return f"""
You are lightly editing one chunk of a lecture transcript for readability.

Return JSON only. Do not include markdown fences.

Expected JSON keys:
- formatted_transcript
- warnings

Rules:
- Fix obvious grammar errors, typos, misspellings, capitalization, punctuation, repeated words, and spacing.
- Add paragraph breaks every 2-5 sentences or whenever the topic shifts.
- Use blank lines between paragraphs so the transcript is not one large blob.
- Keep the instructor's voice and the original order of ideas.
- Preserve names, technical terms, examples, and substantive details.
- Do not summarize.
- Do not add new ideas.
- Do not remove meaningful content.
- Do not add headings unless the speaker clearly introduces a new section.
- Preserve the instructor's meaning.

Lecture id: {request.lecture_id}
Chunk: {chunk["index"]} of {chunk["total"]}

Transcript chunk:
{chunk["text"]}
""".strip()


def _slide_batch_prompt(request: AnalyzeLectureRequest, slides: List[Dict], overview: Dict) -> str:
    slide_lines = "\n".join(
        f"- Slide {slide.get('id')}: {slide.get('filename')} @{slide.get('timestamp_seconds', 0):.1f}s"
        for slide in slides
    )
    context = "\n\n".join(
        f"Slide {slide.get('id')} nearby transcript:\n{_slide_context(request, slide)}" for slide in slides
    )
    return f"""
You are creating slide-by-slide study notes for a lecture.

Return JSON only. Do not include markdown fences.

Expected JSON keys:
- slide_analysis
- warnings

Each slide_analysis item must include:
- slide_id
- descriptive_filename
- caption
- summary
- tags
- instructor_commentary

For each slide, inspect the uploaded image if available and write a factual caption describing exactly what is visible.
If the image is blank, a pre-roll frame, or a mid-animation transition with no instructional content, set caption to null.
Use the nearby transcript for what the instructor says about the slide.
Return one slide_analysis item for every slide listed below.

Lecture title: {overview.get("title") or request.lecture_id}
Lecture summary: {overview.get("executive_summary") or ""}

Slides in this batch:
{slide_lines}

Nearby transcript:
{context}
""".strip()


def _resources_prompt(request: AnalyzeLectureRequest, overview: Dict) -> str:
    return f"""
Use Google Search grounding to find 3-4 high-quality external resources relevant to this lecture.

Return JSON only. Do not include markdown fences.
Keep every JSON string on one line. Do not put raw line breaks inside string values.

Expected JSON keys:
- resources
- warnings

For each resource include:
- title
- url
- summary
- source_quality ("high" or "medium")

Prefer peer-reviewed papers, WEF/McKinsey/industry reports, or reputable educational sources.
Only include URLs confirmed by grounding. Do not invent or guess URLs.

Lecture id: {request.lecture_id}
Lecture title: {overview.get("title") or request.lecture_id}
Lecture summary: {overview.get("executive_summary") or _fallback_summary(request.transcript_text)}
Transcript context:
{_resource_transcript_context(request.transcript_text)}
""".strip()


def _resources_retry_prompt(request: AnalyzeLectureRequest, overview: Dict, previous_payload: Dict) -> str:
    complete_count = _complete_resource_count(previous_payload)
    return f"""
Use Google Search grounding to find 3-4 complete, high-quality external resources relevant to this lecture.

The previous resource attempt returned only {complete_count} complete item(s). Search again and return only complete entries.

Return JSON only. Do not include markdown fences.
Keep every JSON string on one line. Do not put raw line breaks inside string values.

Each resource must include a non-empty:
- title
- url
- summary
- source_quality ("high" or "medium")

Only include URLs confirmed by grounding. Do not invent or guess URLs.
Prefer peer-reviewed papers, WEF/McKinsey/industry reports, or reputable educational sources.

Lecture id: {request.lecture_id}
Lecture title: {overview.get("title") or request.lecture_id}
Lecture summary: {overview.get("executive_summary") or _fallback_summary(request.transcript_text)}
Transcript context:
{_resource_transcript_context(request.transcript_text)}

Previous resource payload:
{json.dumps(previous_payload, sort_keys=True)}
""".strip()


def _json_repair_prompt(malformed_text: str, response_schema: Dict, *, context: str) -> str:
    return f"""
Convert the following {context} response into valid JSON that matches the schema.

Rules:
- Preserve all factual content, URLs, slide IDs, and transcript text from the original response.
- Do not add new facts.
- Do not include markdown fences.
- Return JSON only.

Schema:
{json.dumps(response_schema, sort_keys=True)}

Malformed response:
{malformed_text}
""".strip()


def _transcript_chunks(request: AnalyzeLectureRequest, max_chars: int = TRANSCRIPT_CHUNK_MAX_CHARS) -> List[Dict]:
    if request.segments:
        chunks = []
        current = []
        current_chars = 0
        for segment in request.segments:
            text = str(segment.get("text") or "").strip()
            if not text:
                continue
            line = f"[{segment.get('id')}] {segment.get('start', 0):.1f}-{segment.get('end', 0):.1f}: {text}"
            if current and current_chars + len(line) + 1 > max_chars:
                chunks.append("\n".join(current))
                current = []
                current_chars = 0
            current.append(line)
            current_chars += len(line) + 1
        if current:
            chunks.append("\n".join(current))
    else:
        text = str(request.transcript_text or "").strip()
        chunks = _split_text_by_chars(text, max_chars) if text else [""]
    total = max(1, len(chunks))
    return [{"index": index, "total": total, "text": chunk} for index, chunk in enumerate(chunks, start=1)]


def _split_text_by_chars(text: str, max_chars: int) -> List[str]:
    chunks = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining.strip())
            break
        cut = remaining.rfind(" ", 0, max_chars)
        if cut < max_chars // 2:
            cut = max_chars
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    return chunks or [""]


def _slide_batches(slides: List[Dict], batch_size: int = SLIDE_BATCH_SIZE) -> List[List[Dict]]:
    if not slides:
        return []
    return [slides[index : index + batch_size] for index in range(0, len(slides), batch_size)]


def _slide_batch_key(slides: List[Dict]) -> str:
    ids = [str(slide.get("id") or index) for index, slide in enumerate(slides, start=1)]
    return "-".join(ids)


def _slide_context(request: AnalyzeLectureRequest, slide: Dict) -> str:
    linked_ids = set(slide.get("linked_segment_ids") or [])
    pieces = []
    for segment in request.segments:
        if segment.get("id") in linked_ids:
            text = str(segment.get("text") or "").strip()
            if text:
                pieces.append(text)
    if not pieces:
        timestamp = float(slide.get("timestamp_seconds") or 0.0)
        for segment in request.segments:
            start = float(segment.get("start") or 0.0)
            end = float(segment.get("end") or start)
            if abs(start - timestamp) <= 90 or start <= timestamp <= end + 90:
                text = str(segment.get("text") or "").strip()
                if text:
                    pieces.append(text)
            if len(" ".join(pieces)) > 1800:
                break
    context = " ".join(pieces).strip()
    return context[:2200] if context else "No nearby transcript commentary was found."


def _fallback_title(request: AnalyzeLectureRequest) -> str:
    return request.lecture_id.replace("_", " ").strip() or "Untitled Lecture"


def _fallback_summary(text: str) -> str:
    clean = " ".join(str(text or "").split())
    if not clean:
        return "This lecture was processed, but no transcript text was available for summarization."
    if len(clean) <= 700:
        return clean
    return clean[:697].rsplit(" ", 1)[0] + "..."


def _resource_transcript_context(text: str, max_chars: int = 5000) -> str:
    clean = " ".join(str(text or "").split())
    if not clean:
        return "No transcript context is available."
    if len(clean) <= max_chars:
        return clean

    head_chars = max_chars // 3
    middle_chars = max_chars // 3
    tail_chars = max_chars - head_chars - middle_chars
    midpoint = max(0, (len(clean) - middle_chars) // 2)
    return "\n".join(
        [
            f"Opening: {_trim_to_word(clean[:head_chars])}",
            f"Middle: {_trim_to_word(clean[midpoint:midpoint + middle_chars])}",
            f"Closing: {_trim_to_word(clean[-tail_chars:])}",
        ]
    )


def _trim_to_word(text: str) -> str:
    clean = str(text or "").strip()
    if " " not in clean:
        return clean
    return clean.rsplit(" ", 1)[0].strip()


def _fallback_outline(slides: List[Dict]) -> List[Dict]:
    if not slides:
        return [{"id": 1, "heading": "Lecture overview", "slide_ids": []}]
    outline = []
    for index, batch in enumerate(_slide_batches(slides, 6), start=1):
        ids = [_slide_id(slide, fallback=offset) for offset, slide in enumerate(batch, start=1)]
        heading = f"Slides {ids[0]}-{ids[-1]} discussion" if len(ids) > 1 else f"Slide {ids[0]} discussion"
        outline.append({"id": index, "heading": heading, "slide_ids": ids})
    return outline


def _fallback_slide_analysis(request: AnalyzeLectureRequest, slides: List[Dict]) -> List[Dict]:
    if not slides:
        return [
            {
                "slide_id": 1,
                "descriptive_filename": None,
                "caption": None,
                "summary": "No slides were extracted for this lecture.",
                "tags": ["lecture"],
                "instructor_commentary": _fallback_summary(request.transcript_text),
            }
        ]
    items = []
    for index, slide in enumerate(slides, start=1):
        slide_id = _slide_id(slide, fallback=index)
        commentary = _slide_context(request, slide)
        summary = commentary if len(commentary) <= 320 else commentary[:317].rsplit(" ", 1)[0] + "..."
        items.append(
            {
                "slide_id": slide_id,
                "descriptive_filename": f"slide_{slide_id:04d}_study_note.png",
                "caption": None,
                "summary": summary,
                "tags": ["lecture", "slide", f"slide-{slide_id}"],
                "instructor_commentary": commentary,
            }
        )
    return items


def _normalize_outline(outline: List[Dict]) -> List[Dict]:
    normalized = []
    for index, item in enumerate(outline or [], start=1):
        slide_ids = [_coerce_int(value) for value in item.get("slide_ids") or []]
        normalized.append(
            {
                "id": _coerce_int(item.get("id")) or index,
                "heading": str(item.get("heading") or item.get("title") or f"Section {index}").strip(),
                "slide_ids": [value for value in slide_ids if value is not None],
            }
        )
    return normalized


def _normalize_slide_analysis(items: List[Dict], request: AnalyzeLectureRequest, slides: List[Dict]) -> List[Dict]:
    by_id = {}
    for item in items or []:
        slide_id = _coerce_int(item.get("slide_id"))
        if slide_id is not None:
            by_id[slide_id] = item

    normalized = []
    fallback = {item["slide_id"]: item for item in _fallback_slide_analysis(request, slides)}
    for index, slide in enumerate(slides, start=1):
        slide_id = _slide_id(slide, fallback=index)
        item = by_id.get(slide_id) or fallback[slide_id]
        fallback_item = fallback[slide_id]
        normalized.append(
            {
                "slide_id": slide_id,
                "descriptive_filename": item.get("descriptive_filename") or fallback_item["descriptive_filename"],
                "caption": item.get("caption"),
                "summary": str(item.get("summary") or fallback_item["summary"]),
                "tags": list(item.get("tags") or fallback_item["tags"]),
                "instructor_commentary": str(
                    item.get("instructor_commentary") or fallback_item["instructor_commentary"]
                ),
            }
        )
    return normalized


def _normalize_resources_payload(payload: Dict) -> Dict:
    resources = []
    warnings = list(payload.get("warnings") or [])
    seen_urls = set()
    incomplete_count = 0

    for item in payload.get("resources") or []:
        normalized = _normalize_resource_item(item)
        if not normalized:
            incomplete_count += 1
            continue
        url_key = normalized["url"].strip().lower()
        if url_key in seen_urls:
            continue
        seen_urls.add(url_key)
        resources.append(normalized)

    if incomplete_count:
        warnings.append(f"{incomplete_count} incomplete resource item(s) were discarded.")
    return {"resources": resources[:MAX_RESOURCE_COUNT], "warnings": warnings}


def _normalize_resource_item(item: Dict) -> Optional[Dict]:
    title = str(item.get("title") or "").strip()
    url = str(item.get("url") or "").strip()
    summary = str(item.get("summary") or "").strip()
    quality = str(item.get("source_quality") or "").strip().lower()
    if not title or not url or not summary:
        return None
    if not (url.startswith("http://") or url.startswith("https://")):
        return None
    if quality not in {"high", "medium"}:
        quality = "medium"
    return {
        "title": title,
        "url": url,
        "summary": summary,
        "source_quality": quality,
    }


def _complete_resource_count(payload: Dict) -> int:
    return len(payload.get("resources") or [])


def _slide_id(slide: Dict, *, fallback: int) -> int:
    return _coerce_int(slide.get("id")) or fallback


def _coerce_int(value) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        digits = "".join(character for character in value if character.isdigit())
        if digits:
            return int(digits)
    return None


def _dedupe_warnings(warnings: List[str]) -> List[str]:
    seen = set()
    result = []
    for warning in warnings:
        text = str(warning or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _overview_response_schema() -> Dict:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "executive_summary": {"type": "string"},
            "outline": {"type": "array", "items": _outline_item_schema()},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "executive_summary", "outline", "warnings"],
    }


def _transcript_chunk_response_schema() -> Dict:
    return {
        "type": "object",
        "properties": {
            "formatted_transcript": {"type": "string"},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["formatted_transcript", "warnings"],
    }


def _slide_batch_response_schema() -> Dict:
    return {
        "type": "object",
        "properties": {
            "slide_analysis": {"type": "array", "items": _slide_analysis_item_schema()},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["slide_analysis", "warnings"],
    }


def _outline_item_schema() -> Dict:
    return {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "heading": {"type": "string"},
            "slide_ids": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["id", "heading", "slide_ids"],
    }


def _slide_analysis_item_schema() -> Dict:
    return {
        "type": "object",
        "properties": {
            "slide_id": {"type": "integer"},
            "descriptive_filename": {"type": "string", "nullable": True},
            "caption": {"type": "string", "nullable": True},
            "summary": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "instructor_commentary": {"type": "string"},
        },
        "required": [
            "slide_id",
            "descriptive_filename",
            "caption",
            "summary",
            "tags",
            "instructor_commentary",
        ],
    }


def _resource_item_schema() -> Dict:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "url": {"type": "string"},
            "summary": {"type": "string"},
            "source_quality": {"type": "string", "enum": ["high", "medium"]},
        },
        "required": ["title", "url", "summary", "source_quality"],
    }


def _resources_response_schema() -> Dict:
    return {
        "type": "object",
        "properties": {
            "resources": {"type": "array", "items": _resource_item_schema()},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["resources", "warnings"],
    }


def _response_schema() -> Dict:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "executive_summary": {"type": "string"},
            "formatted_transcript": {"type": "string"},
            "outline": {"type": "array", "items": _outline_item_schema()},
            "slide_analysis": {"type": "array", "items": _slide_analysis_item_schema()},
            "resources": {"type": "array", "items": _resource_item_schema()},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "title",
            "executive_summary",
            "formatted_transcript",
            "outline",
            "slide_analysis",
            "resources",
            "warnings",
        ],
    }


def _response_from_payload(payload: Dict, response) -> AnalyzeLectureResponse:
    for key in ["title", "executive_summary", "formatted_transcript", "outline", "slide_analysis"]:
        if key not in payload:
            raise ProviderResponseError(f"Gemini response is missing '{key}'.")
    usage = getattr(response, "usage_metadata", None)
    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
    return AnalyzeLectureResponse(
        title=str(payload["title"]).strip() or "Untitled Lecture",
        executive_summary=str(payload["executive_summary"]).strip(),
        formatted_transcript=str(payload["formatted_transcript"]).strip(),
        outline=list(payload["outline"] or []),
        slide_analysis=list(payload["slide_analysis"] or []),
        resources=list(payload.get("resources") or []),
        input_token_estimate=input_tokens,
        output_token_estimate=output_tokens,
        raw_response_id=None,
        warnings=list(payload.get("warnings") or []),
    )


def _load_chunk_cache(cache_path: Optional[Path], request: AnalyzeLectureRequest, model: str) -> Dict:
    base = {
        "schema_version": "1.0.0",
        "lecture_id": request.lecture_id,
        "model": model,
        "updated_at": utc_now_iso(),
        "chunks": {},
        "warnings": [],
    }
    if not cache_path or not cache_path.exists():
        return base
    try:
        existing = load_json(cache_path)
    except Exception:
        return base
    if existing.get("lecture_id") != request.lecture_id or existing.get("model") != model:
        return base
    existing.setdefault("chunks", {})
    existing.setdefault("warnings", [])
    return existing


def _save_chunk_cache(cache_path: Optional[Path], cache: Dict) -> None:
    if not cache_path:
        return
    cache["updated_at"] = utc_now_iso()
    try:
        save_json(cache_path, cache)
    except Exception:
        pass


def _add_usage(usage: Dict[str, int], response) -> None:
    metadata = getattr(response, "usage_metadata", None)
    if not metadata:
        return
    usage["input"] += int(getattr(metadata, "prompt_token_count", 0) or 0)
    usage["output"] += int(getattr(metadata, "candidates_token_count", 0) or 0)


class _UsageTotals:
    def __init__(self, prompt_token_count: int, candidates_token_count: int) -> None:
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count


class _CombinedResponse:
    def __init__(self, responses: List[object]) -> None:
        prompt_tokens = 0
        candidate_tokens = 0
        for response in responses:
            metadata = getattr(response, "usage_metadata", None)
            prompt_tokens += int(getattr(metadata, "prompt_token_count", 0) or 0) if metadata else 0
            candidate_tokens += int(getattr(metadata, "candidates_token_count", 0) or 0) if metadata else 0
        self.usage_metadata = _UsageTotals(prompt_tokens, candidate_tokens)


def _combined_response(*responses) -> object:
    valid = [response for response in responses if response is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    return _CombinedResponse(valid)


def _generate_content_with_retry(models, **kwargs):
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            return models.generate_content(**kwargs)
        except Exception as exc:
            mapped = _map_gemini_error(exc)
            if not isinstance(mapped, ProviderTransientError) or attempt >= MAX_RETRY_ATTEMPTS:
                raise mapped from exc
            last_error = mapped
            time.sleep(_retry_delay_seconds(exc, attempt))
    if last_error:
        raise last_error
    raise ProviderTransientError("Gemini request failed before a response was returned.")


def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return max(0.0, min(60.0, retry_after))
    base = min(30.0, 2.0 ** max(0, attempt - 1))
    return base + random.uniform(0.0, 0.25)


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    headers = _exception_headers(exc)
    if not headers:
        return None
    value = None
    for key in ("retry-after", "Retry-After"):
        try:
            value = headers.get(key)
        except AttributeError:
            value = headers.get(key) if isinstance(headers, dict) else None
        if value:
            break
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        retry_at = parsedate_to_datetime(str(value))
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    return max(0.0, retry_at.timestamp() - time.time())


def _exception_headers(exc: Exception):
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        return headers
    return getattr(exc, "headers", None)


def _bounded_concurrency(value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_MAX_CONCURRENCY
    return max(1, min(12, parsed))


def _response_text(response) -> str:
    try:
        return str(getattr(response, "text", "") or "").strip()
    except Exception as exc:
        raise ProviderResponseError(f"Gemini response text could not be read: {exc}") from exc


def _empty_response_error(response, context: str) -> ProviderResponseError:
    suffix = _response_debug_suffix(response)
    return ProviderResponseError(f"{context} returned an empty response{suffix}.")


def _response_debug_suffix(response) -> str:
    details = []
    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback:
        details.append(f"prompt_feedback={_short_debug_value(prompt_feedback)}")
    candidates = list(getattr(response, "candidates", []) or [])
    finish_reasons = []
    safety = []
    for candidate in candidates:
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason:
            finish_reasons.append(str(finish_reason))
        safety_ratings = getattr(candidate, "safety_ratings", None)
        if safety_ratings:
            safety.append(_short_debug_value(safety_ratings))
    if finish_reasons:
        details.append(f"finish_reason={','.join(finish_reasons)}")
    if safety:
        details.append(f"safety={';'.join(safety)}")
    usage = getattr(response, "usage_metadata", None)
    if usage:
        prompt_tokens = getattr(usage, "prompt_token_count", None)
        output_tokens = getattr(usage, "candidates_token_count", None)
        details.append(f"tokens={prompt_tokens}/{output_tokens}")
    return f" ({'; '.join(details)})" if details else ""


def _short_debug_value(value) -> str:
    text = str(value).replace("\n", " ")
    return text[:220] + ("..." if len(text) > 220 else "")


def _emit_progress(progress_callback: Optional[Callable[[Dict], None]], **payload) -> None:
    if not progress_callback:
        return
    try:
        progress_callback(payload)
    except Exception:
        pass


def _json_from_response_text(text: str) -> Dict:
    cleaned = str(text or "").strip()
    candidates = [cleaned]
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        candidates.append("\n".join(lines).strip())
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(cleaned[start : end + 1])
    candidates.extend(_json_repair_candidates(candidates))

    last_error = None
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(payload, dict):
            raise ProviderResponseError("Gemini JSON response must be an object.")
        return payload
    raise ProviderResponseError(f"Gemini returned invalid JSON: {last_error}")


def _json_payload_from_response(response, text: str) -> Dict:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return parsed
    return _json_from_response_text(text)


def _json_repair_candidates(candidates: List[str]) -> List[str]:
    repaired = []
    for candidate in candidates:
        if not candidate:
            continue
        control_fixed = _replace_raw_control_chars_in_strings(candidate)
        trailing_fixed = _remove_trailing_json_commas(control_fixed)
        repaired.extend([control_fixed, trailing_fixed])
    return repaired


def _replace_raw_control_chars_in_strings(text: str) -> str:
    result = []
    in_string = False
    escaped = False
    for character in text:
        if escaped:
            result.append(character)
            escaped = False
            continue
        if character == "\\" and in_string:
            result.append(character)
            escaped = True
            continue
        if character == '"':
            result.append(character)
            in_string = not in_string
            continue
        if in_string and ord(character) < 32:
            result.append(" ")
            continue
        result.append(character)
    return "".join(result)


def _remove_trailing_json_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _map_gemini_error(exc: Exception) -> Exception:
    message = str(exc)
    lowered = message.lower()
    if "api key" in lowered or "permission" in lowered or "unauth" in lowered:
        return ProviderAuthError(message)
    if "model" in lowered and ("not found" in lowered or "unsupported" in lowered):
        return ProviderRequestError(message)
    if "400" in lowered or "invalid" in lowered:
        return ProviderRequestError(message)
    if "block" in lowered or "safety" in lowered:
        return ProviderResponseError(f"Gemini safety settings blocked the response: {message}")
    return ProviderTransientError(message)
