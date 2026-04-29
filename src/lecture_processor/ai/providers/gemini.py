import importlib
import json
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
TRANSCRIPT_CHUNK_MAX_CHARS = 12000
SLIDE_BATCH_SIZE = 5


class GeminiProvider:
    def __init__(self, api_key: str, model: str = DEFAULT_GEMINI_MODEL) -> None:
        if not api_key:
            raise ProviderAuthError("Gemini API key is required.")
        self.model = model or DEFAULT_GEMINI_MODEL
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
    ) -> AnalyzeLectureResponse:
        cache = _load_chunk_cache(cache_path, request, self.model)
        warnings = list(cache.get("warnings") or [])
        usage = {"input": 0, "output": 0}
        transcript_chunks = _transcript_chunks(request)
        slide_batches = _slide_batches(request.slides)
        total_steps = 2 + len(transcript_chunks) + len(slide_batches)
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
        overview = chunks.get("overview")
        if not overview:
            try:
                overview, response = self._generate_json(
                    _overview_prompt(request),
                    request,
                    use_google_search=False,
                    include_images=False,
                    context="Gemini overview",
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
        warnings.extend(overview.get("warnings") or [])
        completed_steps += 1
        emit("overview")

        transcript_cache = chunks.setdefault("transcript", {})
        formatted_parts = []
        for chunk in transcript_chunks:
            key = str(chunk["index"])
            payload = transcript_cache.get(key)
            if not payload:
                try:
                    payload, response = self._generate_json(
                        _transcript_chunk_prompt(request, chunk),
                        request,
                        use_google_search=False,
                        include_images=False,
                        context=f"Gemini transcript chunk {key}",
                    )
                    _add_usage(usage, response)
                except ProviderResponseError as exc:
                    payload = {
                        "formatted_transcript": chunk["text"],
                        "warnings": [f"Transcript chunk {key} used raw transcript fallback: {exc}"],
                    }
                transcript_cache[key] = payload
                _save_chunk_cache(cache_path, cache)
            formatted_parts.append(str(payload.get("formatted_transcript") or chunk["text"]).strip())
            warnings.extend(payload.get("warnings") or [])
            completed_steps += 1
            emit(f"transcript {chunk['index']}/{len(transcript_chunks)}")

        slide_cache = chunks.setdefault("slides", {})
        slide_analysis = []
        for batch_index, batch in enumerate(slide_batches, start=1):
            key = _slide_batch_key(batch)
            payload = slide_cache.get(key)
            if not payload:
                try:
                    payload, response = self._generate_json(
                        _slide_batch_prompt(request, batch, overview),
                        request,
                        slides=batch,
                        use_google_search=False,
                        include_images=True,
                        context=f"Gemini slide batch {batch_index}",
                    )
                    _add_usage(usage, response)
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
                slide_cache[key] = payload
                _save_chunk_cache(cache_path, cache)
            slide_analysis.extend(payload.get("slide_analysis") or [])
            warnings.extend(payload.get("warnings") or [])
            completed_steps += 1
            emit(f"slides {batch_index}/{len(slide_batches)}")

        resources = chunks.get("resources")
        if not resources:
            try:
                resources, response = self._generate_json(
                    _resources_prompt(request, overview),
                    request,
                    use_google_search=True,
                    include_images=False,
                    context="Gemini resources",
                )
                _add_usage(usage, response)
            except ProviderResponseError as exc:
                resources = {
                    "resources": [],
                    "warnings": [f"External resources skipped: {exc}"],
                }
            chunks["resources"] = resources
            _save_chunk_cache(cache_path, cache)
        warnings.extend(resources.get("warnings") or [])
        completed_steps += 1
        emit("resources")

        formatted_transcript = "\n\n".join(part for part in formatted_parts if part).strip()
        if not formatted_transcript:
            formatted_transcript = request.transcript_text.strip()
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
    ) -> Tuple[Dict, object]:
        attempts = [include_images]
        if include_images:
            attempts.append(False)
        else:
            attempts.append(False)
        last_error: Optional[Exception] = None
        for attempt_index, attempt_images in enumerate(attempts, start=1):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=_contents_for_request(
                        request,
                        prompt,
                        self._types,
                        slides=slides,
                        include_images=attempt_images,
                    ),
                    config=_generate_content_config(self._types, use_google_search=use_google_search),
                )
            except Exception as exc:
                raise _map_gemini_error(exc) from exc

            text = _response_text(response)
            if not text:
                last_error = _empty_response_error(
                    response,
                    f"{context}{' without images' if include_images and not attempt_images else ''}",
                )
                continue
            try:
                return _json_from_response_text(text), response
            except ProviderResponseError as exc:
                last_error = exc
                if attempt_index >= len(attempts):
                    break
        if last_error:
            raise last_error
        raise ProviderResponseError(f"{context} did not return usable JSON.")


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


def _generate_content_config(types_module, *, use_google_search: bool = True):
    if not use_google_search:
        return types_module.GenerateContentConfig()
    return types_module.GenerateContentConfig(
        tools=[types_module.Tool(google_search=types_module.GoogleSearch())],
    )


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
        mime_type = _mime_type_for_image(image_path)
        if not mime_type:
            continue
        parts.append(
            types_module.Part.from_text(
                text=f"Slide {slide.get('id')}: {slide.get('filename') or image_path.name}"
            )
        )
        parts.append(types_module.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime_type))
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
- Fix obvious grammar errors, typos, misspellings, capitalization, punctuation, and spacing.
- Add paragraph breaks where helpful.
- Do not summarize.
- Do not add new ideas.
- Do not remove substantive details.
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


def _response_schema() -> Dict:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "executive_summary": {"type": "string"},
            "formatted_transcript": {"type": "string"},
            "outline": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "heading": {"type": "string"},
                        "slide_ids": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["id", "heading", "slide_ids"],
                },
            },
            "slide_analysis": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "slide_id": {"type": "integer"},
                        "descriptive_filename": {"type": ["string", "null"]},
                        "caption": {"type": ["string", "null"]},
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
                },
            },
            "resources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "summary": {"type": "string"},
                        "source_quality": {"type": "string", "enum": ["high", "medium"]},
                    },
                    "required": ["title", "url", "summary", "source_quality"],
                },
            },
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

    last_error = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(payload, dict):
            raise ProviderResponseError("Gemini JSON response must be an object.")
        return payload
    raise ProviderResponseError(f"Gemini returned invalid JSON: {last_error}")


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
