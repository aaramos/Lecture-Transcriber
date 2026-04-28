import importlib
import json
from pathlib import Path
from typing import Dict, List, Optional

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

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"


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
                "gemini-2.5-flash-lite",
                "gemini-2.5-flash",
                "gemini-2.5-pro",
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
            raise ProviderResponseError("Gemini returned an empty response.")
        payload = _json_from_response_text(text)

        return _response_from_payload(payload, response)


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


def _generate_content_config(types_module):
    return types_module.GenerateContentConfig(
        tools=[types_module.Tool(google_search=types_module.GoogleSearch())],
    )


def _contents_for_request(request: AnalyzeLectureRequest, prompt: str, types_module):
    parts = [types_module.Part.from_text(text=prompt)]
    parts.extend(_slide_image_parts(request, types_module))
    if len(parts) == 1:
        return prompt
    return [types_module.Content(role="user", parts=parts)]


def _slide_image_parts(request: AnalyzeLectureRequest, types_module) -> List:
    if not request.lecture_dir:
        return []

    parts = []
    for slide in request.slides:
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
