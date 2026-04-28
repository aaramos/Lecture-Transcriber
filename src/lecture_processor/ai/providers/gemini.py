import importlib
import json
from typing import Dict

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
        except ImportError as exc:
            raise DependencyMissingError(
                "Gemini support is not installed. Install with: python3 -m pip install -e '.[ai]'"
            ) from exc
        self._client = genai.Client(api_key=api_key)

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
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": _response_schema(),
                },
            )
        except Exception as exc:
            raise _map_gemini_error(exc) from exc

        text = str(getattr(response, "text", "") or "").strip()
        if not text:
            raise ProviderResponseError("Gemini returned an empty response.")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(f"Gemini returned invalid JSON: {exc}") from exc

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
    manual_note = """
You may inspect any uploaded slide images, but keep the JSON shape exactly the same.
If an uploaded image conflicts with the transcript, prefer the image for slide descriptions and the transcript for instructor commentary.
""".strip() if manual else ""
    return f"""
You are creating study notes for a student from a processed lecture artifact.

Return JSON only. Do not include markdown fences.

Create:
- A clear lecture title.
- A concise executive summary.
- A slide-aware outline.
- One slide analysis item for each slide listed.

Do not invent external resources or URLs. Return an empty resources array unless a URL appears in the transcript.
{manual_note}

Expected top-level JSON keys:
- title
- executive_summary
- outline
- slide_analysis
- resources
- warnings

Each slide_analysis item must include:
- slide_id
- descriptive_filename
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


def _response_schema() -> Dict:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "executive_summary": {"type": "string"},
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
                        "summary": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "instructor_commentary": {"type": "string"},
                    },
                    "required": ["slide_id", "summary", "tags", "instructor_commentary"],
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
                        "source_quality": {"type": "string"},
                    },
                    "required": ["title", "url", "summary", "source_quality"],
                },
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "executive_summary", "outline", "slide_analysis", "resources", "warnings"],
    }


def _response_from_payload(payload: Dict, response) -> AnalyzeLectureResponse:
    for key in ["title", "executive_summary", "outline", "slide_analysis"]:
        if key not in payload:
            raise ProviderResponseError(f"Gemini response is missing '{key}'.")
    usage = getattr(response, "usage_metadata", None)
    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
    return AnalyzeLectureResponse(
        title=str(payload["title"]).strip() or "Untitled Lecture",
        executive_summary=str(payload["executive_summary"]).strip(),
        outline=list(payload["outline"] or []),
        slide_analysis=list(payload["slide_analysis"] or []),
        resources=list(payload.get("resources") or []),
        input_token_estimate=input_tokens,
        output_token_estimate=output_tokens,
        raw_response_id=None,
        warnings=list(payload.get("warnings") or []),
    )


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
