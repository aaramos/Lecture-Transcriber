import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
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
    disabled_resources,
    local_formatted_transcript,
    local_overview,
    local_slide_analysis,
)

DEFAULT_LOCAL_MODEL = "default"
DEFAULT_TEXT_BASE_URL = "http://localhost:8001/v1"
DEFAULT_VISION_BASE_URL = "http://localhost:8000/v1"
DEFAULT_TIMEOUT_SECONDS = 120
TRANSCRIPT_CHUNK_CHARS = 6000
SLIDE_BATCH_SIZE = 5


class MLXTextProvider:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_TEXT_BASE_URL,
        model: str = DEFAULT_LOCAL_MODEL,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.model = model or DEFAULT_LOCAL_MODEL
        self.base_url = base_url or DEFAULT_TEXT_BASE_URL
        self.timeout_seconds = _bounded_timeout(timeout_seconds)
        self._client = _OpenAICompatibleClient(self.base_url, self.model, self.timeout_seconds)

    @classmethod
    def info(cls) -> ProviderInfo:
        return ProviderInfo(
            name="mlx-text",
            display_name="Local MLX Text",
            available_models=[DEFAULT_LOCAL_MODEL],
            default_model=DEFAULT_LOCAL_MODEL,
            docs_url="",
        )

    def test_connection(self) -> None:
        self._client.chat_json(
            [{"role": "user", "content": "Reply with JSON: {\"ok\": true}"}],
            max_tokens=32,
            temperature=0.0,
            context="MLX text connection test",
        )

    def analyze_lecture(self, request: AnalyzeLectureRequest) -> AnalyzeLectureResponse:
        overview = self.analyze_overview(request)
        transcript = self.analyze_transcript(request)
        slides = local_slide_analysis(request, "mlx-text")
        resources = self.analyze_resources(request, overview=overview)
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
            raw_response_id="mlx-text",
            warnings=_dedupe_warnings(warnings),
        )

    def analyze_overview(self, request: AnalyzeLectureRequest) -> Dict:
        prompt = f"""
Return JSON only. Create a lecture overview with these fields:
- title: short, specific lecture title
- executive_summary: one concise paragraph
- outline: array of objects with id, heading, slide_ids
- warnings: array of strings

Lecture id: {request.lecture_id}
Duration minutes: {request.duration_minutes:.2f}

Transcript:
{_trim_text(request.transcript_text, 24000)}

Slides:
{_slide_index_text(request.slides)}
""".strip()
        try:
            payload, usage = self._client.chat_json(
                _messages(prompt, system="You create concise study-note overviews from lecture transcripts."),
                max_tokens=2048,
                temperature=0.2,
                context="MLX overview",
            )
            normalized = _normalize_overview(payload, request)
            _attach_usage(normalized, usage, prompt, payload)
            normalized.setdefault("warnings", []).append(f"Overview used local MLX text model: {self.model}")
            return normalized
        except Exception as exc:
            fallback = local_overview(request, self.model)
            fallback.setdefault("warnings", []).append(f"MLX overview fallback used: {exc}")
            return fallback

    def analyze_transcript(self, request: AnalyzeLectureRequest) -> Dict:
        chunks = _transcript_chunks(request.transcript_text)
        parts: List[str] = []
        warnings: List[str] = [f"Transcript editing used local MLX text model: {self.model}"]
        input_tokens = 0
        output_tokens = 0
        for index, chunk in enumerate(chunks, start=1):
            prompt = f"""
Return JSON only with fields:
- formatted_transcript: lightly edited transcript text
- warnings: array of strings

Edit only obvious punctuation, capitalization, spacing, and transcription glitches.
Do not summarize, remove details, or add new content.

Chunk {index}/{len(chunks)}:
{chunk}
""".strip()
            try:
                payload, usage = self._client.chat_json(
                    _messages(prompt, system="You lightly clean lecture transcripts without changing meaning."),
                    max_tokens=4096,
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
        }

    def analyze_resources(self, request: AnalyzeLectureRequest, *, overview: Optional[Dict] = None) -> Dict:
        title = str((overview or {}).get("title") or request.lecture_id)
        summary = str((overview or {}).get("executive_summary") or "")
        prompt = f"""
Return JSON only with fields:
- resources: array of up to 4 objects with title, url, summary, source_quality
- warnings: array of strings

You do not have web-search grounding in this local mode. Only include URLs that are broadly well-known and
high confidence. If you cannot verify a URL, return an empty resources array and explain in warnings.

Lecture title: {title}
Summary: {_trim_text(summary, 2000)}
Transcript excerpt: {_trim_text(request.transcript_text, 8000)}
""".strip()
        try:
            payload, usage = self._client.chat_json(
                _messages(prompt, system="You suggest conservative external study resources."),
                max_tokens=2048,
                temperature=0.2,
                context="MLX resources",
            )
            normalized = _normalize_resources(payload)
            _attach_usage(normalized, usage, prompt, payload)
            normalized.setdefault("warnings", []).append(
                f"Resources used local MLX text model: {self.model}; results are not web-grounded."
            )
            return normalized
        except Exception as exc:
            fallback = disabled_resources()
            fallback.setdefault("warnings", []).append(f"MLX resources fallback used: {exc}")
            return fallback

    def analyze_slides(self, request: AnalyzeLectureRequest) -> Dict:
        fallback = local_slide_analysis(request, self.model)
        fallback.setdefault("warnings", []).append(
            "Slide images require the Local MLX Vision route; Local MLX Text used transcript-only slide notes."
        )
        return fallback


class MLXVisionProvider(MLXTextProvider):
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_VISION_BASE_URL,
        model: str = DEFAULT_LOCAL_MODEL,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(base_url=base_url, model=model, timeout_seconds=timeout_seconds)

    @classmethod
    def info(cls) -> ProviderInfo:
        return ProviderInfo(
            name="mlx-vision",
            display_name="Local MLX Vision",
            available_models=[DEFAULT_LOCAL_MODEL],
            default_model=DEFAULT_LOCAL_MODEL,
            docs_url="",
        )

    def test_connection(self) -> None:
        self._client.chat_json(
            [{"role": "user", "content": "Reply with JSON: {\"ok\": true}"}],
            max_tokens=32,
            temperature=0.0,
            context="MLX vision connection test",
        )

    def analyze_slides(self, request: AnalyzeLectureRequest) -> Dict:
        if not request.slides:
            return local_slide_analysis(request, self.model)
        slide_analysis: List[Dict] = []
        warnings = [f"Slide analysis used local MLX vision model: {self.model}"]
        input_tokens = 0
        output_tokens = 0
        for batch in _slide_batches(request.slides):
            prompt = f"""
Return JSON only with field slide_analysis.
Return one slide_analysis item for every slide listed.
Each item must include: slide_id, descriptive_filename, caption, summary, tags, instructor_commentary.

Describe exactly what is visible in the slide image. Use the nearby transcript only for instructor commentary.

Slides:
{_slide_index_text(batch)}

Nearby transcript:
{_batch_transcript_context(request, batch)}
""".strip()
            try:
                payload, usage = self._client.chat_json(
                    _vision_messages(request, batch, prompt),
                    max_tokens=4096,
                    temperature=0.3,
                    context="MLX slide batch",
                )
                normalized = _normalize_slide_analysis(payload.get("slide_analysis") or [], request, batch)
                slide_analysis.extend(normalized)
                warnings.extend(payload.get("warnings") or [])
                input_tokens += usage.input_tokens or _estimate_tokens(prompt)
                output_tokens += usage.output_tokens or _estimate_tokens(json.dumps(payload))
            except Exception as exc:
                fallback = local_slide_analysis(
                    AnalyzeLectureRequest(
                        lecture_id=request.lecture_id,
                        transcript_text=request.transcript_text,
                        segments=request.segments,
                        slides=batch,
                        duration_minutes=request.duration_minutes,
                        lecture_dir=request.lecture_dir,
                    ),
                    self.model,
                )
                slide_analysis.extend(fallback["slide_analysis"])
                warnings.append(f"Slide batch used local fallback: {exc}")
        return {
            "slide_analysis": slide_analysis or local_slide_analysis(request, self.model)["slide_analysis"],
            "warnings": _dedupe_warnings(warnings),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


@dataclass(frozen=True)
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0


class _OpenAICompatibleClient:
    def __init__(self, base_url: str, model: str, timeout_seconds: int) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.model = model or DEFAULT_LOCAL_MODEL
        self.timeout_seconds = _bounded_timeout(timeout_seconds)

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
            raise ProviderResponseError(f"{context} returned invalid JSON from MLX: {exc}") from exc

    def chat_text(self, messages: List[Dict], *, max_tokens: int, temperature: float, context: str) -> Tuple[str, _Usage]:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": "Bearer not-needed",
                "Content-Type": "application/json",
            },
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
                f"{context} could not reach the MLX server at {self.base_url}. "
                "Start the local server or update the MLX server URL in Settings."
            ) from exc
        except TimeoutError as exc:
            raise ProviderTransientError(f"{context} timed out after {self.timeout_seconds}s at {self.base_url}") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(f"{context} returned non-JSON HTTP response from MLX.") from exc
        choices = payload.get("choices") or []
        if not choices:
            raise ProviderResponseError(f"{context} returned no choices from MLX.")
        message = choices[0].get("message") or {}
        text = str(message.get("content") or "").strip()
        if not text:
            raise ProviderResponseError(f"{context} returned an empty response from MLX.")
        usage = payload.get("usage") or {}
        return text, _Usage(
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
        )


def _messages(prompt: str, *, system: str) -> List[Dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


def _vision_messages(request: AnalyzeLectureRequest, slides: List[Dict], prompt: str) -> List[Dict]:
    content = [{"type": "text", "text": prompt}]
    if request.lecture_dir:
        for slide in slides:
            path = _slide_image_path(request.lecture_dir, slide)
            if not path:
                continue
            data, mime_type = _slide_image_bytes(path)
            if not data or not mime_type:
                continue
            encoded = base64.standard_b64encode(data).decode("utf-8")
            content.append({"type": "text", "text": f"Slide {slide.get('id')}: {path.name}"})
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}})
    return [
        {"role": "system", "content": "You analyze lecture slide images and return compact JSON."},
        {"role": "user", "content": content},
    ]


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
        "resources": resources[:4],
        "warnings": list(payload.get("warnings") or []),
    }


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


def _slide_batches(slides: List[Dict]) -> List[List[Dict]]:
    return [slides[index : index + SLIDE_BATCH_SIZE] for index in range(0, len(slides), SLIDE_BATCH_SIZE)]


def _slide_index_text(slides: List[Dict]) -> str:
    if not slides:
        return "No slides."
    return "\n".join(
        f"- slide_id={int(slide.get('id') or index)} timestamp={float(slide.get('timestamp_seconds') or 0.0):.1f}s "
        f"segments={slide.get('linked_segment_ids') or []}"
        for index, slide in enumerate(slides, start=1)
    )


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
        return _trim_text(request.transcript_text, 4000)
    return _trim_text("\n".join(parts), 6000)


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


def _normalize_base_url(base_url: str) -> str:
    return str(base_url or "").strip().rstrip("/") or DEFAULT_TEXT_BASE_URL


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
