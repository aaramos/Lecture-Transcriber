from typing import Dict, List

from .base import AnalyzeLectureRequest, AnalyzeLectureResponse, ProviderInfo


class LocalStubProvider:
    def __init__(self, model: str = "local-stub-v0") -> None:
        self.model = model or "local-stub-v0"

    @classmethod
    def info(cls) -> ProviderInfo:
        return ProviderInfo(
            name="local-stub",
            display_name="Local Stub",
            available_models=["local-stub-v0"],
            default_model="local-stub-v0",
            docs_url="",
        )

    def test_connection(self) -> None:
        return None

    def analyze_lecture(self, request: AnalyzeLectureRequest) -> AnalyzeLectureResponse:
        overview = local_overview(request, self.model)
        transcript = local_formatted_transcript(request, self.model)
        slides = local_slide_analysis(request, self.model)
        resources = local_resources(request, self.model)
        return AnalyzeLectureResponse(
            title=overview["title"],
            executive_summary=overview["executive_summary"],
            outline=overview["outline"],
            formatted_transcript=transcript["formatted_transcript"],
            slide_analysis=slides["slide_analysis"],
            resources=resources["resources"],
            input_token_estimate=max(1, len(request.transcript_text.split())),
            output_token_estimate=300,
            raw_response_id="local-stub",
            warnings=[
                "Local model stub was used. Replace this route with a real local model when configuration is ready."
            ],
            step_token_usage={
                "overview": _token_usage_entry(overview),
                "transcript_cleanup": _token_usage_entry(transcript),
                "slide_analysis": _token_usage_entry(slides),
                "resource_formatter": _token_usage_entry(resources),
            },
        )


def local_overview(request: AnalyzeLectureRequest, model: str) -> Dict:
    return {
        "title": _title_from_transcript(request),
        "executive_summary": (
            "Local overview stub. This placeholder keeps the experimental routing path working "
            "until a real local overview model is configured."
        ),
        "outline": _outline_from_slides(request),
        "warnings": [f"Overview used local stub model: {model}"],
        "input_tokens": _estimate_tokens(request.transcript_text),
        "output_tokens": 80,
    }


def local_formatted_transcript(request: AnalyzeLectureRequest, model: str) -> Dict:
    return {
        "formatted_transcript": " ".join(str(request.transcript_text or "").split()),
        "warnings": [f"Transcript editing used local stub model: {model}"],
        "input_tokens": _estimate_tokens(request.transcript_text),
        "output_tokens": _estimate_tokens(request.transcript_text),
    }


def local_slide_analysis(request: AnalyzeLectureRequest, model: str) -> Dict:
    slides = request.slides or []
    analysis = [_slide_note(request, slide, index) for index, slide in enumerate(slides, start=1)]
    if not analysis:
        analysis = [
            {
                "slide_id": 1,
                "descriptive_filename": None,
                "caption": None,
                "summary": "No slides were extracted for this lecture.",
                "tags": ["lecture"],
                "instructor_commentary": "Local slide stub had no slide images to analyze.",
            }
        ]
    return {
        "slide_analysis": analysis,
        "warnings": [f"Slide analysis used local stub model: {model}"],
        "input_tokens": max(1, len(slides) * 40),
        "output_tokens": max(1, len(analysis) * 80),
    }


def local_resources(_request: AnalyzeLectureRequest, model: str) -> Dict:
    return {
        "resources": [],
        "warnings": [f"Resources used local stub model: {model}; no web resources were generated."],
        "input_tokens": 40,
        "output_tokens": 20,
    }


def disabled_overview(request: AnalyzeLectureRequest) -> Dict:
    return {
        "title": _title_from_transcript(request),
        "executive_summary": "Overview generation was disabled for this experimental route.",
        "outline": _outline_from_slides(request),
        "warnings": ["Overview route is off."],
    }


def disabled_formatted_transcript(request: AnalyzeLectureRequest) -> Dict:
    return {
        "formatted_transcript": " ".join(str(request.transcript_text or "").split()),
        "warnings": ["Transcript editing route is off; raw transcript text was preserved."],
    }


def disabled_slide_analysis(request: AnalyzeLectureRequest) -> Dict:
    return {
        "slide_analysis": local_slide_analysis(request, "off")["slide_analysis"],
        "warnings": ["Slide analysis route is off; placeholder slide notes were generated."],
    }


def disabled_resources() -> Dict:
    return {
        "resources": [],
        "warnings": ["Resources route is off."],
    }


def _title_from_transcript(request: AnalyzeLectureRequest) -> str:
    words = [word.strip(".,:;!?()[]{}") for word in request.transcript_text.split() if word.strip()]
    if words:
        return " ".join(words[:8]).title()
    return request.lecture_id.replace("_", " ").strip().title() or "Untitled Lecture"


def _token_usage_entry(payload: Dict) -> Dict[str, int]:
    input_tokens = max(0, int(payload.get("input_tokens") or 0))
    output_tokens = max(0, int(payload.get("output_tokens") or 0))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _estimate_tokens(text: str) -> int:
    return max(1, len(str(text or "")) // 4)


def _outline_from_slides(request: AnalyzeLectureRequest) -> List[Dict]:
    slides = request.slides or []
    if not slides:
        return [{"id": 1, "heading": "Lecture overview", "slide_ids": []}]
    return [
        {
            "id": index,
            "heading": f"Slide {int(slide.get('id') or index)} discussion",
            "slide_ids": [int(slide.get("id") or index)],
        }
        for index, slide in enumerate(slides, start=1)
    ]


def _slide_note(request: AnalyzeLectureRequest, slide: Dict, index: int) -> Dict:
    slide_id = int(slide.get("id") or index)
    commentary = _commentary_for_slide(request, slide)
    description = str(slide.get("description") or "").strip()
    title = str(slide.get("title") or "").strip()
    summary_source = " ".join(part for part in [title, description] if part) or commentary
    summary = summary_source if len(summary_source) <= 260 else summary_source[:257].rsplit(" ", 1)[0] + "..."
    tags = ["lecture", "local-stub", f"slide-{slide_id}"]
    for key in ("build_stage", "layout"):
        value = str(slide.get(key) or "").strip()
        if value:
            tags.append(value)
    return {
        "slide_id": slide_id,
        "descriptive_filename": f"slide_{slide_id:04d}_local_stub.png",
        "caption": description or None,
        "summary": summary,
        "tags": tags,
        "instructor_commentary": commentary,
    }


def _commentary_for_slide(request: AnalyzeLectureRequest, slide: Dict) -> str:
    segment_ids = set(slide.get("linked_segment_ids") or [])
    parts = [
        str(segment.get("text") or "").strip()
        for segment in request.segments
        if segment.get("id") in segment_ids and str(segment.get("text") or "").strip()
    ]
    return " ".join(parts) or "Local slide stub placeholder. Real local slide analysis is not configured yet."
