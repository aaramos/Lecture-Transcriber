from .base import AnalyzeLectureRequest, AnalyzeLectureResponse, ProviderInfo


class MockProvider:
    def __init__(self, api_key: str = "", model: str = "mock-study-notes-v1") -> None:
        self.model = model or "mock-study-notes-v1"

    @classmethod
    def info(cls) -> ProviderInfo:
        return ProviderInfo(
            name="mock",
            display_name="Mock AI",
            available_models=["mock-study-notes-v1"],
            default_model="mock-study-notes-v1",
            docs_url="",
        )

    def test_connection(self) -> None:
        return None

    def analyze_lecture(self, request: AnalyzeLectureRequest) -> AnalyzeLectureResponse:
        title = _title_from_transcript(request)
        summary = _summary_from_transcript(request.transcript_text)
        outline = _outline_from_slides(request)
        slide_analysis = [_slide_analysis(request, slide) for slide in request.slides]
        if not slide_analysis:
            slide_analysis = [
                {
                    "slide_id": 1,
                    "descriptive_filename": None,
                    "caption": None,
                    "summary": "No slides were extracted for this lecture.",
                    "tags": ["lecture"],
                    "instructor_commentary": summary,
                }
            ]
        return AnalyzeLectureResponse(
            title=title,
            executive_summary=summary,
            outline=outline,
            slide_analysis=slide_analysis,
            formatted_transcript=_formatted_transcript(request.transcript_text),
            resources=[],
            input_token_estimate=max(1, len(request.transcript_text.split())),
            output_token_estimate=250 + (60 * len(slide_analysis)),
            warnings=["Mock enrichment was used. Replace with Gemini for real AI-generated study notes."],
        )


def _title_from_transcript(request: AnalyzeLectureRequest) -> str:
    words = [word.strip(".,:;!?()[]{}") for word in request.transcript_text.split() if word.strip()]
    if not words:
        return request.lecture_id.replace("_", " ").strip() or "Untitled Lecture"
    return " ".join(words[:8]).title()


def _summary_from_transcript(text: str) -> str:
    clean = " ".join(text.split())
    if not clean:
        return "This lecture was processed, but no transcript text was available for summarization."
    if len(clean) <= 420:
        return clean
    return clean[:417].rsplit(" ", 1)[0] + "..."


def _formatted_transcript(text: str) -> str:
    return " ".join(str(text or "").split())


def _outline_from_slides(request: AnalyzeLectureRequest) -> list:
    if not request.slides:
        return [{"id": 1, "heading": "Lecture overview", "slide_ids": []}]
    outline = []
    for index, slide in enumerate(request.slides, start=1):
        outline.append(
            {
                "id": index,
                "heading": f"Slide {slide.get('id', index)} discussion",
                "slide_ids": [int(slide.get("id", index))],
            }
        )
    return outline


def _slide_analysis(request: AnalyzeLectureRequest, slide: dict) -> dict:
    segment_ids = slide.get("linked_segment_ids") or []
    commentary_parts = []
    for segment in request.segments:
        if segment.get("id") in segment_ids:
            text = str(segment.get("text") or "").strip()
            if text:
                commentary_parts.append(text)
    commentary = " ".join(commentary_parts).strip() or "No nearby transcript commentary was found."
    summary = commentary if len(commentary) <= 280 else commentary[:277].rsplit(" ", 1)[0] + "..."
    slide_id = int(slide.get("id") or 1)
    return {
        "slide_id": slide_id,
        "descriptive_filename": f"slide_{slide_id:04d}_study_note.png",
        "caption": None,
        "summary": summary,
        "tags": ["lecture", "slide", f"slide-{slide_id}"],
        "instructor_commentary": commentary,
    }
