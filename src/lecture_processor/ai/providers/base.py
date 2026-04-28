from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    display_name: str
    available_models: List[str]
    default_model: str
    docs_url: str


@dataclass(frozen=True)
class AnalyzeLectureRequest:
    lecture_id: str
    transcript_text: str
    segments: List[Dict]
    slides: List[Dict]
    duration_minutes: float
    prompt_version: str = "v1"
    lecture_dir: Optional[Path] = None


@dataclass(frozen=True)
class AnalyzeLectureResponse:
    title: str
    executive_summary: str
    outline: List[Dict]
    slide_analysis: List[Dict]
    formatted_transcript: str = ""
    resources: List[Dict] = field(default_factory=list)
    input_token_estimate: int = 0
    output_token_estimate: int = 0
    raw_response_id: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class AIProviderError(Exception):
    pass


class ProviderAuthError(AIProviderError):
    pass


class ProviderRequestError(AIProviderError):
    pass


class ProviderResponseError(AIProviderError):
    pass


class ProviderTransientError(AIProviderError):
    pass


class AIProvider(Protocol):
    @classmethod
    def info(cls) -> ProviderInfo:
        ...

    def test_connection(self) -> None:
        ...

    def analyze_lecture(self, request: AnalyzeLectureRequest) -> AnalyzeLectureResponse:
        ...
