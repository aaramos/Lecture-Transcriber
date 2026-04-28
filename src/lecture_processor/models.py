from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


class FileStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    STOPPED = "stopped"


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration_seconds: float
    avg_frame_rate: float = 0.0
    real_frame_rate: float = 0.0
    is_vfr: bool = False
    has_audio: bool = True

    @property
    def intended_frame_rate(self) -> float:
        if self.avg_frame_rate > 0:
            return self.avg_frame_rate
        if self.real_frame_rate > 0:
            return self.real_frame_rate
        return 30.0


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str

    def scaled(self, scale: float) -> "TranscriptSegment":
        return TranscriptSegment(
            start=self.start * scale,
            end=self.end * scale,
            text=self.text,
        )


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    segments: List[TranscriptSegment] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len([w for w in self.text.split() if w.strip()])

    def scaled(self, scale: float) -> "TranscriptResult":
        if scale == 1.0:
            return self
        return TranscriptResult(
            text=self.text,
            segments=[segment.scaled(scale) for segment in self.segments],
        )


@dataclass
class FileResult:
    source: Path
    output_dir: Path
    status: FileStatus
    duration_seconds: float = 0.0
    normalized_duration_seconds: Optional[float] = None
    elapsed_seconds: float = 0.0
    word_count: int = 0
    slide_count: int = 0
    failure_step: Optional[str] = None
    message: str = ""
    lecture_json_path: Optional[Path] = None
    html_path: Optional[Path] = None
    enriched: bool = False
    rendered: bool = False
    title: Optional[str] = None
    short_summary: Optional[str] = None


@dataclass(frozen=True)
class BatchSummary:
    attempted: int
    completed: int
    failed: int
    skipped: int
    results: List[FileResult]
    stopped: int = 0
