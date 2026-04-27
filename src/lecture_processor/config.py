from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .errors import LectureProcessorError


class RecordingSpeed(str, Enum):
    NORMAL = "1x"
    DOUBLE = "2x"


class AudioQuality(str, Enum):
    FAST = "fast"
    HIGH = "high"


class SlideSensitivity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TranscriptionEngine(str, Enum):
    AUTO = "auto"
    FASTER_WHISPER = "faster-whisper"
    OPENAI_WHISPER = "openai-whisper"
    NONE = "none"


@dataclass(frozen=True)
class BatchConfig:
    input_dir: Path
    output_dir: Path
    recording_speed: RecordingSpeed = RecordingSpeed.NORMAL
    confirm_normalization: bool = False
    concurrent_files: int = 4
    min_duration_seconds: float = 60.0
    save_normalized_video: bool = True
    audio_quality: AudioQuality = AudioQuality.FAST
    slide_sensitivity: SlideSensitivity = SlideSensitivity.MEDIUM
    transcription_engine: TranscriptionEngine = TranscriptionEngine.AUTO
    whisper_model: str = "large-v3"
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"

    def validate(self) -> None:
        if not self.input_dir.exists():
            raise LectureProcessorError(f"Input folder does not exist: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise LectureProcessorError(f"Input path is not a folder: {self.input_dir}")
        if self.concurrent_files < 1 or self.concurrent_files > 8:
            raise LectureProcessorError("--concurrent must be between 1 and 8")
        if self.min_duration_seconds < 0:
            raise LectureProcessorError("--min-duration must be zero or greater")
        if self.recording_speed is RecordingSpeed.DOUBLE and not self.confirm_normalization:
            raise LectureProcessorError(
                "2x normalization can create half-speed output if the files are already 1x. "
                "Re-run with --confirm-normalization after confirming the batch was recorded at 2x."
            )

    @property
    def normalized_timestamp_scale(self) -> float:
        if self.recording_speed is RecordingSpeed.DOUBLE and not self.save_normalized_video:
            return 0.5
        return 1.0
