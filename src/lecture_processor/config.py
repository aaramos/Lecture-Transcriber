from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

from .errors import LectureProcessorError
from .profiles import DEFAULT_PROFILE_ID, normalize_profile_id


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
    WHISPER_CPP = "whisper-cpp"
    FASTER_WHISPER = "faster-whisper"
    OPENAI_WHISPER = "openai-whisper"
    NONE = "none"


class TranscriptionQuality(str, Enum):
    ACCURATE = "accurate"
    BALANCED = "balanced"
    FAST = "fast"


class FfmpegHwAccel(str, Enum):
    AUTO = "auto"
    NONE = "none"
    VIDEOTOOLBOX = "videotoolbox"


class SlideBackend(str, Enum):
    AUTO = "auto"
    FFMPEG = "ffmpeg"
    OPENCV = "opencv"


class AIProviderName(str, Enum):
    NONE = "none"
    MOCK = "mock"
    GEMINI = "gemini"


@dataclass(frozen=True)
class BatchConfig:
    input_dir: Path
    output_dir: Path
    recording_speed: RecordingSpeed = RecordingSpeed.NORMAL
    confirm_normalization: bool = False
    concurrent_files: int = 4
    min_duration_seconds: float = 60.0
    save_normalized_video: bool = True
    audio_quality: AudioQuality = AudioQuality.HIGH
    slide_sensitivity: SlideSensitivity = SlideSensitivity.MEDIUM
    transcription_profile: str = DEFAULT_PROFILE_ID
    transcription_engine: TranscriptionEngine = TranscriptionEngine.FASTER_WHISPER
    transcription_quality: TranscriptionQuality = TranscriptionQuality.ACCURATE
    whisper_model: str = "medium.en"
    whisper_cpp_model_dir: str = ""
    require_whisper_cpp_coreml: bool = False
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    ffmpeg_hwaccel: FfmpegHwAccel = FfmpegHwAccel.AUTO
    slide_backend: SlideBackend = SlideBackend.AUTO
    apple_silicon: bool = False
    ai_provider: AIProviderName = AIProviderName.NONE
    ai_model: str = ""
    render_html: bool = True
    skip_files: Tuple[str, ...] = ()
    control_file: Optional[Path] = None

    def validate(self) -> None:
        if not self.input_dir.exists():
            raise LectureProcessorError(f"Input folder does not exist: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise LectureProcessorError(f"Input path is not a folder: {self.input_dir}")
        if self.concurrent_files < 1 or self.concurrent_files > 8:
            raise LectureProcessorError("--concurrent must be between 1 and 8")
        if self.min_duration_seconds < 0:
            raise LectureProcessorError("--min-duration must be zero or greater")
        try:
            object.__setattr__(self, "transcription_profile", normalize_profile_id(self.transcription_profile))
        except ValueError as exc:
            raise LectureProcessorError(str(exc)) from exc
        if self.require_whisper_cpp_coreml and self.transcription_engine not in (
            TranscriptionEngine.AUTO,
            TranscriptionEngine.WHISPER_CPP,
        ):
            raise LectureProcessorError("--require-whisper-cpp-coreml requires whisper-cpp or auto transcription")
        if self.recording_speed is RecordingSpeed.DOUBLE and not self.confirm_normalization:
            raise LectureProcessorError(
                "2x normalization can create half-speed output if the files are already 1x. "
                "Re-run with --confirm-normalization after confirming the batch was recorded at 2x."
            )
        if self.ai_provider is AIProviderName.GEMINI and not self.ai_model:
            object.__setattr__(self, "ai_model", "gemini-2.5-flash")

    @property
    def normalized_timestamp_scale(self) -> float:
        if self.recording_speed is RecordingSpeed.DOUBLE and not self.save_normalized_video:
            return 0.5
        return 1.0
