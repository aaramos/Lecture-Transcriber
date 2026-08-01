import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
import urllib.parse

from .errors import LectureProcessorError
from .profiles import DEFAULT_PROFILE_ID, normalize_profile_id

DEFAULT_LM_STUDIO_BASE_URL = "http://192.168.86.22:1234/v1"


def default_local_text_base_url() -> str:
    return (
        os.environ.get("MLX_TEXT_SERVER_URL")
        or os.environ.get("LM_STUDIO_BASE_URL")
        or os.environ.get("OLLAMA_BASE_URL")
        or DEFAULT_LM_STUDIO_BASE_URL
    )


def default_local_vision_base_url() -> str:
    return (
        os.environ.get("MLX_VISION_SERVER_URL")
        or os.environ.get("LM_STUDIO_BASE_URL")
        or os.environ.get("OLLAMA_BASE_URL")
        or DEFAULT_LM_STUDIO_BASE_URL
    )


class RecordingSpeed(str, Enum):
    NORMAL = "1x"
    DOUBLE = "2x"


class AudioQuality(str, Enum):
    FAST = "fast"
    HIGH = "high"


class AudioEnhancementMode(str, Enum):
    NONE = "none"
    HYBRID = "hybrid"
    STRONG = "strong"


class SlideSensitivity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TranscriptionEngine(str, Enum):
    AUTO = "auto"
    WHISPER_CPP = "whisper-cpp"
    FASTER_WHISPER = "faster-whisper"
    PARAKEET_MLX = "parakeet-mlx"
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
    LM_STUDIO = "lm-studio"


class AIModelProvider(str, Enum):
    GEMINI = "gemini"
    MLX_TEXT = "mlx-text"
    MLX_VISION = "mlx-vision"
    LOCAL_STUB = "local-stub"
    OFF = "off"


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
    audio_enhancement: AudioEnhancementMode = AudioEnhancementMode.NONE
    slide_sensitivity: SlideSensitivity = SlideSensitivity.MEDIUM
    transcription_profile: str = DEFAULT_PROFILE_ID
    transcription_engine: TranscriptionEngine = TranscriptionEngine.FASTER_WHISPER
    transcription_quality: TranscriptionQuality = TranscriptionQuality.ACCURATE
    whisper_model: str = "medium.en"
    whisper_cpp_model_dir: str = ""
    require_whisper_cpp_coreml: bool = False
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    deep_filter_path: str = ""
    ffmpeg_hwaccel: FfmpegHwAccel = FfmpegHwAccel.AUTO
    slide_backend: SlideBackend = SlideBackend.AUTO
    apple_silicon: bool = False
    ai_provider: AIProviderName = AIProviderName.NONE
    ai_model: str = ""
    ai_overview_provider: AIModelProvider = AIModelProvider.MLX_TEXT
    ai_overview_model: str = ""
    ai_transcript_provider: AIModelProvider = AIModelProvider.MLX_TEXT
    ai_transcript_model: str = ""
    ai_slides_provider: AIModelProvider = AIModelProvider.MLX_VISION
    ai_slides_model: str = ""
    ai_resources_provider: AIModelProvider = AIModelProvider.MLX_TEXT
    ai_resources_model: str = ""
    mlx_text_base_url: str = field(default_factory=default_local_text_base_url)
    mlx_vision_base_url: str = field(default_factory=default_local_vision_base_url)
    mlx_request_timeout_seconds: int = 120
    mlx_disable_thinking: bool = False
    skip_ai_enrichment_reason: str = ""
    gemini_max_concurrency: int = 3
    render_html: bool = True
    skip_files: Tuple[str, ...] = ()
    control_file: Optional[Path] = None

    def validate(self) -> None:
        if not self.input_dir.exists():
            raise LectureProcessorError(f"Input path does not exist: {self.input_dir}")
        if not self.input_dir.is_dir() and not self.input_dir.is_file():
            raise LectureProcessorError(f"Input path is not a file or folder: {self.input_dir}")
        if self.concurrent_files < 1 or self.concurrent_files > 8:
            raise LectureProcessorError("--concurrent must be between 1 and 8")
        if self.min_duration_seconds < 0:
            raise LectureProcessorError("--min-duration must be zero or greater")
        if self.gemini_max_concurrency < 1 or self.gemini_max_concurrency > 12:
            raise LectureProcessorError("--ai-max-concurrency must be between 1 and 12")
        if self.mlx_request_timeout_seconds < 10 or self.mlx_request_timeout_seconds > 600:
            raise LectureProcessorError("--mlx-timeout must be between 10 and 600 seconds")
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
        self._validate_mlx_urls()

    def _validate_mlx_urls(self) -> None:
        for url_field in ("mlx_text_base_url", "mlx_vision_base_url"):
            url = getattr(self, url_field)
            if url and not self._is_valid_url(url):
                raise LectureProcessorError(f"Invalid URL for {url_field}: {url}")

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        try:
            result = urllib.parse.urlparse(url)
            return all([result.scheme in ("http", "https"), result.netloc])
        except Exception:
            return False

    def ai_step_provider(self, step: str) -> AIModelProvider:
        return getattr(self, f"ai_{step}_provider")

    def ai_step_model(self, step: str) -> str:
        explicit = str(getattr(self, f"ai_{step}_model") or "").strip()
        provider = self.ai_step_provider(step)
        if explicit:
            return explicit
        if provider is AIModelProvider.GEMINI:
            return self.ai_model or "gemini-2.5-flash"
        if provider in (AIModelProvider.MLX_TEXT, AIModelProvider.MLX_VISION):
            return "default"
        if provider is AIModelProvider.LOCAL_STUB:
            return "local-stub-v0"
        return "off"

    @property
    def ai_uses_gemini(self) -> bool:
        if self.ai_provider is not AIProviderName.GEMINI:
            return False
        return any(
            self.ai_step_provider(step) is AIModelProvider.GEMINI
            for step in ("overview", "transcript", "slides", "resources")
        )

    @property
    def ai_uses_local_stub(self) -> bool:
        return any(
            self.ai_step_provider(step) is AIModelProvider.LOCAL_STUB
            for step in ("overview", "transcript", "slides", "resources")
        )

    @property
    def ai_uses_mlx(self) -> bool:
        return any(
            self.ai_step_provider(step) in (AIModelProvider.MLX_TEXT, AIModelProvider.MLX_VISION)
            for step in ("overview", "transcript", "slides", "resources")
        )

    @property
    def ai_model_routing(self) -> dict:
        return {
            step: {
                "provider": self.ai_step_provider(step).value,
                "model": self.ai_step_model(step),
            }
            for step in ("overview", "transcript", "slides", "resources")
        }

    @property
    def normalized_timestamp_scale(self) -> float:
        if self.recording_speed is RecordingSpeed.DOUBLE and not self.save_normalized_video:
            return 0.5
        return 1.0
