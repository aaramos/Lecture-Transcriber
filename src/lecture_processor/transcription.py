import importlib
import os
import threading
from pathlib import Path
from typing import Optional, Protocol

from .config import TranscriptionEngine
from .errors import DependencyMissingError, ProcessingError
from .models import TranscriptResult, TranscriptSegment


class Transcriber(Protocol):
    def transcribe(self, media_path: Path) -> TranscriptResult:
        ...


class NullTranscriber:
    def transcribe(self, media_path: Path) -> TranscriptResult:
        return TranscriptResult(
            text="",
            segments=[],
        )


class LockedTranscriber:
    """Serializes transcription so large Whisper models are loaded once."""

    def __init__(self, transcriber: Transcriber) -> None:
        self._transcriber = transcriber
        self._lock = threading.Lock()

    def transcribe(self, media_path: Path) -> TranscriptResult:
        with self._lock:
            return self._transcriber.transcribe(media_path)


class FasterWhisperTranscriber:
    def __init__(self, model_name: str) -> None:
        module = importlib.import_module("faster_whisper")
        self._model = module.WhisperModel(model_name)

    def transcribe(self, media_path: Path) -> TranscriptResult:
        segments_iter, _info = self._model.transcribe(str(media_path))
        segments = []
        text_parts = []
        for segment in segments_iter:
            text = segment.text.strip()
            if text:
                text_parts.append(text)
            segments.append(
                TranscriptSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=text,
                )
            )
        text = " ".join(text_parts).strip()
        if not text:
            raise ProcessingError("Whisper returned empty transcript")
        return TranscriptResult(text=text, segments=segments)


class OpenAIWhisperTranscriber:
    def __init__(self, model_name: str) -> None:
        module = importlib.import_module("whisper")
        self._model = module.load_model(model_name)

    def transcribe(self, media_path: Path) -> TranscriptResult:
        result = self._model.transcribe(str(media_path))
        text = str(result.get("text") or "").strip()
        if not text:
            raise ProcessingError("Whisper returned empty transcript")
        segments = [
            TranscriptSegment(
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=str(segment.get("text") or "").strip(),
            )
            for segment in result.get("segments", [])
        ]
        return TranscriptResult(text=text, segments=segments)


class WhisperCppTranscriber:
    def __init__(
        self,
        model_name: str,
        models_dir: Optional[str] = None,
        require_coreml: bool = False,
        n_threads: Optional[int] = None,
    ) -> None:
        configure_whisper_cpp_runtime_env()
        module = importlib.import_module("pywhispercpp.model")
        if require_coreml:
            system_info = str(module.Model.system_info())
            if "COREML = 1" not in system_info:
                raise DependencyMissingError(
                    "pywhispercpp is installed, but it was not built with CoreML support. "
                    "Run scripts/setup-whisper-cpp-coreml.sh, then keep the ggml .bin model "
                    "next to its matching *-encoder.mlmodelc directory."
                )

        model_kwargs = {
            "print_realtime": False,
            "print_progress": False,
            "redirect_whispercpp_logs_to": False,
        }
        if models_dir:
            model_kwargs["models_dir"] = models_dir
        if n_threads:
            model_kwargs["n_threads"] = n_threads

        self._model = module.Model(model_name, **model_kwargs)

    def transcribe(self, media_path: Path) -> TranscriptResult:
        segments_raw = self._model.transcribe(str(media_path))
        segments = []
        text_parts = []
        for segment in segments_raw:
            text = str(getattr(segment, "text", "") or "").strip()
            if text:
                text_parts.append(text)
            start, end = _whisper_cpp_segment_times(segment)
            segments.append(TranscriptSegment(start=start, end=end, text=text))
        text = " ".join(text_parts).strip()
        if not text:
            raise ProcessingError("Whisper returned empty transcript")
        return TranscriptResult(text=text, segments=segments)


def build_transcriber(
    engine: TranscriptionEngine,
    model_name: str,
    *,
    prefer_whisper_cpp: bool = False,
    whisper_cpp_model_dir: str = "",
    require_whisper_cpp_coreml: bool = False,
    n_threads: Optional[int] = None,
) -> LockedTranscriber:
    if engine is TranscriptionEngine.NONE:
        return LockedTranscriber(NullTranscriber())

    wants_whisper_cpp = engine is TranscriptionEngine.WHISPER_CPP or (
        engine is TranscriptionEngine.AUTO and (prefer_whisper_cpp or require_whisper_cpp_coreml)
    )
    if wants_whisper_cpp:
        try:
            return LockedTranscriber(
                WhisperCppTranscriber(
                    model_name,
                    models_dir=whisper_cpp_model_dir or None,
                    require_coreml=require_whisper_cpp_coreml,
                    n_threads=n_threads,
                )
            )
        except ImportError:
            if engine is TranscriptionEngine.WHISPER_CPP or require_whisper_cpp_coreml:
                raise DependencyMissingError(
                    "pywhispercpp is not installed. For Apple Silicon acceleration, run "
                    "scripts/setup-whisper-cpp-coreml.sh or install pywhispercpp from source "
                    "with CoreML enabled."
                )
        except DependencyMissingError:
            if engine is TranscriptionEngine.WHISPER_CPP or require_whisper_cpp_coreml:
                raise
        except Exception as exc:
            if engine is TranscriptionEngine.WHISPER_CPP or require_whisper_cpp_coreml:
                raise ProcessingError(
                    f"Could not initialize pywhispercpp model '{model_name}': {exc}"
                ) from exc

    if engine in (TranscriptionEngine.AUTO, TranscriptionEngine.FASTER_WHISPER):
        try:
            return LockedTranscriber(FasterWhisperTranscriber(model_name))
        except ImportError:
            if engine is TranscriptionEngine.FASTER_WHISPER:
                raise DependencyMissingError(
                    "faster-whisper is not installed. Install with: "
                    "python3 -m pip install -e '.[transcription]'"
                )

    if engine in (TranscriptionEngine.AUTO, TranscriptionEngine.OPENAI_WHISPER):
        try:
            return LockedTranscriber(OpenAIWhisperTranscriber(model_name))
        except ImportError:
            if engine is TranscriptionEngine.OPENAI_WHISPER:
                raise DependencyMissingError(
                    "openai-whisper is not installed. Install the whisper package or use "
                    "--transcription-engine faster-whisper."
                )

    raise DependencyMissingError(
        "No Whisper transcription engine is installed. Install optional dependencies or "
        "use --transcription-engine none for media-only smoke tests."
    )


def configure_whisper_cpp_runtime_env() -> None:
    os.environ.setdefault("PYWHISPERCPP_USE_GPU", "0")
    os.environ.setdefault("PYWHISPERCPP_FLASH_ATTN", "0")


def _whisper_cpp_segment_times(segment) -> tuple:
    if hasattr(segment, "start") and hasattr(segment, "end"):
        return float(segment.start), float(segment.end)
    if hasattr(segment, "t0") and hasattr(segment, "t1"):
        return float(segment.t0) / 100.0, float(segment.t1) / 100.0
    return 0.0, 0.0
