import importlib
import threading
from pathlib import Path
from typing import Protocol

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


def build_transcriber(engine: TranscriptionEngine, model_name: str) -> LockedTranscriber:
    if engine is TranscriptionEngine.NONE:
        return LockedTranscriber(NullTranscriber())

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
