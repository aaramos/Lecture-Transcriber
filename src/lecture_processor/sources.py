from pathlib import Path
from typing import Iterable, List


VIDEO_EXTENSIONS = {
    ".mov",
    ".mp4",
    ".m4v",
    ".mkv",
    ".webm",
    ".avi",
    ".wmv",
    ".mpg",
    ".mpeg",
    ".mts",
    ".m2ts",
    ".ts",
}

AUDIO_EXTENSIONS = {
    ".mp3",
    ".m4a",
    ".wav",
    ".aac",
    ".flac",
}

TRANSCRIPT_EXTENSIONS = {
    ".srt",
    ".vtt",
    ".txt",
    ".docx",
}

SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS | TRANSCRIPT_EXTENSIONS


def source_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    if suffix in TRANSCRIPT_EXTENSIONS:
        return "transcript"
    return "unsupported"


def is_supported_source(path: Path) -> bool:
    return source_kind(path) != "unsupported"


def is_media_source(path: Path) -> bool:
    return source_kind(path) in {"video", "audio"}


def is_transcript_source(path: Path) -> bool:
    return source_kind(path) == "transcript"


def discover_source_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path] if is_supported_source(path) else []
    if not path.is_dir():
        return []
    return sorted(
        (item for item in path.iterdir() if item.is_file() and is_supported_source(item)),
        key=lambda item: item.name.lower(),
    )


def supported_extensions_text(extensions: Iterable[str] = SUPPORTED_EXTENSIONS) -> str:
    return ", ".join(sorted(extensions))
