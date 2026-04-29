import re
import zipfile
from pathlib import Path
from typing import List, Optional
from xml.etree import ElementTree

from .errors import ProcessingError
from .models import TranscriptResult, TranscriptSegment


_TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
)


def import_transcript(path: Path) -> TranscriptResult:
    suffix = path.suffix.lower()
    if suffix == ".srt":
        return _parse_caption_text(_read_text(path), strip_vtt_header=False)
    if suffix == ".vtt":
        return _parse_caption_text(_read_text(path), strip_vtt_header=True)
    if suffix == ".txt":
        return _plain_text_transcript(_read_text(path))
    if suffix == ".docx":
        return _plain_text_transcript(_read_docx_text(path))
    raise ProcessingError(f"Unsupported transcript file: {path.name}")


def transcript_duration_seconds(transcript: TranscriptResult) -> float:
    if not transcript.segments:
        return 0.0
    return max((segment.end for segment in transcript.segments), default=0.0)


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeError:
            continue
    raise ProcessingError(f"Could not read transcript text from {path.name}")


def _read_docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            xml_bytes = archive.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile, OSError) as exc:
        raise ProcessingError(f"Could not read Word transcript from {path.name}") from exc

    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError as exc:
        raise ProcessingError(f"Could not parse Word transcript from {path.name}") from exc

    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs: List[str] = []
    for paragraph in root.iter(f"{namespace}p"):
        pieces = [node.text or "" for node in paragraph.iter(f"{namespace}t")]
        text = "".join(pieces).strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def _plain_text_transcript(text: str) -> TranscriptResult:
    cleaned = _clean_plain_text(text)
    if not cleaned:
        raise ProcessingError("Transcript file is empty.")
    return TranscriptResult(
        text=cleaned,
        segments=[TranscriptSegment(start=0.0, end=0.0, text=cleaned)],
    )


def _parse_caption_text(text: str, strip_vtt_header: bool) -> TranscriptResult:
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    segments: List[TranscriptSegment] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if strip_vtt_header and lines[0].upper().startswith("WEBVTT"):
            continue

        timestamp_index = _timestamp_line_index(lines)
        if timestamp_index is None:
            continue
        match = _TIMESTAMP_RE.search(lines[timestamp_index])
        if not match:
            continue
        caption_lines = lines[timestamp_index + 1 :]
        if not caption_lines:
            continue
        caption = _clean_caption_text(" ".join(caption_lines))
        if not caption:
            continue
        segments.append(
            TranscriptSegment(
                start=_parse_timestamp(match.group("start")),
                end=_parse_timestamp(match.group("end")),
                text=caption,
            )
        )

    if not segments:
        return _plain_text_transcript(text)
    return TranscriptResult(
        text="\n".join(segment.text for segment in segments),
        segments=segments,
    )


def _timestamp_line_index(lines: List[str]) -> Optional[int]:
    for index, line in enumerate(lines):
        if _TIMESTAMP_RE.search(line):
            return index
    return None


def _parse_timestamp(value: str) -> float:
    hours, minutes, seconds = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _clean_caption_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", "", value)
    return _clean_plain_text(without_tags)


def _clean_plain_text(value: str) -> str:
    return re.sub(r"[ \t]+", " ", value).strip()
