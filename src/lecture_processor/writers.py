from pathlib import Path

from .models import TranscriptResult
from .timecode import segments_to_srt


def write_transcript(output_dir: Path, transcript: TranscriptResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_text(output_dir / "transcript.txt", transcript.text + ("\n" if transcript.text else ""))
    _write_text(output_dir / "transcript.srt", segments_to_srt(transcript.segments))


def write_processing_log(output_dir: Path, lines: list) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_text(output_dir / "processing_log.txt", "\n".join(lines).rstrip() + "\n")


def _write_text(path: Path, text: str) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)
