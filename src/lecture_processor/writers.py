from pathlib import Path

from .models import TranscriptResult
from .timecode import segments_to_srt


def write_transcript(output_dir: Path, transcript: TranscriptResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(output_dir / "transcript.txt", transcript.text + ("\n" if transcript.text else ""))
    write_text_atomic(output_dir / "transcript.srt", segments_to_srt(transcript.segments))


def write_processing_log(output_dir: Path, lines: list) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(output_dir / "processing_log.txt", "\n".join(lines).rstrip() + "\n")


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        temp_path.write_text(text, encoding="utf-8")
        temp_path.replace(path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
