from pathlib import Path

from .models import TranscriptResult
from .timecode import segments_to_srt


def write_transcript(output_dir: Path, transcript: TranscriptResult) -> None:
    (output_dir / "transcript.txt").write_text(transcript.text + ("\n" if transcript.text else ""), encoding="utf-8")
    (output_dir / "transcript.srt").write_text(segments_to_srt(transcript.segments), encoding="utf-8")


def write_processing_log(output_dir: Path, lines: list) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "processing_log.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
