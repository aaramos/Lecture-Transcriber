import argparse
import json
import sys
import threading
from pathlib import Path

from .config import (
    AudioQuality,
    BatchConfig,
    RecordingSpeed,
    SlideSensitivity,
    TranscriptionEngine,
)
from .errors import LectureProcessorError
from .media import ensure_media_tools
from .models import BatchSummary, FileStatus
from .pipeline import BatchProcessor, discover_mov_files
from .slides import SlideExtractor
from .transcription import build_transcriber

EVENT_PREFIX = "__LECTURE_PROCESSOR_EVENT__ "


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lecture-processor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process", help="Process a folder of .mov lecture recordings")
    process.add_argument("input_dir", type=Path)
    process.add_argument("--output", type=Path)
    process.add_argument("--recording-speed", choices=[item.value for item in RecordingSpeed], default="1x")
    process.add_argument("--confirm-normalization", action="store_true")
    process.add_argument("--concurrent", type=int, default=4)
    process.add_argument("--min-duration", type=float, default=60.0)
    process.add_argument("--no-save-normalized-video", action="store_true")
    process.add_argument("--audio-quality", choices=[item.value for item in AudioQuality], default="fast")
    process.add_argument("--slide-sensitivity", choices=[item.value for item in SlideSensitivity], default="medium")
    process.add_argument(
        "--transcription-engine",
        choices=[item.value for item in TranscriptionEngine],
        default="auto",
    )
    process.add_argument("--whisper-model", default="large-v3")
    process.add_argument("--ffmpeg", default="ffmpeg")
    process.add_argument("--ffprobe", default="ffprobe")
    process.add_argument("--json-events", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "process":
        return _run_process(args)
    parser.error("Unknown command")
    return 2


def _run_process(args) -> int:
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output.expanduser().resolve() if args.output else _default_output_dir(input_dir)
    config = BatchConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        recording_speed=RecordingSpeed(args.recording_speed),
        confirm_normalization=args.confirm_normalization,
        concurrent_files=args.concurrent,
        min_duration_seconds=args.min_duration,
        save_normalized_video=not args.no_save_normalized_video,
        audio_quality=AudioQuality(args.audio_quality),
        slide_sensitivity=SlideSensitivity(args.slide_sensitivity),
        transcription_engine=TranscriptionEngine(args.transcription_engine),
        whisper_model=args.whisper_model,
        ffmpeg_path=args.ffmpeg,
        ffprobe_path=args.ffprobe,
    )

    try:
        config.validate()
        if not discover_mov_files(config.input_dir):
            raise LectureProcessorError("No .mov files found. Try a different folder.")
        needs_ffmpeg = (
            config.recording_speed is RecordingSpeed.DOUBLE
            or config.transcription_engine is not TranscriptionEngine.NONE
        )
        ensure_media_tools(
            ffprobe_path=config.ffprobe_path,
            ffmpeg_path=config.ffmpeg_path,
            needs_ffmpeg=needs_ffmpeg,
        )
        transcriber = build_transcriber(config.transcription_engine, config.whisper_model)
        processor = BatchProcessor(
            config=config,
            transcriber=transcriber,
            slide_extractor=SlideExtractor(config.slide_sensitivity),
            progress_callback=_build_event_printer(args.json_events),
        )
        summary = processor.run()
    except LectureProcessorError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(_format_summary(summary, output_dir))
    return 1 if summary.failed else 0


def _build_event_printer(enabled: bool):
    if not enabled:
        return None

    lock = threading.Lock()

    def print_event(event) -> None:
        with lock:
            print(f"{EVENT_PREFIX}{json.dumps(event, sort_keys=True)}", flush=True)

    return print_event


def _default_output_dir(input_dir: Path) -> Path:
    return input_dir.parent / f"{input_dir.name}_processed"


def _format_summary(summary: BatchSummary, output_dir: Path) -> str:
    lines = [
        "Batch finished",
        f"Output:    {output_dir}",
        f"Attempted: {summary.attempted}",
        f"Completed: {summary.completed}",
        f"Failed:    {summary.failed}",
        f"Skipped:   {summary.skipped}",
        "",
        "Files:",
    ]
    for result in summary.results:
        if result.status is FileStatus.COMPLETED:
            detail = f"{result.word_count} words, {result.slide_count} slides"
        elif result.status is FileStatus.FAILED:
            detail = f"{result.failure_step}: {result.message}"
        else:
            detail = result.message
        lines.append(f"- {result.source.name}: {result.status.value} ({detail})")
    return "\n".join(lines)
