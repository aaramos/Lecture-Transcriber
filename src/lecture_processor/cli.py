import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

from .config import (
    AIProviderName,
    AudioQuality,
    BatchConfig,
    FfmpegHwAccel,
    RecordingSpeed,
    SlideBackend,
    SlideSensitivity,
    TranscriptionEngine,
    TranscriptionQuality,
)
from .ai.enrichment import enrich_lecture_artifact
from .errors import LectureProcessorError
from .gemini_export import export_gemini_test_package
from .html_renderer import render_lecture_page
from .media import ensure_media_tools, resolve_media_tool
from .models import BatchSummary, FileStatus
from .pipeline import BatchProcessor, discover_mov_files, enrich_processed_batch
from .slides import SlideExtractor
from .temp_cleanup import cleanup_slide_temp_dirs
from .transcription import build_transcriber
from .writers import write_text_atomic

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
    process.add_argument("--audio-quality", choices=[item.value for item in AudioQuality], default="high")
    process.add_argument("--slide-sensitivity", choices=[item.value for item in SlideSensitivity], default="medium")
    process.add_argument("--slide-backend", choices=[item.value for item in SlideBackend], default="auto")
    process.add_argument(
        "--transcription-engine",
        choices=[item.value for item in TranscriptionEngine],
        default="faster-whisper",
    )
    process.add_argument(
        "--transcription-quality",
        choices=[item.value for item in TranscriptionQuality],
        default="accurate",
    )
    process.add_argument("--whisper-model", default="medium.en")
    process.add_argument("--whisper-cpp-model-dir", default="")
    process.add_argument("--require-whisper-cpp-coreml", action="store_true")
    process.add_argument("--apple-silicon", action="store_true")
    process.add_argument("--ffmpeg-hwaccel", choices=[item.value for item in FfmpegHwAccel], default="auto")
    process.add_argument("--ai-provider", choices=[item.value for item in AIProviderName], default="none")
    process.add_argument("--ai-model", default="")
    process.add_argument("--no-render-html", action="store_true")
    process.add_argument("--skip-file", action="append", default=[], help=argparse.SUPPRESS)
    process.add_argument("--control-file", type=Path, default=None, help=argparse.SUPPRESS)
    process.add_argument("--ffmpeg", default="ffmpeg")
    process.add_argument("--ffprobe", default="ffprobe")
    process.add_argument("--json-events", action="store_true", help=argparse.SUPPRESS)

    enrich = subparsers.add_parser("enrich", help="Enrich an existing lecture.json artifact")
    enrich.add_argument("lecture_json", type=Path)
    enrich.add_argument("--ai-provider", choices=["mock", "gemini"], default="mock")
    enrich.add_argument("--ai-model", default="")
    enrich.add_argument("--render-html", action="store_true")

    enrich_batch = subparsers.add_parser("enrich-batch", help="Enhance an existing processed batch folder")
    enrich_batch.add_argument("processed_dir", type=Path)
    enrich_batch.add_argument("--ai-provider", choices=["mock", "gemini"], default="gemini")
    enrich_batch.add_argument("--ai-model", default="")
    enrich_batch.add_argument("--concurrent", type=int, default=1)
    enrich_batch.add_argument("--skip-file", action="append", default=[], help=argparse.SUPPRESS)
    enrich_batch.add_argument("--no-render-html", action="store_true")
    enrich_batch.add_argument("--json-events", action="store_true", help=argparse.SUPPRESS)

    render = subparsers.add_parser("render", help="Render an existing lecture.json artifact")
    render.add_argument("lecture_json", type=Path)

    export_gemini = subparsers.add_parser(
        "export-gemini-test",
        help="Create a prompt and zip package for manual Gemini testing",
    )
    export_gemini.add_argument("lecture", type=Path, help="Lecture folder or lecture.json")
    export_gemini.add_argument("--output", type=Path)
    export_gemini.add_argument("--max-slides", type=int, default=40)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "process":
        return _run_process(args)
    if args.command == "enrich":
        return _run_enrich(args)
    if args.command == "enrich-batch":
        return _run_enrich_batch(args)
    if args.command == "render":
        return _run_render(args)
    if args.command == "export-gemini-test":
        return _run_export_gemini_test(args)
    parser.error("Unknown command")
    return 2


def _run_process(args) -> int:
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output.expanduser().resolve() if args.output else _default_output_dir(input_dir)
    apple_silicon = args.apple_silicon or _detect_apple_silicon()
    event_printer = _build_event_printer(args.json_events)
    cleanup_slide_temp_dirs()
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
        slide_backend=SlideBackend(args.slide_backend),
        transcription_engine=TranscriptionEngine(args.transcription_engine),
        transcription_quality=TranscriptionQuality(args.transcription_quality),
        whisper_model=args.whisper_model,
        whisper_cpp_model_dir=args.whisper_cpp_model_dir,
        require_whisper_cpp_coreml=args.require_whisper_cpp_coreml,
        ffmpeg_path=args.ffmpeg,
        ffprobe_path=args.ffprobe,
        ffmpeg_hwaccel=FfmpegHwAccel(args.ffmpeg_hwaccel),
        apple_silicon=apple_silicon,
        ai_provider=AIProviderName(args.ai_provider),
        ai_model=args.ai_model,
        render_html=not args.no_render_html,
        skip_files=tuple(args.skip_file or ()),
        control_file=args.control_file,
    )

    try:
        config.validate()
        if config.ai_provider is AIProviderName.GEMINI and not _gemini_api_key_available():
            raise LectureProcessorError("Gemini enrichment requires a saved or exported GEMINI_API_KEY.")
        if config.ai_provider is AIProviderName.GEMINI and not _gemini_dependency_available():
            raise LectureProcessorError("Gemini support is not installed. Install with: python3 -m pip install -e '.[ai]'")
        if not discover_mov_files(config.input_dir):
            raise LectureProcessorError("No .mov files found. Try a different folder.")
        needs_ffmpeg = (
            config.recording_speed is RecordingSpeed.DOUBLE
            or config.transcription_engine is not TranscriptionEngine.NONE
            or config.slide_backend is not SlideBackend.OPENCV
        )
        ensure_media_tools(
            ffprobe_path=config.ffprobe_path,
            ffmpeg_path=config.ffmpeg_path,
            needs_ffmpeg=needs_ffmpeg,
        )
        if needs_ffmpeg:
            _prepend_tool_parent_to_path(config.ffmpeg_path)
        transcriber = build_transcriber(
            config.transcription_engine,
            config.whisper_model,
            quality=config.transcription_quality,
            prefer_whisper_cpp=False,
            whisper_cpp_model_dir=config.whisper_cpp_model_dir,
            require_whisper_cpp_coreml=config.require_whisper_cpp_coreml,
        )
        processor = BatchProcessor(
            config=config,
            transcriber=transcriber,
            slide_extractor=SlideExtractor(
                config.slide_sensitivity,
                backend=config.slide_backend,
                ffmpeg_path=config.ffmpeg_path,
                ffmpeg_hwaccel=config.ffmpeg_hwaccel,
                apple_silicon=config.apple_silicon,
            ),
            progress_callback=event_printer,
        )
        summary = processor.run()
    except LectureProcessorError as exc:
        _write_run_error(output_dir, str(exc))
        if event_printer:
            event_printer({"kind": "batch_failed", "message": str(exc), "output_dir": str(output_dir)})
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        message = f"Unexpected processor error: {exc}"
        _write_run_error(output_dir, message)
        if event_printer:
            event_printer({"kind": "batch_failed", "message": message, "output_dir": str(output_dir)})
        print(f"Error: {message}", file=sys.stderr)
        return 1

    print(_format_summary(summary, output_dir))
    return 1 if summary.failed else 0


def _run_enrich(args) -> int:
    lecture_json = args.lecture_json.expanduser().resolve()
    if not lecture_json.exists():
        print(f"Error: lecture artifact not found: {lecture_json}", file=sys.stderr)
        return 2
    provider = AIProviderName(args.ai_provider)
    if provider is AIProviderName.GEMINI and not _gemini_api_key_available():
        print("Error: Gemini enrichment requires GEMINI_API_KEY.", file=sys.stderr)
        return 2
    if provider is AIProviderName.GEMINI and not _gemini_dependency_available():
        print("Error: Gemini support is not installed. Install with: python3 -m pip install -e '.[ai]'", file=sys.stderr)
        return 2
    config = BatchConfig(
        input_dir=lecture_json.parent,
        output_dir=lecture_json.parent,
        transcription_engine=TranscriptionEngine.NONE,
        ai_provider=provider,
        ai_model=args.ai_model,
    )
    try:
        artifact = enrich_lecture_artifact(lecture_json, config)
        if args.render_html:
            html_path = render_lecture_page(lecture_json)
            print(f"Rendered: {html_path}")
        print(f"Enriched: {artifact.get('enrichment', {}).get('title', lecture_json.name)}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _run_enrich_batch(args) -> int:
    processed_dir = args.processed_dir.expanduser().resolve()
    provider = AIProviderName(args.ai_provider)
    event_printer = _build_event_printer(args.json_events)
    if provider is AIProviderName.GEMINI and not _gemini_api_key_available():
        print("Error: Gemini enrichment requires GEMINI_API_KEY.", file=sys.stderr)
        return 2
    if provider is AIProviderName.GEMINI and not _gemini_dependency_available():
        print("Error: Gemini support is not installed. Install with: python3 -m pip install -e '.[ai]'", file=sys.stderr)
        return 2
    config = BatchConfig(
        input_dir=processed_dir,
        output_dir=processed_dir,
        transcription_engine=TranscriptionEngine.NONE,
        concurrent_files=args.concurrent,
        ai_provider=provider,
        ai_model=args.ai_model,
        render_html=not args.no_render_html,
        skip_files=tuple(args.skip_file or ()),
    )
    try:
        summary = enrich_processed_batch(config, progress_callback=event_printer)
    except LectureProcessorError as exc:
        if event_printer:
            event_printer({"kind": "batch_failed", "message": str(exc), "output_dir": str(processed_dir)})
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        message = f"Unexpected enrichment error: {exc}"
        if event_printer:
            event_printer({"kind": "batch_failed", "message": message, "output_dir": str(processed_dir)})
        print(f"Error: {message}", file=sys.stderr)
        return 1

    print(_format_summary(summary, processed_dir))
    return 1 if summary.failed else 0


def _run_render(args) -> int:
    lecture_json = args.lecture_json.expanduser().resolve()
    if not lecture_json.exists():
        print(f"Error: lecture artifact not found: {lecture_json}", file=sys.stderr)
        return 2
    try:
        html_path = render_lecture_page(lecture_json)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Rendered: {html_path}")
    return 0


def _run_export_gemini_test(args) -> int:
    try:
        zip_path = export_gemini_test_package(
            args.lecture,
            output_path=args.output,
            max_slides=args.max_slides,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Gemini test package: {zip_path}")
    return 0


def _build_event_printer(enabled: bool):
    if not enabled:
        return None

    lock = threading.Lock()

    def print_event(event) -> None:
        with lock:
            print(f"{EVENT_PREFIX}{json.dumps(event, sort_keys=True)}", flush=True)

    return print_event


def _detect_apple_silicon() -> bool:
    if sys.platform != "darwin":
        return False
    if platform.machine() == "arm64":
        return True
    try:
        completed = subprocess.run(
            ["sysctl", "-n", "hw.optional.arm64"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return completed.returncode == 0 and completed.stdout.strip() == "1"


def _gemini_api_key_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("LECTURE_PROCESSOR_GEMINI_API_KEY"))


def _gemini_dependency_available() -> bool:
    try:
        return importlib.util.find_spec("google.genai") is not None
    except ModuleNotFoundError:
        return False


def _default_output_dir(input_dir: Path) -> Path:
    return input_dir.parent / f"{input_dir.name}_processed"


def _prepend_tool_parent_to_path(command: str) -> None:
    tool_path = Path(resolve_media_tool(command))
    tool_dir = str(tool_path.parent)
    existing = os.environ.get("PATH", "")
    paths = existing.split(os.pathsep) if existing else []
    if tool_dir not in paths:
        os.environ["PATH"] = os.pathsep.join([tool_dir, *paths])


def _write_run_error(output_dir: Path, message: str) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_text_atomic(
            output_dir / "batch_error.txt",
            "\n".join(
                [
                    "Batch failed before all video logs were available.",
                    f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Environment: {platform.platform()}",
                    "",
                    "Error:",
                    message,
                    "",
                ]
            ),
        )
        write_text_atomic(
            output_dir / "batch_summary.txt",
            "\n".join(
                [
                    "Batch summary",
                    "Attempted: 0",
                    "Completed: 0",
                    "Failed:    1",
                    "Skipped:   0",
                    "",
                    "Run failed before file processing completed:",
                    f"- {message}",
                    "",
                ]
            ),
        )
    except OSError:
        pass


def _format_summary(summary: BatchSummary, output_dir: Path) -> str:
    lines = [
        "Batch finished",
        f"Output:    {output_dir}",
        f"Attempted: {summary.attempted}",
        f"Completed: {summary.completed}",
        f"Failed:    {summary.failed}",
        f"Skipped:   {summary.skipped}",
        f"Stopped:   {summary.stopped}",
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
