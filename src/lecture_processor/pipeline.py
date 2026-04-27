import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from .config import BatchConfig, RecordingSpeed
from .errors import LectureProcessorError
from .media import MediaInspector, MediaNormalizer
from .models import BatchSummary, FileResult, FileStatus, TranscriptResult
from .slides import SlideExtractor
from .transcription import Transcriber
from .writers import write_processing_log, write_transcript


class BatchProcessor:
    def __init__(
        self,
        config: BatchConfig,
        inspector: Optional[MediaInspector] = None,
        normalizer: Optional[MediaNormalizer] = None,
        transcriber: Optional[Transcriber] = None,
        slide_extractor: Optional[SlideExtractor] = None,
    ) -> None:
        self.config = config
        self.inspector = inspector or MediaInspector(config.ffprobe_path)
        self.normalizer = normalizer or MediaNormalizer(config.ffmpeg_path)
        if transcriber is None:
            raise LectureProcessorError("BatchProcessor requires a transcriber")
        self.transcriber = transcriber
        self.slide_extractor = slide_extractor or SlideExtractor(config.slide_sensitivity)

    def run(self) -> BatchSummary:
        self.config.validate()
        files = discover_mov_files(self.config.input_dir)
        if not files:
            raise LectureProcessorError("No .mov files found. Try a different folder.")

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs = allocate_output_dirs(files, self.config.output_dir)
        results: List[FileResult] = []
        with ThreadPoolExecutor(max_workers=self.config.concurrent_files) as executor:
            futures = [executor.submit(self._process_file, path, output_dirs[path]) for path in files]
            for future in as_completed(futures):
                results.append(future.result())

        results.sort(key=lambda item: item.source.name.lower())
        summary = BatchSummary(
            attempted=len(results),
            completed=sum(1 for item in results if item.status is FileStatus.COMPLETED),
            failed=sum(1 for item in results if item.status is FileStatus.FAILED),
            skipped=sum(1 for item in results if item.status is FileStatus.SKIPPED),
            results=results,
        )
        write_batch_summary(self.config.output_dir, summary)
        return summary

    def _process_file(self, source: Path, output_dir: Path) -> FileResult:
        reset_output_dir(output_dir)
        started = time.monotonic()
        log_lines = [
            f"File:     {source.name}",
            f"Started:  {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        temp_normalized: Optional[Path] = None
        step_state = {"current": "Probe"}

        try:
            media_info = self._time_step("Probe", log_lines, step_state, lambda: self.inspector.probe(source))
            if media_info.duration_seconds < self.config.min_duration_seconds:
                message = (
                    f"File too short ({media_info.duration_seconds:.0f}s < "
                    f"{self.config.min_duration_seconds:.0f}s minimum)"
                )
                log_lines.append(f"Skipped: {message}")
                write_processing_log(output_dir, log_lines)
                return FileResult(
                    source=source,
                    output_dir=output_dir,
                    status=FileStatus.SKIPPED,
                    duration_seconds=media_info.duration_seconds,
                    message=message,
                )

            work_video = source
            normalized_duration = None
            if self.config.recording_speed is RecordingSpeed.DOUBLE:
                destination = output_dir / "normalized_video.mp4"
                if not self.config.save_normalized_video:
                    destination = output_dir / ".normalized_work.mp4"
                    temp_normalized = destination
                work_video = self._time_step(
                    "Normalize",
                    log_lines,
                    step_state,
                    lambda: self.normalizer.normalize(
                        source=source,
                        destination=destination,
                        media_info=media_info,
                        recording_speed=self.config.recording_speed,
                        audio_quality=self.config.audio_quality,
                    ),
                )
                normalized_duration = media_info.duration_seconds * 2.0

            transcript = self._time_step(
                "Transcribe",
                log_lines,
                step_state,
                lambda: self.transcriber.transcribe(work_video),
            )
            transcript = _scale_transcript(transcript, self.config.normalized_timestamp_scale)
            write_transcript(output_dir, transcript)
            log_lines[-1] = f"{log_lines[-1]} ({transcript.word_count} words)"

            slides_dir = output_dir / "slides"
            slide_count = self._time_step(
                "Slides",
                log_lines,
                step_state,
                lambda: self.slide_extractor.extract(
                    work_video,
                    slides_dir,
                    timestamp_scale=self.config.normalized_timestamp_scale,
                ),
            )
            log_lines[-1] = f"{log_lines[-1]} ({slide_count} slides extracted)"

            if temp_normalized and temp_normalized.exists():
                temp_normalized.unlink()

            elapsed = time.monotonic() - started
            log_lines.append(f"Complete in {elapsed:.1f}s")
            write_processing_log(output_dir, log_lines)
            return FileResult(
                source=source,
                output_dir=output_dir,
                status=FileStatus.COMPLETED,
                duration_seconds=media_info.duration_seconds,
                normalized_duration_seconds=normalized_duration,
                word_count=transcript.word_count,
                slide_count=slide_count,
                message="Complete",
            )
        except Exception as exc:
            cleanup_partial_output(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            step = step_state["current"]
            message = str(exc)
            log_lines.append(f"{step}  ERROR: {message}")
            log_lines.append("Partial output cleaned. File skipped.")
            write_processing_log(output_dir, log_lines)
            return FileResult(
                source=source,
                output_dir=output_dir,
                status=FileStatus.FAILED,
                failure_step=step,
                message=message,
            )

    def _time_step(self, name: str, log_lines: List[str], step_state, operation):
        step_state["current"] = name
        started = time.monotonic()
        result = operation()
        elapsed = time.monotonic() - started
        log_lines.append(f"{name:<10} OK  {elapsed:.1f}s")
        return result


def discover_mov_files(folder: Path) -> List[Path]:
    return sorted(
        (item for item in folder.iterdir() if item.is_file() and item.suffix.lower() == ".mov"),
        key=lambda item: item.name.lower(),
    )


def safe_folder_name(stem: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return value or "lecture"


def allocate_output_dirs(files: List[Path], output_root: Path) -> dict:
    allocated = {}
    seen = {}
    for path in sorted(files, key=lambda item: item.name.lower()):
        base = safe_folder_name(path.stem)
        index = seen.get(base.lower(), 0) + 1
        seen[base.lower()] = index
        folder_name = base if index == 1 else f"{base}_{index}"
        allocated[path] = output_root / folder_name
    return allocated


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def cleanup_partial_output(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)


def write_batch_summary(output_dir: Path, summary: BatchSummary) -> None:
    lines = [
        "Batch summary",
        f"Attempted: {summary.attempted}",
        f"Completed: {summary.completed}",
        f"Failed:    {summary.failed}",
        f"Skipped:   {summary.skipped}",
        "",
        "Per-file results:",
    ]
    for result in summary.results:
        status = result.status.value
        detail = result.message
        if result.status is FileStatus.COMPLETED:
            detail = f"{result.word_count} words, {result.slide_count} slides"
        elif result.status is FileStatus.FAILED and result.failure_step:
            detail = f"{result.failure_step}: {result.message}"
        lines.append(f"- {result.source.name}: {status} - {detail}")
    (output_dir / "batch_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scale_transcript(transcript: TranscriptResult, scale: float) -> TranscriptResult:
    return transcript.scaled(scale)
