import json
import os
import re
import shutil
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

from .artifacts import LECTURE_ARTIFACT_NAME, load_json, utc_now_iso, write_batch_artifact, write_lecture_artifact
from .ai.enrichment import enrich_lecture_artifact
from .config import BatchConfig, RecordingSpeed, TranscriptionEngine
from .control import ProcessingControl, default_control_file
from .errors import LectureProcessorError
from .errors import ProcessingStopped
from .html_renderer import render_batch_index, render_lecture_page
from .media import CleanAudioExtractor, MediaInspector, MediaNormalizer
from .models import BatchSummary, FileResult, FileStatus, TranscriptResult
from .slides import SlideExtractor
from .temp_cleanup import cleanup_processor_temp_files, remove_temp_path
from .transcription import Transcriber
from .writers import write_processing_log, write_text_atomic, write_transcript

OUTPUT_LOCK_FILE = ".lecture_processor.lock"
ALREADY_PROCESSED_MESSAGE = "Already processed"
ALREADY_ENHANCED_MESSAGE = "Already enhanced"


class BatchProcessor:
    def __init__(
        self,
        config: BatchConfig,
        inspector: Optional[MediaInspector] = None,
        normalizer: Optional[MediaNormalizer] = None,
        audio_extractor: Optional[CleanAudioExtractor] = None,
        transcriber: Optional[Transcriber] = None,
        slide_extractor: Optional[SlideExtractor] = None,
        progress_callback: Optional[Callable[[Dict], None]] = None,
    ) -> None:
        self.config = config
        self.inspector = inspector or MediaInspector(config.ffprobe_path)
        self.normalizer = normalizer or MediaNormalizer(
            config.ffmpeg_path,
            ffmpeg_hwaccel=config.ffmpeg_hwaccel,
            apple_silicon=config.apple_silicon,
        )
        self.audio_extractor = audio_extractor or CleanAudioExtractor(config.ffmpeg_path)
        if transcriber is None:
            raise LectureProcessorError("BatchProcessor requires a transcriber")
        self.transcriber = transcriber
        self.slide_extractor = slide_extractor or SlideExtractor(
            config.slide_sensitivity,
            backend=config.slide_backend,
            ffmpeg_path=config.ffmpeg_path,
            ffmpeg_hwaccel=config.ffmpeg_hwaccel,
            apple_silicon=config.apple_silicon,
        )
        self.progress_callback = progress_callback
        self.control = ProcessingControl(config.control_file or default_control_file(config.output_dir))

    def run(self) -> BatchSummary:
        self.config.validate()
        files = discover_mov_files(self.config.input_dir)
        if not files:
            raise LectureProcessorError("No .mov files found. Try a different folder.")

        with output_dir_lock(self.config.output_dir):
            batch_started_at = utc_now_iso()
            cleanup_processor_temp_files(self.config.output_dir)
            output_dirs = allocate_output_dirs(files, self.config.output_dir)
            transcriber_metadata = getattr(self.transcriber, "metadata", {})
            self._emit(
                "batch_started",
                attempted=len(files),
                output_dir=str(self.config.output_dir),
                transcription_profile=transcriber_metadata.get("profile") or self.config.transcription_profile,
                transcription_engine=transcriber_metadata.get("resolved_engine") or self.config.transcription_engine.value,
                transcription_model=transcriber_metadata.get("model") or self.config.whisper_model,
            )
            results: List[FileResult] = []
            skip_files = set(self.config.skip_files) | self.control.skipped_files()
            process_files = []
            for path in files:
                processed_result = completed_output_result(path, output_dirs[path])
                if processed_result:
                    results.append(processed_result)
                    self._emit_file_finished(processed_result, results, len(files))
                elif path.name in skip_files:
                    result = self._skipped_file_result(path, output_dirs[path], "Skipped by user")
                    results.append(result)
                    self._emit_file_finished(result, results, len(files))
                else:
                    process_files.append(path)

            with ThreadPoolExecutor(max_workers=self.config.concurrent_files) as executor:
                futures = [executor.submit(self._process_file, path, output_dirs[path]) for path in process_files]
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    self._emit_file_finished(result, results, len(files))

            results.sort(key=lambda item: item.source.name.lower())
            summary = BatchSummary(
                attempted=len(results),
                completed=sum(1 for item in results if item.status is FileStatus.COMPLETED),
                failed=sum(1 for item in results if item.status is FileStatus.FAILED),
                skipped=sum(1 for item in results if item.status is FileStatus.SKIPPED),
                results=results,
                stopped=sum(1 for item in results if item.status is FileStatus.STOPPED),
            )
            write_batch_summary(self.config.output_dir, summary)
            write_batch_artifact(
                output_dir=self.config.output_dir,
                config=self.config,
                summary=summary,
                started_at=batch_started_at,
                finished_at=utc_now_iso(),
            )
            if self.config.render_html:
                render_batch_index(self.config.output_dir, summary)
            self._emit(
                "batch_finished",
                attempted=summary.attempted,
                completed=summary.completed,
                failed=summary.failed,
                skipped=summary.skipped,
                stopped=summary.stopped,
            )
            return summary

    def _process_file(self, source: Path, output_dir: Path) -> FileResult:
        started = time.monotonic()
        started_at = utc_now_iso()
        log_lines = [
            f"File:     {source.name}",
            f"Started:  {time.strftime('%Y-%m-%d %H:%M:%S')}",
            self._transcription_log_line(),
            "",
        ]
        temp_normalized: Optional[Path] = None
        temp_transcription_audio: Optional[Path] = None
        normalized_output: Optional[Path] = None
        normalized_duration = None
        media_info = None
        transcript = None
        lecture_json_path = None
        html_path = None
        enriched = False
        rendered = False
        title = None
        short_summary = None
        stage_timings: Dict[str, float] = {}
        step_state = {"current": "Probe"}

        try:
            if self.control.should_skip(source.name):
                return self._skipped_file_result(source, output_dir, "Skipped by user")
            if self.control.should_stop(source.name):
                raise ProcessingStopped("Stopped by user")

            self._emit("file_started", source=source.name)
            reset_output_dir(output_dir)
            media_info = self._time_step(
                "Probe",
                source,
                log_lines,
                step_state,
                stage_timings,
                lambda: self.inspector.probe(source),
            )
            self._emit(
                "file_media",
                source=source.name,
                source_duration_seconds=round(media_info.duration_seconds, 3),
                duration_seconds=round(_effective_duration_seconds(media_info.duration_seconds, self.config), 3),
            )
            if media_info.duration_seconds < self.config.min_duration_seconds:
                message = (
                    f"File too short ({media_info.duration_seconds:.0f}s < "
                    f"{self.config.min_duration_seconds:.0f}s minimum)"
                )
                log_lines.append(f"Skipped: {message}")
                elapsed = time.monotonic() - started
                lecture_json_path = write_lecture_artifact(
                    output_dir=output_dir,
                    source=source,
                    config=self.config,
                    status=FileStatus.SKIPPED,
                    started_at=started_at,
                    finished_at=utc_now_iso(),
                    elapsed_seconds=elapsed,
                    media_info=media_info,
                    stage_timings=stage_timings,
                    warnings=[message],
                )
                write_processing_log(output_dir, log_lines)
                return FileResult(
                    source=source,
                    output_dir=output_dir,
                    status=FileStatus.SKIPPED,
                    duration_seconds=media_info.duration_seconds,
                    elapsed_seconds=elapsed,
                    message=message,
                    lecture_json_path=lecture_json_path,
                )

            work_video = source
            transcription_input = work_video
            has_transcription = self.config.transcription_engine is not TranscriptionEngine.NONE
            if self.config.recording_speed is RecordingSpeed.DOUBLE:
                destination = normalized_video_output_path(source, output_dir)
                if not self.config.save_normalized_video:
                    destination = output_dir / ".normalized_work.mp4"
                    temp_normalized = destination
                if has_transcription:
                    temp_transcription_audio = output_dir / ".transcription_audio.wav"
                    work_video, transcription_input = self._prepare_double_speed_media(
                        source=source,
                        normalized_destination=destination,
                        transcription_audio_destination=temp_transcription_audio,
                        media_info=media_info,
                        log_lines=log_lines,
                        step_state=step_state,
                        stage_timings=stage_timings,
                    )
                    self._append_audio_command_log(log_lines, temp_transcription_audio)
                else:
                    work_video = self._time_step(
                        "Normalize",
                        source,
                        log_lines,
                        step_state,
                        stage_timings,
                        lambda: self.normalizer.normalize(
                            source=source,
                            destination=destination,
                            media_info=media_info,
                            recording_speed=self.config.recording_speed,
                            audio_quality=self.config.audio_quality,
                            stop_requested=lambda: self.control.should_stop(source.name),
                        ),
                    )
                if self.config.save_normalized_video:
                    normalized_output = work_video
                normalized_duration = media_info.duration_seconds * 2.0

            if self.config.recording_speed is not RecordingSpeed.DOUBLE and has_transcription:
                temp_transcription_audio = output_dir / ".transcription_audio.wav"
                transcription_input = self._time_step(
                    "Audio",
                    source,
                    log_lines,
                    step_state,
                    stage_timings,
                    lambda: self.audio_extractor.extract_for_transcription(
                        source=source,
                        destination=temp_transcription_audio,
                        media_info=media_info,
                        recording_speed=self.config.recording_speed,
                        audio_quality=self.config.audio_quality,
                        stop_requested=lambda: self.control.should_stop(source.name),
                    ),
                )
                log_lines[-1] = f"{log_lines[-1]} (16 kHz mono WAV)"
                self._append_audio_command_log(log_lines, temp_transcription_audio)

            transcript = self._time_step(
                "Transcribe",
                source,
                log_lines,
                step_state,
                stage_timings,
                lambda: self.transcriber.transcribe(transcription_input),
            )
            if self.control.should_stop(source.name):
                raise ProcessingStopped("Stopped by user")
            if temp_transcription_audio:
                remove_temp_path(temp_transcription_audio)
                temp_transcription_audio = None
            transcript = _scale_transcript(transcript, self.config.normalized_timestamp_scale)
            write_transcript(output_dir, transcript)
            log_lines[-1] = f"{log_lines[-1]} ({transcript.word_count} words)"

            slides_dir = output_dir / "slides"
            slide_count = self._time_step(
                "Slides",
                source,
                log_lines,
                step_state,
                stage_timings,
                lambda: self.slide_extractor.extract(
                    work_video,
                    slides_dir,
                    timestamp_scale=self.config.normalized_timestamp_scale,
                    stop_requested=lambda: self.control.should_stop(source.name),
                ),
            )
            log_lines[-1] = f"{log_lines[-1]} ({slide_count} slides extracted)"

            if temp_normalized:
                remove_temp_path(temp_normalized)
                temp_normalized = None

            elapsed = time.monotonic() - started
            lecture_json_path = write_lecture_artifact(
                output_dir=output_dir,
                source=source,
                config=self.config,
                status=FileStatus.COMPLETED,
                started_at=started_at,
                finished_at=utc_now_iso(),
                elapsed_seconds=elapsed,
                media_info=media_info,
                normalized_path=normalized_output,
                normalized_duration_seconds=normalized_duration,
                transcript=transcript,
                transcriber_metadata=getattr(self.transcriber, "metadata", {}),
                stage_timings=stage_timings,
            )
            if self.config.ai_provider.value != "none":
                artifact = self._time_step(
                    "Enrich",
                    source,
                    log_lines,
                    step_state,
                    stage_timings,
                    lambda: enrich_lecture_artifact(
                        lecture_json_path,
                        self.config,
                        progress_callback=self.progress_callback,
                    ),
                )
                enrichment = artifact.get("enrichment") or {}
                enriched = bool(enrichment)
                title = enrichment.get("title")
                short_summary = enrichment.get("executive_summary")
                log_lines[-1] = f"{log_lines[-1]} ({self.config.ai_provider.value})"

            if self.config.render_html:
                html_path = self._time_step(
                    "Render",
                    source,
                    log_lines,
                    step_state,
                    stage_timings,
                    lambda: render_lecture_page(lecture_json_path),
                )
                rendered = True

            elapsed = time.monotonic() - started
            log_lines.append(f"Complete in {elapsed:.1f}s")
            write_processing_log(output_dir, log_lines)
            return FileResult(
                source=source,
                output_dir=output_dir,
                status=FileStatus.COMPLETED,
                duration_seconds=media_info.duration_seconds,
                normalized_duration_seconds=normalized_duration,
                elapsed_seconds=elapsed,
                word_count=transcript.word_count,
                slide_count=slide_count,
                message="Complete",
                lecture_json_path=lecture_json_path,
                html_path=html_path,
                enriched=enriched,
                rendered=rendered,
                title=title,
                short_summary=short_summary,
            )
        except ProcessingStopped:
            if temp_transcription_audio:
                remove_temp_path(temp_transcription_audio)
                temp_transcription_audio = None
            if temp_normalized:
                remove_temp_path(temp_normalized)
                temp_normalized = None
            remove_temp_path(output_dir)
            elapsed = time.monotonic() - started
            return FileResult(
                source=source,
                output_dir=output_dir,
                status=FileStatus.STOPPED,
                duration_seconds=media_info.duration_seconds if media_info else 0.0,
                normalized_duration_seconds=normalized_duration,
                elapsed_seconds=elapsed,
                message="Stopped by user",
            )
        except Exception as exc:
            output_dir.mkdir(parents=True, exist_ok=True)
            step = step_state["current"]
            message = str(exc)
            log_lines.append(f"{step}  ERROR: {message}")
            if step == "Audio":
                self._append_audio_command_log(log_lines)
            log_lines.append("Partial output preserved for review. File marked failed.")
            elapsed = time.monotonic() - started
            try:
                lecture_json_path = write_lecture_artifact(
                    output_dir=output_dir,
                    source=source,
                    config=self.config,
                    status=FileStatus.FAILED,
                    started_at=started_at,
                    finished_at=utc_now_iso(),
                    elapsed_seconds=elapsed,
                    media_info=media_info,
                    normalized_path=normalized_output,
                    normalized_duration_seconds=normalized_duration,
                    transcript=transcript,
                    transcriber_metadata=getattr(self.transcriber, "metadata", {}),
                    stage_timings=stage_timings,
                    failure_step=step,
                    failure_message=message,
                )
            except Exception:
                lecture_json_path = None
            write_processing_log(output_dir, log_lines)
            return FileResult(
                source=source,
                output_dir=output_dir,
                status=FileStatus.FAILED,
                duration_seconds=media_info.duration_seconds if media_info else 0.0,
                normalized_duration_seconds=normalized_duration,
                elapsed_seconds=elapsed,
                word_count=transcript.word_count if transcript else 0,
                slide_count=len(list((output_dir / "slides").glob("*.png"))) if (output_dir / "slides").exists() else 0,
                failure_step=step,
                message=message,
                lecture_json_path=lecture_json_path,
            )
        finally:
            if temp_transcription_audio:
                remove_temp_path(temp_transcription_audio)
            if temp_normalized:
                remove_temp_path(temp_normalized)

    def _skipped_file_result(self, source: Path, output_dir: Path, message: str) -> FileResult:
        return FileResult(
            source=source,
            output_dir=output_dir,
            status=FileStatus.SKIPPED,
            message=message,
        )

    def _transcription_log_line(self) -> str:
        metadata = getattr(self.transcriber, "metadata", {}) or {}
        profile = metadata.get("profile_name") or metadata.get("profile") or self.config.transcription_profile
        engine = metadata.get("resolved_engine") or self.config.transcription_engine.value
        model = metadata.get("model") or self.config.whisper_model
        return f"Profile:  {profile} ({engine}, {model})"

    def _append_audio_command_log(self, log_lines: List[str], destination: Optional[Path] = None) -> None:
        command_text = ""
        if destination and hasattr(self.audio_extractor, "command_text_for"):
            command_text = self.audio_extractor.command_text_for(destination)
        if not command_text:
            command_text = getattr(self.audio_extractor, "last_command_text", "")
        if command_text and not any(line.startswith("Audio cmd:") for line in log_lines):
            log_lines.append(f"Audio cmd: {command_text}")

    def _prepare_double_speed_media(
        self,
        source: Path,
        normalized_destination: Path,
        transcription_audio_destination: Path,
        media_info,
        log_lines: List[str],
        step_state,
        stage_timings,
    ):
        def normalize(stop_requested):
            return self.normalizer.normalize(
                source=source,
                destination=normalized_destination,
                media_info=media_info,
                recording_speed=self.config.recording_speed,
                audio_quality=self.config.audio_quality,
                stop_requested=stop_requested,
            )

        def extract_audio(stop_requested):
            return self.audio_extractor.extract_for_transcription(
                source=source,
                destination=transcription_audio_destination,
                media_info=media_info,
                recording_speed=self.config.recording_speed,
                audio_quality=self.config.audio_quality,
                stop_requested=stop_requested,
            )

        results = self._time_steps_parallel(
            source=source,
            log_lines=log_lines,
            step_state=step_state,
            stage_timings=stage_timings,
            operations={
                "Normalize": normalize,
                "Audio": extract_audio,
            },
            log_suffixes={
                "Audio": " (16 kHz mono WAV)",
            },
        )
        return results["Normalize"], results["Audio"]

    def _time_step(self, name: str, source: Path, log_lines: List[str], step_state, stage_timings, operation):
        if self.control.should_stop(source.name):
            raise ProcessingStopped("Stopped by user")
        step_state["current"] = name
        self._emit("step_started", source=source.name, step=name)
        started = time.monotonic()
        result = operation()
        if self.control.should_stop(source.name):
            raise ProcessingStopped("Stopped by user")
        elapsed = time.monotonic() - started
        stage_timings[name] = round(elapsed, 3)
        log_lines.append(f"{name:<10} OK  {elapsed:.1f}s")
        self._emit("step_finished", source=source.name, step=name, elapsed_seconds=round(elapsed, 1))
        return result

    def _time_steps_parallel(
        self,
        source: Path,
        log_lines: List[str],
        step_state,
        stage_timings,
        operations: Dict[str, Callable[[Callable[[], bool]], object]],
        log_suffixes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, object]:
        cancel_requested = threading.Event()
        step_lock = threading.Lock()
        log_suffixes = log_suffixes or {}

        def should_stop() -> bool:
            return cancel_requested.is_set() or self.control.should_stop(source.name)

        def run_step(name: str, operation: Callable[[Callable[[], bool]], object]):
            if should_stop():
                raise ProcessingStopped("Stopped by user")
            with step_lock:
                step_state["current"] = name
            self._emit("step_started", source=source.name, step=name)
            started = time.monotonic()
            try:
                result = operation(should_stop)
            except Exception:
                cancel_requested.set()
                raise
            if should_stop():
                cancel_requested.set()
                raise ProcessingStopped("Stopped by user")
            elapsed = time.monotonic() - started
            self._emit("step_finished", source=source.name, step=name, elapsed_seconds=round(elapsed, 1))
            return name, result, elapsed

        results: Dict[str, object] = {}
        successes: Dict[str, float] = {}
        failures = []
        with ThreadPoolExecutor(max_workers=len(operations)) as executor:
            futures = {
                executor.submit(run_step, name, operation): name
                for name, operation in operations.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    completed_name, result, elapsed = future.result()
                    results[completed_name] = result
                    successes[completed_name] = elapsed
                except Exception as exc:
                    cancel_requested.set()
                    failures.append((name, exc))

        for name in operations:
            if name in successes:
                elapsed = successes[name]
                stage_timings[name] = round(elapsed, 3)
                log_lines.append(f"{name:<10} OK  {elapsed:.1f}s{log_suffixes.get(name, '')}")

        if failures:
            failed_name, exc = next(
                ((name, error) for name, error in failures if not isinstance(error, ProcessingStopped)),
                failures[0],
            )
            step_state["current"] = failed_name
            raise exc

        return results

    def _emit_file_finished(self, result: FileResult, results: List[FileResult], attempted: int) -> None:
        self._emit(
            "file_finished",
            source=result.source.name,
            status=result.status.value,
            message=result.message,
            completed=sum(1 for item in results if item.status is FileStatus.COMPLETED),
            failed=sum(1 for item in results if item.status is FileStatus.FAILED),
            skipped=sum(1 for item in results if item.status is FileStatus.SKIPPED),
            stopped=sum(1 for item in results if item.status is FileStatus.STOPPED),
            attempted=len(results),
            total=attempted,
            word_count=result.word_count,
            slide_count=result.slide_count,
            duration_seconds=round(_result_effective_duration_seconds(result), 3),
            source_duration_seconds=round(result.duration_seconds, 3),
            normalized_duration_seconds=(
                round(result.normalized_duration_seconds, 3)
                if result.normalized_duration_seconds is not None
                else None
            ),
            lecture_duration_seconds=round(_result_effective_duration_seconds(result), 3),
            elapsed_seconds=round(result.elapsed_seconds, 3),
            enriched=result.enriched,
            rendered=result.rendered,
            failure_step=result.failure_step,
        )

    def _emit(self, kind: str, **payload) -> None:
        if not self.progress_callback:
            return
        event = {"kind": kind, **payload}
        try:
            self.progress_callback(event)
        except Exception:
            pass


def discover_mov_files(folder: Path) -> List[Path]:
    return sorted(
        (item for item in folder.iterdir() if item.is_file() and item.suffix.lower() == ".mov"),
        key=lambda item: item.name.lower(),
    )


def _effective_duration_seconds(duration_seconds: float, config: BatchConfig) -> float:
    if config.recording_speed is RecordingSpeed.DOUBLE:
        return duration_seconds * 2.0
    return duration_seconds


def _result_effective_duration_seconds(result: FileResult) -> float:
    return result.normalized_duration_seconds or result.duration_seconds


def discover_lecture_artifacts(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    root_artifact = folder / LECTURE_ARTIFACT_NAME
    if root_artifact.exists():
        return [root_artifact]
    return sorted(
        (item / LECTURE_ARTIFACT_NAME for item in folder.iterdir() if (item / LECTURE_ARTIFACT_NAME).exists()),
        key=lambda item: item.parent.name.lower(),
    )


def completed_output_result(source: Path, output_dir: Path) -> Optional[FileResult]:
    artifact = _load_completed_artifact(output_dir)
    if not artifact:
        return None
    artifact_source = str(artifact.get("source", {}).get("filename") or "")
    if artifact_source and artifact_source != source.name:
        return None
    return _file_result_from_artifact(
        source=source,
        output_dir=output_dir,
        artifact=artifact,
        status=FileStatus.SKIPPED,
        message=ALREADY_PROCESSED_MESSAGE,
    )


def enrich_processed_batch(
    config: BatchConfig,
    *,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> BatchSummary:
    config.validate()
    artifacts = discover_lecture_artifacts(config.input_dir)
    if not artifacts:
        raise LectureProcessorError("No processed lectures found. Choose a folder with lecture.json files.")

    with output_dir_lock(config.output_dir):
        batch_started_at = utc_now_iso()
        cleanup_processor_temp_files(config.output_dir)
        results: List[FileResult] = []
        skip_files = set(config.skip_files)
        _emit(progress_callback, "batch_started", attempted=len(artifacts), output_dir=str(config.output_dir))
        for lecture_json_path in artifacts:
            result = _enrich_processed_lecture(lecture_json_path, config, skip_files, progress_callback)
            results.append(result)
            _emit_file_finished(progress_callback, result, results, len(artifacts))

        results.sort(key=lambda item: item.source.name.lower())
        summary = BatchSummary(
            attempted=len(results),
            completed=sum(1 for item in results if item.status is FileStatus.COMPLETED),
            failed=sum(1 for item in results if item.status is FileStatus.FAILED),
            skipped=sum(1 for item in results if item.status is FileStatus.SKIPPED),
            stopped=sum(1 for item in results if item.status is FileStatus.STOPPED),
            results=results,
        )
        write_batch_summary(config.output_dir, summary)
        write_batch_artifact(
            output_dir=config.output_dir,
            config=config,
            summary=summary,
            started_at=batch_started_at,
            finished_at=utc_now_iso(),
        )
        if config.render_html:
            render_batch_index(config.output_dir, summary)
        _emit(
            progress_callback,
            "batch_finished",
            attempted=summary.attempted,
            completed=summary.completed,
            failed=summary.failed,
            skipped=summary.skipped,
            stopped=summary.stopped,
        )
        return summary


def _enrich_processed_lecture(
    lecture_json_path: Path,
    config: BatchConfig,
    skip_files: set,
    progress_callback: Optional[Callable[[Dict], None]],
) -> FileResult:
    started = time.monotonic()
    output_dir = lecture_json_path.parent
    artifact = load_json(lecture_json_path)
    source = _source_path_from_artifact(artifact, output_dir)
    if source.name in skip_files or output_dir.name in skip_files:
        return _file_result_from_artifact(
            source=source,
            output_dir=output_dir,
            artifact=artifact,
            status=FileStatus.SKIPPED,
            message="Skipped by user",
            elapsed_seconds=time.monotonic() - started,
        )
    if not _artifact_can_be_enriched(artifact):
        return _file_result_from_artifact(
            source=source,
            output_dir=output_dir,
            artifact=artifact,
            status=FileStatus.SKIPPED,
            message="Not completed",
            elapsed_seconds=time.monotonic() - started,
        )
    if artifact.get("enrichment"):
        return _file_result_from_artifact(
            source=source,
            output_dir=output_dir,
            artifact=artifact,
            status=FileStatus.SKIPPED,
            message=ALREADY_ENHANCED_MESSAGE,
            elapsed_seconds=time.monotonic() - started,
        )

    _emit(progress_callback, "file_started", source=source.name)
    try:
        _emit(progress_callback, "step_started", source=source.name, step="Enrich")
        step_started = time.monotonic()
        enriched_artifact = enrich_lecture_artifact(lecture_json_path, config, progress_callback=progress_callback)
        _emit(
            progress_callback,
            "step_finished",
            source=source.name,
            step="Enrich",
            elapsed_seconds=round(time.monotonic() - step_started, 1),
        )
        html_path = None
        rendered = False
        if config.render_html:
            _emit(progress_callback, "step_started", source=source.name, step="Render")
            step_started = time.monotonic()
            html_path = render_lecture_page(lecture_json_path)
            rendered = True
            _emit(
                progress_callback,
                "step_finished",
                source=source.name,
                step="Render",
                elapsed_seconds=round(time.monotonic() - step_started, 1),
            )
        return _file_result_from_artifact(
            source=source,
            output_dir=output_dir,
            artifact=enriched_artifact,
            status=FileStatus.COMPLETED,
            message="Enhanced",
            html_path=html_path,
            rendered=rendered,
            elapsed_seconds=time.monotonic() - started,
        )
    except Exception as exc:
        return _file_result_from_artifact(
            source=source,
            output_dir=output_dir,
            artifact=artifact,
            status=FileStatus.FAILED,
            message=str(exc),
            failure_step="Enrich",
            elapsed_seconds=time.monotonic() - started,
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


def normalized_video_output_path(source: Path, output_dir: Path) -> Path:
    return output_dir / f"{safe_folder_name(source.stem)}.mp4"


def _load_completed_artifact(output_dir: Path) -> Optional[Dict]:
    path = output_dir / LECTURE_ARTIFACT_NAME
    if not path.exists():
        return None
    try:
        artifact = load_json(path)
    except Exception:
        return None
    if artifact.get("processing", {}).get("status") != FileStatus.COMPLETED.value:
        return None
    return artifact


def _file_result_from_artifact(
    *,
    source: Path,
    output_dir: Path,
    artifact: Dict,
    status: FileStatus,
    message: str,
    html_path: Optional[Path] = None,
    rendered: Optional[bool] = None,
    elapsed_seconds: float = 0.0,
    failure_step: Optional[str] = None,
) -> FileResult:
    enrichment = artifact.get("enrichment") or {}
    transcript = artifact.get("transcript") or {}
    slides = artifact.get("slides") or []
    lecture_json_path = output_dir / LECTURE_ARTIFACT_NAME
    existing_html_path = output_dir / "html" / "index.html"
    html_path = html_path or (existing_html_path if existing_html_path.exists() else None)
    rendered_value = rendered if rendered is not None else html_path is not None
    return FileResult(
        source=source,
        output_dir=output_dir,
        status=status,
        duration_seconds=float(artifact.get("media", {}).get("duration_seconds") or 0.0),
        normalized_duration_seconds=artifact.get("media", {}).get("normalized_duration_seconds"),
        elapsed_seconds=elapsed_seconds,
        word_count=int(transcript.get("word_count") or len(str(transcript.get("text") or "").split())),
        slide_count=len(slides),
        failure_step=failure_step,
        message=message,
        lecture_json_path=lecture_json_path if lecture_json_path.exists() else None,
        html_path=html_path,
        enriched=bool(enrichment),
        rendered=rendered_value,
        title=enrichment.get("title"),
        short_summary=enrichment.get("executive_summary"),
    )


def _artifact_can_be_enriched(artifact: Dict) -> bool:
    processing = artifact.get("processing") or {}
    if processing.get("status") == FileStatus.COMPLETED.value:
        return True
    if processing.get("status") != FileStatus.FAILED.value:
        return False
    if processing.get("failure_step") != "Enrich":
        return False
    transcript = artifact.get("transcript") or {}
    return bool(transcript.get("text") or transcript.get("segments") or artifact.get("slides"))


def _source_path_from_artifact(artifact: Dict, output_dir: Path) -> Path:
    source = artifact.get("source") or {}
    filename = source.get("filename") or f"{output_dir.name}.mov"
    absolute_path = source.get("absolute_path")
    if absolute_path:
        return Path(str(absolute_path))
    return output_dir / str(filename)


def _emit(progress_callback: Optional[Callable[[Dict], None]], kind: str, **payload) -> None:
    if not progress_callback:
        return
    try:
        progress_callback({"kind": kind, **payload})
    except Exception:
        pass


def _emit_file_finished(
    progress_callback: Optional[Callable[[Dict], None]],
    result: FileResult,
    results: List[FileResult],
    attempted: int,
) -> None:
    _emit(
        progress_callback,
        "file_finished",
        source=result.source.name,
        status=result.status.value,
        message=result.message,
        completed=sum(1 for item in results if item.status is FileStatus.COMPLETED),
        failed=sum(1 for item in results if item.status is FileStatus.FAILED),
        skipped=sum(1 for item in results if item.status is FileStatus.SKIPPED),
        stopped=sum(1 for item in results if item.status is FileStatus.STOPPED),
        attempted=len(results),
        total=attempted,
        word_count=result.word_count,
        slide_count=result.slide_count,
        enriched=result.enriched,
        rendered=result.rendered,
        failure_step=result.failure_step,
    )


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


@contextmanager
def output_dir_lock(output_dir: Path) -> Iterator[None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / OUTPUT_LOCK_FILE

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            lock_info = _read_output_lock(lock_path)
            pid = _lock_pid(lock_info)
            if _pid_is_running(pid):
                raise LectureProcessorError(
                    "This output folder is already being processed. Wait for the current batch to finish, "
                    "cancel it, or choose a different output folder."
                )
            try:
                lock_path.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise LectureProcessorError(f"Could not clear a stale output-folder lock: {exc}") from exc
            continue
        break

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
            json.dump({"pid": os.getpid(), "created_at": time.time()}, lock_file)
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _read_output_lock(lock_path: Path) -> Dict:
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _lock_pid(lock_info: Dict) -> Optional[int]:
    pid = lock_info.get("pid")
    return pid if isinstance(pid, int) and pid > 0 else None


def _pid_is_running(pid: Optional[int]) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def write_batch_summary(output_dir: Path, summary: BatchSummary) -> None:
    lines = [
        "Batch summary",
        f"Attempted: {summary.attempted}",
        f"Completed: {summary.completed}",
        f"Failed:    {summary.failed}",
        f"Skipped:   {summary.skipped}",
        f"Stopped:   {summary.stopped}",
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
    write_text_atomic(output_dir / "batch_summary.txt", "\n".join(lines) + "\n")


def _scale_transcript(transcript: TranscriptResult, scale: float) -> TranscriptResult:
    return transcript.scaled(scale)
