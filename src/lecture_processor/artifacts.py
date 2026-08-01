import hashlib
import json
import platform
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import BatchConfig, RecordingSpeed
from .models import BatchSummary, FileResult, FileStatus, MediaInfo, TranscriptResult
from .writers import write_json_atomic

SCHEMA_VERSION = "1.0.0"
LECTURE_ARTIFACT_NAME = "lecture.json"
BATCH_ARTIFACT_NAME = "batch.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_lecture_artifact(
    *,
    output_dir: Path,
    source: Path,
    config: BatchConfig,
    status: FileStatus,
    started_at: str,
    finished_at: str,
    elapsed_seconds: float,
    media_info: Optional[MediaInfo] = None,
    normalized_path: Optional[Path] = None,
    normalized_duration_seconds: Optional[float] = None,
    transcript: Optional[TranscriptResult] = None,
    transcriber_metadata: Optional[Dict] = None,
    stage_timings: Optional[Dict[str, float]] = None,
    warnings: Optional[List[str]] = None,
    failure_step: Optional[str] = None,
    failure_message: Optional[str] = None,
) -> Path:
    artifact = build_lecture_artifact(
        output_dir=output_dir,
        source=source,
        config=config,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=elapsed_seconds,
        media_info=media_info,
        normalized_path=normalized_path,
        normalized_duration_seconds=normalized_duration_seconds,
        transcript=transcript,
        transcriber_metadata=transcriber_metadata,
        stage_timings=stage_timings,
        warnings=warnings,
        failure_step=failure_step,
        failure_message=failure_message,
    )
    path = output_dir / LECTURE_ARTIFACT_NAME
    write_json_atomic(path, artifact)
    return path


def build_lecture_artifact(
    *,
    output_dir: Path,
    source: Path,
    config: BatchConfig,
    status: FileStatus,
    started_at: str,
    finished_at: str,
    elapsed_seconds: float,
    media_info: Optional[MediaInfo] = None,
    normalized_path: Optional[Path] = None,
    normalized_duration_seconds: Optional[float] = None,
    transcript: Optional[TranscriptResult] = None,
    transcriber_metadata: Optional[Dict] = None,
    stage_timings: Optional[Dict[str, float]] = None,
    warnings: Optional[List[str]] = None,
    failure_step: Optional[str] = None,
    failure_message: Optional[str] = None,
) -> Dict:
    transcript = transcript or TranscriptResult(text="", segments=[])
    transcriber_metadata = transcriber_metadata or {}
    media_duration = media_info.duration_seconds if media_info else 0.0
    source_stat = _stat_or_none(source)
    normalized_relative = _relative_or_none(normalized_path, output_dir)
    slides = build_slide_records(output_dir / "slides", transcript)
    warnings = list(warnings or [])
    if status is FileStatus.FAILED and failure_message:
        warnings.append(f"{failure_step or 'Processing'} failed: {failure_message}")

    return {
        "schema_version": SCHEMA_VERSION,
        "lecture_id": output_dir.name,
        "source": {
            "filename": source.name,
            "absolute_path": str(source.resolve()),
            "byte_size": source_stat["byte_size"],
            "sha256": _sha256_file(source),
            "modified_at": source_stat["modified_at"],
        },
        "media": {
            "duration_seconds": media_duration,
            "intended_frame_rate": media_info.intended_frame_rate if media_info else 0.0,
            "is_vfr": bool(media_info.is_vfr) if media_info else False,
            "has_audio": bool(media_info.has_audio) if media_info else False,
            "has_video": bool(media_info.has_video) if media_info else False,
            "recording_speed_input": config.recording_speed.value,
            "recording_speed_output": _recording_speed_output(config),
            "normalized_path": normalized_relative,
            "normalized_duration_seconds": normalized_duration_seconds,
        },
        "transcript": {
            "engine": _resolved_transcription_engine(transcriber_metadata, config),
            "model": str(transcriber_metadata.get("model") or config.whisper_model or ""),
            "profile": str(transcriber_metadata.get("profile") or config.transcription_profile or ""),
            "audio_enhancement": config.audio_enhancement.value,
            "quality": str(transcriber_metadata.get("quality") or config.transcription_quality.value),
            "compute_type": str(transcriber_metadata.get("compute_type") or ""),
            "coreml_used": bool(transcriber_metadata.get("coreml_used", False)),
            "language": str(transcriber_metadata.get("language") or "en"),
            "word_count": transcript.word_count,
            "text": transcript.text,
            "segments": [
                {
                    "id": index,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
                for index, segment in enumerate(transcript.segments)
            ],
        },
        "slides": slides,
        "processing": {
            "status": status.value,
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "host": _host_info(config),
            "stage_timings": {key: round(value, 3) for key, value in (stage_timings or {}).items()},
            "warnings": warnings,
            "failure_step": failure_step,
            "failure_message": failure_message,
        },
        "enrichment": None,
    }


def build_slide_records(slides_dir: Path, transcript: TranscriptResult) -> List[Dict]:
    if not slides_dir.exists():
        return []
    records = []
    slide_paths = sorted(slides_dir.glob("*.png"))
    slide_timestamps = [_slide_timestamp(path.name) for path in slide_paths]
    for index, path in enumerate(slide_paths, start=1):
        timestamp = slide_timestamps[index - 1]
        next_timestamp = slide_timestamps[index] if index < len(slide_timestamps) else None
        metadata = _slide_metadata(path)
        records.append(
            {
                "id": index,
                "filename": path.name,
                "relative_path": f"slides/{path.name}",
                "thumbnail_path": None,
                "timestamp_seconds": timestamp,
                "title": metadata.get("title"),
                "build_stage": metadata.get("build_stage"),
                "layout": metadata.get("layout"),
                "description": metadata.get("description"),
                "slide_analysis": None,
                "image": {
                    "width": _image_size(path)[0],
                    "height": _image_size(path)[1],
                    "byte_size": _stat_or_none(path)["byte_size"],
                    "sha256": _sha256_file(path),
                },
                "linked_segment_ids": _linked_segment_ids(transcript, timestamp, next_timestamp),
            }
        )
    return records


def _slide_metadata(path: Path) -> Dict:
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        "title": _clean_optional_string(payload.get("title")),
        "build_stage": _clean_optional_string(payload.get("build_stage")),
        "layout": _clean_optional_string(payload.get("layout")),
        "description": _clean_optional_string(payload.get("description")),
    }


def _clean_optional_string(value) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def write_batch_artifact(
    *,
    output_dir: Path,
    config: BatchConfig,
    summary: BatchSummary,
    started_at: str,
    finished_at: str,
) -> Path:
    artifact = build_batch_artifact(
        output_dir=output_dir,
        config=config,
        summary=summary,
        started_at=started_at,
        finished_at=finished_at,
    )
    path = output_dir / BATCH_ARTIFACT_NAME
    write_json_atomic(path, artifact)
    return path


def build_batch_artifact(
    *,
    output_dir: Path,
    config: BatchConfig,
    summary: BatchSummary,
    started_at: str,
    finished_at: str,
) -> Dict:
    lectures = [_batch_lecture_record(result, output_dir) for result in summary.results]
    return {
        "schema_version": SCHEMA_VERSION,
        "batch_id": output_dir.name,
        "input_dir": str(config.input_dir),
        "output_dir": str(output_dir),
        "started_at": started_at,
        "finished_at": finished_at,
        "config": {
            "recording_speed": config.recording_speed.value,
            "audio_quality": config.audio_quality.value,
            "audio_enhancement": config.audio_enhancement.value,
            "transcription_profile": config.transcription_profile,
            "transcription_engine": config.transcription_engine.value,
            "transcription_quality": config.transcription_quality.value,
            "whisper_model": config.whisper_model,
            "slide_sensitivity": config.slide_sensitivity.value,
            "concurrent_files": config.concurrent_files,
            "ai_provider": config.ai_provider.value,
            "ai_model": config.ai_model or None,
            "ai_model_routing": config.ai_model_routing,
            "gemini_max_concurrency": config.gemini_max_concurrency,
            "mlx_text_base_url": config.mlx_text_base_url,
            "mlx_vision_base_url": config.mlx_vision_base_url,
            "mlx_request_timeout_seconds": config.mlx_request_timeout_seconds,
            "mlx_disable_thinking": config.mlx_disable_thinking,
            "render_html": config.render_html,
        },
        "lectures": lectures,
        "summary": {
            "attempted": summary.attempted,
            "completed": summary.completed,
            "failed": summary.failed,
            "skipped": summary.skipped,
            "stopped": summary.stopped,
            "enriched": sum(1 for result in summary.results if result.enriched),
            "rendered": sum(1 for result in summary.results if result.rendered),
        },
    }


def load_json(path: Path) -> Dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Dict) -> None:
    write_json_atomic(path, payload)


def _batch_lecture_record(result: FileResult, output_root: Path) -> Dict:
    html_path = _relative_or_none(result.html_path, output_root)
    lecture_json_path = _relative_or_none(result.lecture_json_path, output_root) or str(
        result.output_dir.relative_to(output_root) / LECTURE_ARTIFACT_NAME
    )
    return {
        "lecture_id": result.output_dir.name,
        "source_filename": result.source.name,
        "status": result.status.value,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "word_count": result.word_count,
        "slide_count": result.slide_count,
        "enriched": result.enriched,
        "rendered": result.rendered,
        "title": result.title,
        "short_summary": result.short_summary,
        "lecture_json_path": lecture_json_path,
        "html_path": html_path,
        "thumbnail_relative_path": _first_slide_path(result.output_dir, output_root),
        "duration_minutes": round(_result_duration_seconds(result) / 60.0, 2) if _result_duration_seconds(result) else None,
        "failure_step": result.failure_step,
        "failure_message": result.message if result.status in (FileStatus.FAILED, FileStatus.STOPPED) else None,
    }


def _stat_or_none(path: Path) -> Dict:
    try:
        stat = path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc).replace(microsecond=0)
        return {
            "byte_size": stat.st_size,
            "modified_at": modified.isoformat().replace("+00:00", "Z"),
        }
    except OSError:
        return {
            "byte_size": 0,
            "modified_at": utc_now_iso(),
        }


def _result_duration_seconds(result: FileResult) -> float:
    return result.normalized_duration_seconds or result.duration_seconds


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        digest.update(b"")
    return digest.hexdigest()


def _image_size(path: Path) -> tuple:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return 1, 1


def _slide_timestamp(filename: str) -> float:
    match = re.search(r"_(\d{2})-(\d{2})-(\d{2})(?:\D|$)", filename)
    if not match:
        return 0.0
    hours, minutes, seconds = (int(part) for part in match.groups())
    return float(hours * 3600 + minutes * 60 + seconds)


def _linked_segment_ids(transcript: TranscriptResult, timestamp: float, next_timestamp: Optional[float] = None) -> List[int]:
    if not transcript.segments:
        return []
    window_start = max(0.0, float(timestamp or 0.0))
    transcript_end = max((segment.end for segment in transcript.segments), default=window_start)
    window_end = float(next_timestamp) if next_timestamp is not None and next_timestamp > window_start else transcript_end
    if window_end <= window_start:
        window_end = window_start
    linked = []
    for index, segment in enumerate(transcript.segments):
        if segment.end > window_start and segment.start < window_end:
            linked.append(index)
    if linked:
        return linked
    fallback_id = _segment_id_at_or_after(transcript, window_start)
    return [fallback_id] if fallback_id is not None else []


def _segment_id_at_or_after(transcript: TranscriptResult, timestamp: float) -> Optional[int]:
    for index, segment in enumerate(transcript.segments):
        if segment.start <= timestamp <= segment.end:
            return index
    for index, segment in enumerate(transcript.segments):
        if segment.start >= timestamp:
            return index
    if transcript.segments:
        return len(transcript.segments) - 1
    return None


def _resolved_transcription_engine(metadata: Dict, config: BatchConfig) -> str:
    engine = metadata.get("resolved_engine") or metadata.get("engine") or config.transcription_engine.value
    if engine == "auto":
        return "none" if config.transcription_engine.value == "none" else "faster-whisper"
    return str(engine)


def _recording_speed_output(config: BatchConfig) -> str:
    if config.recording_speed is RecordingSpeed.DOUBLE and config.save_normalized_video:
        return "1x"
    return config.recording_speed.value


def _host_info(config: BatchConfig) -> Dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "apple_silicon": bool(config.apple_silicon),
        "chip": _chip_name(),
        "ram_gb": None,
        "hostname": socket.gethostname(),
    }


def _chip_name() -> Optional[str]:
    if platform.system() != "Darwin":
        return None
    try:
        import subprocess

        completed = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            value = completed.stdout.strip()
            return value or None
    except OSError:
        return None
    return None


def _relative_or_none(path: Optional[Path], root: Path) -> Optional[str]:
    if not path:
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _first_slide_path(output_dir: Path, output_root: Path) -> Optional[str]:
    slides = sorted((output_dir / "slides").glob("*.png"))
    if not slides:
        return None
    return _relative_or_none(slides[0], output_root)
