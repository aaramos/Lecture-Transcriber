import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from .config import AudioQuality, RecordingSpeed
from .errors import DependencyMissingError, ProcessingError
from .models import MediaInfo


def ensure_media_tools(ffprobe_path: str, ffmpeg_path: str, needs_ffmpeg: bool) -> None:
    _require_command(ffprobe_path)
    if needs_ffmpeg:
        _require_command(ffmpeg_path)


def _require_command(command: str) -> str:
    if "/" in command:
        if Path(command).exists():
            return command
        raise DependencyMissingError(f"Required command not found: {command}")
    local = _local_tool_candidate(command)
    if local:
        return str(local)
    resolved = shutil.which(command)
    if not resolved:
        raise DependencyMissingError(f"Required command not found on PATH: {command}")
    return resolved


def _local_tool_candidate(command: str) -> Optional[Path]:
    if command not in {"ffmpeg", "ffprobe"}:
        return None

    search_roots = [Path.cwd(), Path(__file__).resolve()]
    for root in search_roots:
        for parent in [root, *root.parents]:
            candidate = parent / ".tools" / "darwin_arm64" / command
            if candidate.exists():
                return candidate
    return None


def _parse_fraction(value: Optional[str]) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" not in value:
        try:
            return float(value)
        except ValueError:
            return 0.0
    numerator, denominator = value.split("/", 1)
    try:
        denominator_f = float(denominator)
        if denominator_f == 0:
            return 0.0
        return float(numerator) / denominator_f
    except ValueError:
        return 0.0


class MediaInspector:
    def __init__(self, ffprobe_path: str = "ffprobe") -> None:
        self.ffprobe_path = ffprobe_path

    def probe(self, path: Path) -> MediaInfo:
        ffprobe = _require_command(self.ffprobe_path)
        command = [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise ProcessingError(
                f"ffprobe failed for {path.name}: {completed.stderr.strip() or completed.stdout.strip()}"
            )

        payload = json.loads(completed.stdout)
        duration = _extract_duration(payload)
        video_stream = _first_video_stream(payload)
        avg_frame_rate = _parse_fraction(video_stream.get("avg_frame_rate"))
        real_frame_rate = _parse_fraction(video_stream.get("r_frame_rate"))
        is_vfr = bool(
            avg_frame_rate
            and real_frame_rate
            and abs(avg_frame_rate - real_frame_rate) > 0.01
        )

        return MediaInfo(
            path=path,
            duration_seconds=duration,
            avg_frame_rate=avg_frame_rate,
            real_frame_rate=real_frame_rate,
            is_vfr=is_vfr,
        )


class MediaNormalizer:
    def __init__(self, ffmpeg_path: str = "ffmpeg") -> None:
        self.ffmpeg_path = ffmpeg_path

    def normalize(
        self,
        source: Path,
        destination: Path,
        media_info: MediaInfo,
        recording_speed: RecordingSpeed,
        audio_quality: AudioQuality,
    ) -> Path:
        if recording_speed is RecordingSpeed.NORMAL:
            return source

        ffmpeg = _require_command(self.ffmpeg_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        video_filter = "setpts=2.0*PTS"
        if media_info.is_vfr:
            video_filter = f"fps={media_info.intended_frame_rate:.3f},setpts=2.0*PTS"

        if audio_quality is AudioQuality.HIGH:
            audio_filter = "rubberband=tempo=0.5"
        else:
            audio_filter = "atempo=0.5"

        command = [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-filter_complex",
            f"[0:v]{video_filter}[v];[0:a]{audio_filter}[a]",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            str(destination),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise ProcessingError(
                f"ffmpeg normalization failed for {source.name}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        return destination


def _extract_duration(payload: Dict[str, Any]) -> float:
    format_data = payload.get("format") or {}
    duration = format_data.get("duration")
    if duration:
        return float(duration)
    for stream in payload.get("streams", []):
        if stream.get("duration"):
            return float(stream["duration"])
    raise ProcessingError("Could not determine media duration")


def _first_video_stream(payload: Dict[str, Any]) -> Dict[str, Any]:
    for stream in payload.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    raise ProcessingError("No video stream found")
