import json
import shutil
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .config import AudioQuality, FfmpegHwAccel, RecordingSpeed
from .errors import DependencyMissingError, ProcessingError, ProcessingStopped
from .models import MediaInfo
from .temp_cleanup import remove_temp_path


def ensure_media_tools(ffprobe_path: str, ffmpeg_path: str, needs_ffmpeg: bool) -> None:
    resolve_media_tool(ffprobe_path)
    if needs_ffmpeg:
        resolve_media_tool(ffmpeg_path)


def resolve_media_tool(command: str) -> str:
    return _require_command(command)


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
        has_audio = _has_audio_stream(payload)
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
            has_audio=has_audio,
        )


class MediaNormalizer:
    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        runner: Callable = subprocess.run,
        ffmpeg_hwaccel: FfmpegHwAccel = FfmpegHwAccel.AUTO,
        apple_silicon: bool = False,
    ) -> None:
        self.ffmpeg_path = ffmpeg_path
        self._runner = runner
        self.ffmpeg_hwaccel = ffmpeg_hwaccel
        self.apple_silicon = apple_silicon

    def normalize(
        self,
        source: Path,
        destination: Path,
        media_info: MediaInfo,
        recording_speed: RecordingSpeed,
        audio_quality: AudioQuality,
        stop_requested: Callable[[], bool] = None,
    ) -> Path:
        if recording_speed is RecordingSpeed.NORMAL:
            return source

        ffmpeg = _require_command(self.ffmpeg_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_destination = destination.with_name(f".{destination.name}.ffmpeg.tmp")
        video_filter = "setpts=2.0*PTS"
        if media_info.is_vfr:
            video_filter = f"fps={media_info.intended_frame_rate:.3f},setpts=2.0*PTS"

        maps = ["-map", "[v]"]
        audio_filter = None
        if media_info.has_audio:
            audio_filter = _audio_filter_for(ffmpeg, audio_quality)
            maps.extend(["-map", "[a]"])

        filter_complex = f"[0:v:0]{video_filter}[v]"
        if audio_filter:
            filter_complex = f"{filter_complex};[0:a:0]{audio_filter}[a]"

        hwaccel_args = _hwaccel_args(self.ffmpeg_hwaccel, self.apple_silicon)
        def build_command(output_path: Path, hardware_args: list) -> list:
            command = [
                ffmpeg,
                "-y",
                *hardware_args,
                "-i",
                str(source),
                "-filter_complex",
                filter_complex,
                *maps,
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
            ]
            if audio_filter:
                command.extend(["-c:a", "aac"])
            else:
                command.append("-an")
            command.extend([
                "-movflags",
                "+faststart",
                "-f",
                "mp4",
                str(output_path),
            ])
            return command

        try:
            completed = _run_interruptible(
                build_command(temp_destination, hwaccel_args),
                self._runner,
                stop_requested,
            )
            if completed.returncode != 0 and hwaccel_args and self.ffmpeg_hwaccel is FfmpegHwAccel.AUTO:
                remove_temp_path(temp_destination)
                completed = _run_interruptible(
                    build_command(temp_destination, []),
                    self._runner,
                    stop_requested,
                )
            if completed.returncode != 0:
                raise ProcessingError(
                    f"ffmpeg normalization failed for {source.name}: "
                    f"{completed.stderr.strip() or completed.stdout.strip()}"
                )
            temp_destination.replace(destination)
        finally:
            remove_temp_path(temp_destination)
        return destination


class CleanAudioExtractor:
    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        runner: Callable = subprocess.run,
        filter_available: Callable[[str, str], bool] = None,
    ) -> None:
        self.ffmpeg_path = ffmpeg_path
        self._runner = runner
        self._filter_available = filter_available

    def extract(
        self,
        source: Path,
        destination: Path,
        media_info: MediaInfo,
        stop_requested: Callable[[], bool] = None,
    ) -> Path:
        if not media_info.has_audio:
            raise ProcessingError(f"No audio stream found for transcription in {source.name}")

        ffmpeg = _require_command(self.ffmpeg_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_destination = destination.with_name(f".{destination.name}.ffmpeg.tmp")
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
        ]
        audio_filter = _clean_audio_filter_for(ffmpeg, self._filter_available)
        if audio_filter:
            command.extend(["-af", audio_filter])
        command.extend([
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(temp_destination),
        ])

        try:
            completed = _run_interruptible(command, self._runner, stop_requested)
            if completed.returncode != 0:
                raise ProcessingError(
                    f"ffmpeg audio extraction failed for {source.name}: "
                    f"{completed.stderr.strip() or completed.stdout.strip()}"
                )
            temp_destination.replace(destination)
        finally:
            remove_temp_path(temp_destination)
        return destination


def _run_interruptible(command: list, runner: Callable, stop_requested: Callable[[], bool] = None):
    if stop_requested is None:
        return runner(command, capture_output=True, text=True)
    if stop_requested():
        raise ProcessingStopped("Stopped by user")

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        while process.poll() is None:
            if stop_requested():
                process.terminate()
                try:
                    stdout, stderr = process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                raise ProcessingStopped("Stopped by user") from None
            time.sleep(0.1)
        stdout, stderr = process.communicate()
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except Exception:
        if process.poll() is None:
            process.kill()
            process.communicate()
        raise


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


def _has_audio_stream(payload: Dict[str, Any]) -> bool:
    return any(stream.get("codec_type") == "audio" for stream in payload.get("streams", []))


def _audio_filter_for(
    ffmpeg_path: str,
    audio_quality: AudioQuality,
    filter_available: Callable[[str, str], bool] = None,
) -> str:
    checker = filter_available or _filter_available
    if audio_quality is AudioQuality.HIGH and checker(ffmpeg_path, "rubberband"):
        return "rubberband=tempo=0.5"
    return "atempo=0.5"


def _clean_audio_filter_for(
    ffmpeg_path: str,
    filter_available: Callable[[str, str], bool] = None,
) -> Optional[str]:
    checker = filter_available or _filter_available
    filters = []
    if checker(ffmpeg_path, "highpass"):
        filters.append("highpass=f=80")
    if checker(ffmpeg_path, "loudnorm"):
        filters.append("loudnorm=I=-18:TP=-2:LRA=11")
    return ",".join(filters) if filters else None


@lru_cache(maxsize=None)
def _filter_available(ffmpeg_path: str, filter_name: str) -> bool:
    try:
        completed = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    if completed.returncode != 0:
        return False
    return any(line.split()[1:2] == [filter_name] for line in completed.stdout.splitlines())


def _hwaccel_args(ffmpeg_hwaccel: FfmpegHwAccel, apple_silicon: bool) -> list:
    if ffmpeg_hwaccel is FfmpegHwAccel.NONE:
        return []
    if ffmpeg_hwaccel is FfmpegHwAccel.VIDEOTOOLBOX or (
        ffmpeg_hwaccel is FfmpegHwAccel.AUTO and apple_silicon
    ):
        return ["-hwaccel", "videotoolbox"]
    return []
