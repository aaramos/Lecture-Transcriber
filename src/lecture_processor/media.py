from collections import deque
import json
import shlex
import shutil
import subprocess
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional

from .config import AudioQuality, FfmpegHwAccel, RecordingSpeed
from .errors import DependencyMissingError, ProcessingError, ProcessingStopped
from .models import MediaInfo
from .temp_cleanup import remove_temp_path

_SUBPROCESS_OUTPUT_LIMIT = 65536
_SUBPROCESS_POLL_INTERVAL_SECONDS = 0.2
_FFMPEG_STALL_TIMEOUT_SECONDS = 600.0


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
    system = _known_system_tool_candidate(command)
    if system:
        return str(system)
    resolved = shutil.which(command)
    if not resolved:
        raise DependencyMissingError(f"Required command not found on PATH: {command}")
    return resolved


def _known_system_tool_candidate(command: str) -> Optional[Path]:
    if command not in {"ffmpeg", "ffprobe"}:
        return None

    for root in (Path("/opt/homebrew/bin"), Path("/usr/local/bin")):
        candidate = root / command
        if candidate.exists():
            return candidate
    return None


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
                progress_path=temp_destination,
                stall_timeout_seconds=_FFMPEG_STALL_TIMEOUT_SECONDS,
            )
            if completed.returncode != 0 and hwaccel_args and self.ffmpeg_hwaccel is FfmpegHwAccel.AUTO:
                remove_temp_path(temp_destination)
                completed = _run_interruptible(
                    build_command(temp_destination, []),
                    self._runner,
                    stop_requested,
                    progress_path=temp_destination,
                    stall_timeout_seconds=_FFMPEG_STALL_TIMEOUT_SECONDS,
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
        self.last_command = []
        self._command_lock = threading.Lock()
        self._commands_by_destination: Dict[Path, list] = {}

    @property
    def last_command_text(self) -> str:
        with self._command_lock:
            return _quote_command(self.last_command)

    def command_text_for(self, destination: Path) -> str:
        with self._command_lock:
            return _quote_command(self._commands_by_destination.get(Path(destination), []))

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
            self._remember_command(destination, command)
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

    def extract_for_transcription(
        self,
        source: Path,
        destination: Path,
        media_info: MediaInfo,
        recording_speed: RecordingSpeed,
        audio_quality: AudioQuality,
        stop_requested: Callable[[], bool] = None,
    ) -> Path:
        if recording_speed is RecordingSpeed.NORMAL:
            return self.extract(source, destination, media_info, stop_requested=stop_requested)
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
            "-af",
            _transcription_audio_filter_for(
                ffmpeg,
                recording_speed,
                audio_quality,
                self._filter_available,
            ),
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(temp_destination),
        ]

        try:
            self._remember_command(destination, command)
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

    def _remember_command(self, destination: Path, command: list) -> None:
        with self._command_lock:
            self.last_command = list(command)
            self._commands_by_destination[Path(destination)] = list(command)


def _quote_command(command: list) -> str:
    return shlex.join(str(item) for item in command) if command else ""


def _run_interruptible(
    command: list,
    runner: Callable,
    stop_requested: Callable[[], bool] = None,
    progress_path: Optional[Path] = None,
    stall_timeout_seconds: Optional[float] = None,
    poll_interval_seconds: float = _SUBPROCESS_POLL_INTERVAL_SECONDS,
):
    if stop_requested is None:
        return runner(command, capture_output=True, text=True)
    if stop_requested():
        raise ProcessingStopped("Stopped by user")

    stdout_buffer = _BoundedOutputBuffer()
    stderr_buffer = _BoundedOutputBuffer()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout_thread = threading.Thread(
        target=_drain_stream,
        args=(process.stdout, stdout_buffer),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_drain_stream,
        args=(process.stderr, stderr_buffer),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    last_progress = _file_progress(progress_path)
    last_progress_at = time.monotonic()
    try:
        while process.poll() is None:
            if stop_requested():
                _terminate_process(process)
                _join_reader_threads(stdout_thread, stderr_thread)
                raise ProcessingStopped("Stopped by user") from None
            if progress_path is not None and stall_timeout_seconds:
                current_progress = _file_progress(progress_path)
                if current_progress != last_progress:
                    last_progress = current_progress
                    last_progress_at = time.monotonic()
                elif time.monotonic() - last_progress_at >= stall_timeout_seconds:
                    _terminate_process(process)
                    _join_reader_threads(stdout_thread, stderr_thread)
                    stderr = stderr_buffer.text()
                    stall_message = (
                        f"Process stalled for {stall_timeout_seconds:.0f}s without output file progress: "
                        f"{progress_path}"
                    )
                    stderr = f"{stderr.rstrip()}\n{stall_message}".strip()
                    return subprocess.CompletedProcess(
                        command,
                        process.returncode if process.returncode is not None else 1,
                        stdout_buffer.text(),
                        stderr,
                    )
            time.sleep(poll_interval_seconds)
        _join_reader_threads(stdout_thread, stderr_thread)
        return subprocess.CompletedProcess(
            command,
            process.returncode,
            stdout_buffer.text(),
            stderr_buffer.text(),
        )
    except Exception:
        if process.poll() is None:
            _terminate_process(process, force=True)
        _join_reader_threads(stdout_thread, stderr_thread)
        raise


class _BoundedOutputBuffer:
    def __init__(self, limit: int = _SUBPROCESS_OUTPUT_LIMIT) -> None:
        self.limit = limit
        self._chunks: Deque[str] = deque()
        self._size = 0
        self._lock = threading.Lock()

    def append(self, chunk: str) -> None:
        if not chunk:
            return
        with self._lock:
            self._chunks.append(chunk)
            self._size += len(chunk)
            while self._size > self.limit and self._chunks:
                excess = self._size - self.limit
                first = self._chunks[0]
                if excess >= len(first):
                    self._chunks.popleft()
                    self._size -= len(first)
                else:
                    self._chunks[0] = first[excess:]
                    self._size -= excess

    def text(self) -> str:
        with self._lock:
            return "".join(self._chunks)


def _drain_stream(stream, buffer: _BoundedOutputBuffer) -> None:
    if stream is None:
        return
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                return
            buffer.append(chunk)
    finally:
        stream.close()


def _join_reader_threads(*threads: threading.Thread) -> None:
    for thread in threads:
        thread.join(timeout=1)


def _terminate_process(process: subprocess.Popen, force: bool = False) -> None:
    if process.poll() is not None:
        return
    if force:
        process.kill()
        process.wait(timeout=2)
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def _file_progress(path: Optional[Path]) -> Optional[tuple]:
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns


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


def _transcription_audio_filter_for(
    ffmpeg_path: str,
    recording_speed: RecordingSpeed,
    audio_quality: AudioQuality,
    filter_available: Callable[[str, str], bool] = None,
) -> str:
    filters = []
    if recording_speed is RecordingSpeed.DOUBLE:
        # Speed correction has to run before highpass/loudnorm so those filters see the final 1x audio.
        filters.append(_audio_filter_for(ffmpeg_path, audio_quality, filter_available))
    clean_filter = _clean_audio_filter_for(ffmpeg_path, filter_available)
    if clean_filter:
        filters.extend(clean_filter.split(","))
    return ",".join(filters)


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
