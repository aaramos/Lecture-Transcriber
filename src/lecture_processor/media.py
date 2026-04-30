from collections import deque
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

from .config import AudioEnhancementMode, AudioQuality, FfmpegHwAccel, RecordingSpeed
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


def ensure_audio_enhancement_tools(deep_filter_path: str = "") -> None:
    resolve_deep_filter_tool(deep_filter_path)


def resolve_media_tool(command: str) -> str:
    return _require_command(command)


def resolve_deep_filter_tool(command: str = "") -> str:
    explicit = command.strip()
    if explicit:
        return _require_command(explicit)
    for candidate in _deep_filter_candidates():
        if candidate.exists() and candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    resolved = shutil.which("deep-filter")
    if resolved:
        return resolved
    raise DependencyMissingError(
        "DeepFilterNet is not installed. Install the ClearVoice tools, install deep-filter on PATH, "
        "or set DEEP_FILTER_PATH to the deep-filter binary."
    )


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


def _deep_filter_candidates() -> List[Path]:
    candidates: List[Path] = []
    for key in ("LECTURE_PROCESSOR_DEEP_FILTER_PATH", "DEEP_FILTER_PATH"):
        value = os.environ.get(key, "").strip()
        if value:
            candidates.append(Path(value).expanduser())

    home = Path.home()
    candidates.extend(
        [
            home / "Library/Application Support/ClearVoice/Tools/deep-filter/deep-filter",
            home / "Library/Application Support/Lecture Processor/Tools/deep-filter/deep-filter",
            home / "Library/Application Support/com.lecture.processor/runtime/tools/deep-filter/deep-filter",
            Path("/opt/homebrew/bin/deep-filter"),
            Path("/usr/local/bin/deep-filter"),
            Path("/tmp/deep-filter"),
        ]
    )
    return candidates

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
        has_video = bool(video_stream)
        has_audio = _has_audio_stream(payload)
        if not has_video and not has_audio:
            raise ProcessingError(f"No audio or video stream found in {path.name}")
        avg_frame_rate = _parse_fraction(video_stream.get("avg_frame_rate")) if video_stream else 0.0
        real_frame_rate = _parse_fraction(video_stream.get("r_frame_rate")) if video_stream else 0.0
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
            has_video=has_video,
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


@dataclass(frozen=True)
class _EnhancementProfile:
    highpass_frequency: int
    lowpass_frequency: int
    click_threshold: int
    click_burst: int
    clip_threshold: int
    noise_reduction: int
    noise_floor: int
    gain_smooth: int
    gate_threshold: float
    gate_ratio: float
    gate_range: float
    gate_attack: int
    gate_release: int
    speech_expansion: float
    speech_release: float


class DeepFilterAudioEnhancer:
    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        deep_filter_path: str = "",
        runner: Callable = subprocess.run,
    ) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.deep_filter_path = deep_filter_path
        self._runner = runner
        self.last_command_text = ""
        self._command_lock = threading.Lock()
        self._commands_by_destination: Dict[Path, List[list]] = {}

    def command_text_for(self, destination: Path) -> str:
        with self._command_lock:
            return _quote_commands(self._commands_by_destination.get(Path(destination), []))

    def enhance(
        self,
        source: Path,
        destination: Path,
        mode: AudioEnhancementMode,
        stop_requested: Callable[[], bool] = None,
    ) -> Path:
        if mode is AudioEnhancementMode.NONE:
            return source

        ffmpeg = _require_command(self.ffmpeg_path)
        deep_filter = resolve_deep_filter_tool(self.deep_filter_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_destination = destination.with_name(f".{destination.name}.ffmpeg.tmp")
        work_dir = Path(tempfile.mkdtemp(prefix="lecture-deepfilter-"))
        repaired_input = work_dir / "deepfilter_input.wav"
        deep_filter_output_dir = work_dir / "deepfilter_out"
        deep_filter_output_dir.mkdir(parents=True, exist_ok=True)

        preprocess_command = self._preprocess_command(ffmpeg, source, repaired_input, mode)
        deep_filter_command = self._deep_filter_command(deep_filter, repaired_input, deep_filter_output_dir)

        try:
            self._remember_commands(destination, [preprocess_command, deep_filter_command])
            completed = _run_interruptible(preprocess_command, self._runner, stop_requested)
            if completed.returncode != 0:
                raise ProcessingError(
                    f"audio enhancement pre-clean failed for {source.name}: "
                    f"{completed.stderr.strip() or completed.stdout.strip()}"
                )

            completed = _run_interruptible(deep_filter_command, self._runner, stop_requested)
            if completed.returncode != 0:
                raise ProcessingError(
                    f"DeepFilterNet failed for {source.name}: "
                    f"{completed.stderr.strip() or completed.stdout.strip()}"
                )

            enhanced_wav = self._locate_deep_filter_output(
                expected_filename=repaired_input.name,
                output_dir=deep_filter_output_dir,
            )
            if not enhanced_wav:
                raise ProcessingError("DeepFilterNet finished but did not create an enhanced WAV.")

            postprocess_command = self._postprocess_command(ffmpeg, enhanced_wav, temp_destination, mode)
            self._remember_commands(destination, [preprocess_command, deep_filter_command, postprocess_command])
            completed = _run_interruptible(
                postprocess_command,
                self._runner,
                stop_requested,
                progress_path=temp_destination,
                stall_timeout_seconds=_FFMPEG_STALL_TIMEOUT_SECONDS,
            )
            if completed.returncode != 0:
                raise ProcessingError(
                    f"audio enhancement export failed for {source.name}: "
                    f"{completed.stderr.strip() or completed.stdout.strip()}"
                )
            temp_destination.replace(destination)
        finally:
            remove_temp_path(temp_destination)
            remove_temp_path(work_dir)
        return destination

    def _preprocess_command(self, ffmpeg: str, source: Path, destination: Path, mode: AudioEnhancementMode) -> list:
        return [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-af",
            _deep_filter_preprocess_filter_for(mode),
            "-ac",
            "1",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]

    def _deep_filter_command(self, deep_filter: str, source: Path, output_dir: Path) -> list:
        return [
            deep_filter,
            "--compensate-delay",
            "-o",
            str(output_dir),
            str(source),
        ]

    def _postprocess_command(self, ffmpeg: str, source: Path, destination: Path, mode: AudioEnhancementMode) -> list:
        return [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-af",
            _deep_filter_postprocess_filter_for(mode),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(destination),
        ]

    def _locate_deep_filter_output(self, expected_filename: str, output_dir: Path) -> Optional[Path]:
        expected = output_dir / expected_filename
        if expected.exists():
            return expected
        try:
            return sorted(path for path in output_dir.iterdir() if path.suffix.lower() == ".wav")[0]
        except (IndexError, FileNotFoundError):
            return None

    def _remember_commands(self, destination: Path, commands: List[list]) -> None:
        with self._command_lock:
            self.last_command_text = _quote_commands(commands)
            self._commands_by_destination[Path(destination)] = [list(command) for command in commands]


def _quote_command(command: list) -> str:
    return shlex.join(str(item) for item in command) if command else ""


def _quote_commands(commands: List[list]) -> str:
    return " && ".join(_quote_command(command) for command in commands if command)


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


def _first_video_stream(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for stream in payload.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    return None


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


def _deep_filter_preprocess_filter_for(mode: AudioEnhancementMode) -> str:
    profile = _deep_filter_profile_for(mode)
    return ",".join(
        [
            (
                "adeclick=window=20:overlap=75:arorder=2:"
                f"threshold={profile.click_threshold}:burst={profile.click_burst}:method=save"
            ),
            (
                "adeclip=window=55:overlap=75:arorder=8:"
                f"threshold={profile.clip_threshold}:hsize=1200:method=save"
            ),
            f"highpass=f={profile.highpass_frequency}",
            f"lowpass=f={profile.lowpass_frequency}",
            (
                f"afftdn=nr={profile.noise_reduction}:nf={profile.noise_floor}:"
                f"tn=1:gs={profile.gain_smooth}"
            ),
            (
                f"agate=threshold={profile.gate_threshold}:ratio={profile.gate_ratio}:"
                f"range={profile.gate_range}:attack={profile.gate_attack}:"
                f"release={profile.gate_release}:detection=rms"
            ),
            f"speechnorm=e={profile.speech_expansion}:r={profile.speech_release}:l=1",
        ]
    )


def _deep_filter_postprocess_filter_for(mode: AudioEnhancementMode) -> str:
    lowpass = 7200 if mode is AudioEnhancementMode.STRONG else 7800
    return ",".join(
        [
            "highpass=f=80",
            f"lowpass=f={lowpass}",
            "speechnorm=e=4.0:r=0.0001:l=1",
        ]
    )


def _deep_filter_profile_for(mode: AudioEnhancementMode) -> _EnhancementProfile:
    if mode is AudioEnhancementMode.STRONG:
        return _EnhancementProfile(
            highpass_frequency=110,
            lowpass_frequency=6800,
            click_threshold=3,
            click_burst=4,
            clip_threshold=8,
            noise_reduction=22,
            noise_floor=-58,
            gain_smooth=10,
            gate_threshold=0.035,
            gate_ratio=3.0,
            gate_range=0.30,
            gate_attack=30,
            gate_release=420,
            speech_expansion=10.0,
            speech_release=0.00005,
        )
    return _EnhancementProfile(
        highpass_frequency=90,
        lowpass_frequency=7600,
        click_threshold=6,
        click_burst=2,
        clip_threshold=12,
        noise_reduction=14,
        noise_floor=-50,
        gain_smooth=6,
        gate_threshold=0.022,
        gate_ratio=1.6,
        gate_range=0.65,
        gate_attack=20,
        gate_release=240,
        speech_expansion=6.0,
        speech_release=0.00008,
    )


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
