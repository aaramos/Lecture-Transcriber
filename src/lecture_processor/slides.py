import importlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from .config import FfmpegHwAccel, SlideBackend, SlideSensitivity
from .errors import DependencyMissingError, ProcessingError, ProcessingStopped
from .media import _hwaccel_args, _require_command, _run_interruptible
from .timecode import format_timestamp_for_filename


class SlideExtractor:
    def __init__(
        self,
        sensitivity: SlideSensitivity = SlideSensitivity.MEDIUM,
        backend: SlideBackend = SlideBackend.AUTO,
        ffmpeg_path: str = "ffmpeg",
        ffmpeg_hwaccel: FfmpegHwAccel = FfmpegHwAccel.AUTO,
        apple_silicon: bool = False,
    ) -> None:
        self.sensitivity = sensitivity
        self.backend = backend
        self.ffmpeg_path = ffmpeg_path
        self.ffmpeg_hwaccel = ffmpeg_hwaccel
        self.apple_silicon = apple_silicon

    def extract(
        self,
        media_path: Path,
        output_dir: Path,
        timestamp_scale: float = 1.0,
        stop_requested: Callable[[], bool] = None,
    ) -> int:
        if self.backend is SlideBackend.OPENCV:
            return self._extract_opencv(media_path, output_dir, timestamp_scale, stop_requested)

        try:
            return self._extract_ffmpeg(media_path, output_dir, timestamp_scale, stop_requested)
        except (DependencyMissingError, ProcessingError):
            if self.backend is SlideBackend.FFMPEG:
                raise
            return self._extract_opencv(media_path, output_dir, timestamp_scale, stop_requested)

    def _extract_ffmpeg(
        self,
        media_path: Path,
        output_dir: Path,
        timestamp_scale: float,
        stop_requested: Callable[[], bool] = None,
    ) -> int:
        try:
            image_module = importlib.import_module("PIL.Image")
            image_chops = importlib.import_module("PIL.ImageChops")
            image_stat = importlib.import_module("PIL.ImageStat")
        except ImportError as exc:
            raise DependencyMissingError(
                "Pillow is not installed. Install with: python3 -m pip install -e '.[slides]'"
            ) from exc

        ffmpeg = _require_command(self.ffmpeg_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        threshold = _threshold_for(self.sensitivity)

        with tempfile.TemporaryDirectory(prefix=f"lecture-slides-{os.getpid()}-") as tmp:
            tmp_path = Path(tmp)
            frame_pattern = tmp_path / "frame_%06d.png"
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                *_hwaccel_args(self.ffmpeg_hwaccel, self.apple_silicon),
                "-i",
                str(media_path),
                "-vf",
                "fps=1/2",
                str(frame_pattern),
            ]
            completed = _run_interruptible(command, subprocess.run, stop_requested)
            if completed.returncode != 0:
                raise ProcessingError(
                    f"ffmpeg slide extraction failed for {media_path.name}: "
                    f"{completed.stderr.strip() or completed.stdout.strip()}"
                )

            previous_frame = None
            saved = 0
            try:
                for sample_index, frame_path in enumerate(sorted(tmp_path.glob("frame_*.png"))):
                    if stop_requested and stop_requested():
                        raise ProcessingStopped("Stopped by user")
                    frame = None
                    try:
                        frame = image_module.open(frame_path).convert("RGB")
                        should_save = previous_frame is None
                        if previous_frame is not None:
                            diff = image_chops.difference(previous_frame, frame)
                            means = image_stat.Stat(diff).mean
                            should_save = (sum(means) / len(means)) >= threshold

                        if should_save:
                            timestamp = (sample_index * 2.0) * timestamp_scale
                            saved += 1
                            filename = f"slide_{saved:04d}_{format_timestamp_for_filename(timestamp)}.png"
                            shutil.copyfile(frame_path, output_dir / filename)
                            if previous_frame is not None:
                                previous_frame.close()
                            previous_frame = frame.copy()
                    finally:
                        if frame is not None:
                            frame.close()
                        try:
                            frame_path.unlink()
                        except FileNotFoundError:
                            pass
            finally:
                if previous_frame is not None:
                    previous_frame.close()

        return saved

    def _extract_opencv(
        self,
        media_path: Path,
        output_dir: Path,
        timestamp_scale: float,
        stop_requested: Callable[[], bool] = None,
    ) -> int:
        try:
            cv2 = importlib.import_module("cv2")
        except ImportError as exc:
            raise DependencyMissingError(
                "opencv-python is not installed. Install with: "
                "python3 -m pip install -e '.[slides]'"
            ) from exc

        output_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(media_path))
        if not capture.isOpened():
            raise ProcessingError(f"Could not open video for slide extraction: {media_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        sample_stride = max(1, int(round(fps * 2.0)))
        threshold = _threshold_for(self.sensitivity)
        previous_frame = None
        saved = 0
        frame_index = 0

        try:
            while True:
                if stop_requested and stop_requested():
                    raise ProcessingStopped("Stopped by user")
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % sample_stride != 0:
                    frame_index += 1
                    continue

                should_save = previous_frame is None
                if previous_frame is not None:
                    diff = cv2.absdiff(previous_frame, frame)
                    should_save = float(diff.mean()) >= threshold

                if should_save:
                    timestamp = (frame_index / fps) * timestamp_scale
                    saved += 1
                    filename = f"slide_{saved:04d}_{format_timestamp_for_filename(timestamp)}.png"
                    cv2.imwrite(str(output_dir / filename), frame)
                    previous_frame = frame

                frame_index += 1
        finally:
            capture.release()

        return saved


def _threshold_for(sensitivity: SlideSensitivity) -> float:
    if sensitivity is SlideSensitivity.HIGH:
        return 8.0
    if sensitivity is SlideSensitivity.LOW:
        return 28.0
    return 16.0
