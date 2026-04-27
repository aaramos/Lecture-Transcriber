import importlib
from pathlib import Path

from .config import SlideSensitivity
from .errors import DependencyMissingError, ProcessingError
from .timecode import format_timestamp_for_filename


class SlideExtractor:
    def __init__(self, sensitivity: SlideSensitivity = SlideSensitivity.MEDIUM) -> None:
        self.sensitivity = sensitivity

    def extract(self, media_path: Path, output_dir: Path, timestamp_scale: float = 1.0) -> int:
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
