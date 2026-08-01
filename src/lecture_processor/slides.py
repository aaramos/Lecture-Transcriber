import importlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .config import FfmpegHwAccel, SlideBackend, SlideSensitivity
from .errors import DependencyMissingError, ProcessingError, ProcessingStopped
from .media import _hwaccel_args, _require_command, _run_interruptible
from .slide_classifier import LmStudioSlideClassifier, SlideClassifierResult
from .timecode import format_timestamp_for_filename
from .writers import write_json_atomic


@dataclass(frozen=True)
class FrameCandidate:
    path: Path
    timestamp: float


@dataclass(frozen=True)
class ClassifiedFrame:
    path: Path
    timestamp: float
    verdict: str
    description: str = ""
    title: Optional[str] = None
    build_stage: Optional[str] = None
    layout: Optional[str] = None
    raw_response: str = ""


@dataclass(frozen=True)
class SlideExtractionResult:
    saved_count: int
    candidate_count: int
    warnings: List[str] = field(default_factory=list)


class SlideExtractor:
    def __init__(
        self,
        sensitivity: SlideSensitivity = SlideSensitivity.MEDIUM,
        backend: SlideBackend = SlideBackend.AUTO,
        ffmpeg_path: str = "ffmpeg",
        ffmpeg_hwaccel: FfmpegHwAccel = FfmpegHwAccel.AUTO,
        apple_silicon: bool = False,
        classifier: Optional[LmStudioSlideClassifier] = None,
    ) -> None:
        self.sensitivity = sensitivity
        self.backend = backend
        self.ffmpeg_path = ffmpeg_path
        self.ffmpeg_hwaccel = ffmpeg_hwaccel
        self.apple_silicon = apple_silicon
        self.classifier = classifier
        self.last_warnings: List[str] = []
        self.last_candidate_count = 0
        self.last_slide_count = 0

    def extract(
        self,
        media_path: Path,
        output_dir: Path,
        timestamp_scale: float = 1.0,
        stop_requested: Callable[[], bool] = None,
    ) -> int:
        return self.extract_with_result(media_path, output_dir, timestamp_scale, stop_requested).saved_count

    def extract_with_result(
        self,
        media_path: Path,
        output_dir: Path,
        timestamp_scale: float = 1.0,
        stop_requested: Callable[[], bool] = None,
    ) -> SlideExtractionResult:
        self.last_warnings = []
        self.last_candidate_count = 0
        self.last_slide_count = 0
        warnings: List[str] = []
        if self.backend is SlideBackend.OPENCV:
            result = self._extract_opencv(media_path, output_dir, timestamp_scale, stop_requested, warnings)
            self._remember_result(result)
            return result

        try:
            result = self._extract_ffmpeg(media_path, output_dir, timestamp_scale, stop_requested, warnings)
        except (DependencyMissingError, ProcessingError):
            if self.backend is SlideBackend.FFMPEG:
                raise
            result = self._extract_opencv(media_path, output_dir, timestamp_scale, stop_requested, warnings)
        self._remember_result(result)
        return result

    def _remember_result(self, result: SlideExtractionResult) -> None:
        self.last_warnings = list(result.warnings)
        self.last_candidate_count = result.candidate_count
        self.last_slide_count = result.saved_count

    def _extract_ffmpeg(
        self,
        media_path: Path,
        output_dir: Path,
        timestamp_scale: float,
        stop_requested: Callable[[], bool] = None,
        warnings: Optional[List[str]] = None,
    ) -> SlideExtractionResult:
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
            candidate_dir = tmp_path / "candidates" if self.classifier else output_dir
            candidate_dir.mkdir(parents=True, exist_ok=True)
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
            candidates: List[FrameCandidate] = []
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
                            candidate_path = candidate_dir / filename
                            shutil.copyfile(frame_path, candidate_path)
                            candidates.append(FrameCandidate(path=candidate_path, timestamp=timestamp))
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

            if self.classifier:
                selected_count = self._classify_and_write_best_slides(
                    candidates,
                    output_dir,
                    stop_requested,
                    warnings=warnings,
                )
                return SlideExtractionResult(
                    saved_count=selected_count,
                    candidate_count=len(candidates),
                    warnings=list(warnings or []),
                )

        return SlideExtractionResult(saved_count=saved, candidate_count=saved, warnings=list(warnings or []))

    def _extract_opencv(
        self,
        media_path: Path,
        output_dir: Path,
        timestamp_scale: float,
        stop_requested: Callable[[], bool] = None,
        warnings: Optional[List[str]] = None,
    ) -> SlideExtractionResult:
        try:
            cv2 = importlib.import_module("cv2")
        except ImportError as exc:
            raise DependencyMissingError(
                "opencv-python is not installed. Install with: "
                "python3 -m pip install -e '.[slides]'"
            ) from exc

        output_dir.mkdir(parents=True, exist_ok=True)
        temp_candidate_dir = None
        candidate_dir = output_dir
        if self.classifier:
            temp_candidate_dir = tempfile.TemporaryDirectory(prefix=f"lecture-slide-candidates-{os.getpid()}-")
            candidate_dir = Path(temp_candidate_dir.name)
        capture = cv2.VideoCapture(str(media_path))
        if not capture.isOpened():
            if temp_candidate_dir is not None:
                temp_candidate_dir.cleanup()
            raise ProcessingError(f"Could not open video for slide extraction: {media_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0 or fps > 240:
            fps = 30.0
        sample_stride = max(1, int(round(fps * 2.0)))
        threshold = _threshold_for(self.sensitivity)
        previous_frame = None
        candidates: List[FrameCandidate] = []
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
                    candidate_path = candidate_dir / filename
                    if not cv2.imwrite(str(candidate_path), frame):
                        raise ProcessingError(f"Failed to save slide image: {candidate_path}")
                    candidates.append(FrameCandidate(path=candidate_path, timestamp=timestamp))
                    previous_frame = frame

                frame_index += 1
        finally:
            capture.release()

        if self.classifier:
            try:
                selected_count = self._classify_and_write_best_slides(
                    candidates,
                    output_dir,
                    stop_requested,
                    warnings=warnings,
                )
                return SlideExtractionResult(
                    saved_count=selected_count,
                    candidate_count=len(candidates),
                    warnings=list(warnings or []),
                )
            finally:
                if temp_candidate_dir is not None:
                    temp_candidate_dir.cleanup()

        return SlideExtractionResult(saved_count=saved, candidate_count=saved, warnings=list(warnings or []))

    def _classify_and_write_best_slides(
        self,
        candidates: List[FrameCandidate],
        output_dir: Path,
        stop_requested: Callable[[], bool] = None,
        warnings: Optional[List[str]] = None,
    ) -> int:
        if not self.classifier or not candidates:
            return 0
        warning_list = warnings if warnings is not None else self.last_warnings
        classified = []
        for index, candidate in enumerate(candidates, start=1):
            if stop_requested and stop_requested():
                raise ProcessingStopped("Stopped by user")
            try:
                result = self.classifier.classify(candidate.path)
            except Exception as exc:
                warning_list.append(
                    f"Smart slide classifier failed for {candidate.path.name}; kept frame as a slide candidate: {exc}"
                )
                result = SlideClassifierResult(
                    description=f"Classifier failed for {candidate.path.name}.",
                    verdict="SLIDE",
                    title=f"Slide candidate {index}",
                    build_stage="partial",
                    layout=None,
                )
            classified.append(
                ClassifiedFrame(
                    path=candidate.path,
                    timestamp=candidate.timestamp,
                    verdict=result.verdict,
                    description=result.description,
                    title=result.title,
                    build_stage=result.build_stage,
                    layout=result.layout,
                    raw_response=result.raw_response,
                )
            )

        selected = select_best_unique_slides(classified)
        for existing in output_dir.glob("*.png"):
            existing.unlink()
        for existing in output_dir.glob("*.json"):
            existing.unlink()
        for index, frame in enumerate(selected, start=1):
            filename = f"slide_{index:04d}_{format_timestamp_for_filename(frame.timestamp)}.png"
            target = output_dir / filename
            shutil.copyfile(frame.path, target)
            write_json_atomic(
                target.with_suffix(".json"),
                {
                    "title": frame.title,
                    "build_stage": frame.build_stage,
                    "layout": frame.layout,
                    "description": frame.description,
                    "timestamp": frame.timestamp,
                    "source_filename": frame.path.name,
                    "verdict": frame.verdict,
                },
            )
        warning_list.append(
            f"Smart slide extraction kept {len(selected)} unique slide(s) from {len(candidates)} changed frame(s)."
        )
        return len(selected)


def _threshold_for(sensitivity: SlideSensitivity) -> float:
    if sensitivity is SlideSensitivity.HIGH:
        return 8.0
    if sensitivity is SlideSensitivity.LOW:
        return 28.0
    return 16.0


def select_best_unique_slides(classified_frames: List[ClassifiedFrame]) -> List[ClassifiedFrame]:
    slides = [frame for frame in classified_frames if frame.verdict == "SLIDE"]
    if not slides:
        return []
    clusters = _title_clusters(slides)
    clusters = _merge_nearby_clusters(clusters, classified_frames)
    best_frames = [min(cluster, key=_rank_frame) for cluster in clusters]
    return sorted(best_frames, key=lambda frame: frame.timestamp)


def _title_clusters(slides: List[ClassifiedFrame]) -> List[List[ClassifiedFrame]]:
    clusters: List[List[ClassifiedFrame]] = []
    titles: List[str] = []
    for frame in sorted(slides, key=lambda item: item.timestamp):
        title = _normalized_title(frame.title or frame.description or frame.path.stem)
        matched_index = None
        for index, cluster_title in enumerate(titles):
            if _same_title_group(title, cluster_title):
                matched_index = index
                break
        if matched_index is None:
            clusters.append([frame])
            titles.append(title)
        else:
            clusters[matched_index].append(frame)
    return clusters


def _merge_nearby_clusters(
    clusters: List[List[ClassifiedFrame]],
    classified_frames: List[ClassifiedFrame],
) -> List[List[ClassifiedFrame]]:
    ordered = sorted(clusters, key=lambda cluster: min(frame.timestamp for frame in cluster))
    merged: List[List[ClassifiedFrame]] = []
    for cluster in ordered:
        if not merged:
            merged.append(cluster)
            continue
        previous = merged[-1]
        previous_end = max(frame.timestamp for frame in previous)
        cluster_start = min(frame.timestamp for frame in cluster)
        has_not_slide_between = any(
            frame.verdict != "SLIDE" and previous_end < frame.timestamp < cluster_start
            for frame in classified_frames
        )
        if cluster_start - previous_end <= 10.0 and not has_not_slide_between:
            previous.extend(cluster)
        else:
            merged.append(cluster)
    return merged


STAGE_RANK = {"full": 0, "partial": 1, "transitioning": 2}
LAYOUT_RANK = {"full-screen": 0, "split-left": 1, "split-right": 2}


def _rank_frame(frame: ClassifiedFrame) -> tuple:
    return (
        STAGE_RANK.get(str(frame.build_stage or "").strip().lower(), 9),
        LAYOUT_RANK.get(str(frame.layout or "").strip().lower(), 9),
        -float(frame.timestamp or 0.0),
    )


def _same_title_group(title_a: str, title_b: str, threshold: float = 0.85) -> bool:
    if not title_a or not title_b:
        return False
    distance = _levenshtein_distance(title_a, title_b)
    longest = max(len(title_a), len(title_b), 1)
    return (1.0 - (distance / longest)) >= threshold


def _normalized_title(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insert = current[right_index - 1] + 1
            delete = previous[right_index] + 1
            replace = previous[right_index - 1] + (0 if left_char == right_char else 1)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]
