import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

OUTPUT_LOCK_FILE = ".lecture_processor.lock"
NORMALIZED_WORK_FILE = ".normalized_work.mp4"
TRANSCRIPTION_AUDIO_FILE = ".transcription_audio.wav"
FFMPEG_TEMP_SUFFIX = ".ffmpeg.tmp"
SLIDE_TEMP_PREFIX = "lecture-slides-"
ATOMIC_TEMP_FILES = {
    ".batch.json.tmp",
    ".batch_error.txt.tmp",
    ".batch_summary.txt.tmp",
    ".index.html.tmp",
    ".lecture.json.tmp",
    ".processing_log.txt.tmp",
    ".transcript.srt.tmp",
    ".transcript.txt.tmp",
    "..lecture_processor_control.json.tmp",
}


def cleanup_processor_temp_files(output_dir: Optional[Path] = None) -> int:
    removed = cleanup_slide_temp_dirs()
    if output_dir:
        removed += cleanup_output_temp_files(output_dir)
    return removed


def cleanup_output_temp_files(output_dir: Path) -> int:
    if not output_dir.exists():
        return 0

    removed = 0
    for path in _iter_output_temp_files(output_dir):
        removed += remove_temp_path(path)

    lock_path = output_dir / OUTPUT_LOCK_FILE
    if lock_path.exists() and not _lock_owner_is_running(lock_path):
        removed += remove_temp_path(lock_path)

    return removed


def cleanup_slide_temp_dirs(temp_root: Optional[Path] = None) -> int:
    root = temp_root or Path(tempfile.gettempdir())
    if not root.exists():
        return 0

    removed = 0
    for child in root.iterdir():
        if not child.name.startswith(SLIDE_TEMP_PREFIX) or not child.is_dir():
            continue
        pid = _pid_from_slide_temp_name(child.name)
        if pid and _pid_is_running(pid):
            continue
        removed += remove_temp_path(child)
    return removed


def remove_temp_path(path: Path) -> int:
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except FileNotFoundError:
        return 0
    except OSError:
        return 0
    return 1


def _iter_output_temp_files(output_dir: Path):
    for root, _dirnames, filenames in os.walk(output_dir):
        root_path = Path(root)
        for filename in filenames:
            if (
                filename == NORMALIZED_WORK_FILE
                or filename == TRANSCRIPTION_AUDIO_FILE
                or filename in ATOMIC_TEMP_FILES
                or filename.endswith(FFMPEG_TEMP_SUFFIX)
            ):
                yield root_path / filename


def _lock_owner_is_running(lock_path: Path) -> bool:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    pid = payload.get("pid")
    return isinstance(pid, int) and pid > 0 and _pid_is_running(pid)


def _pid_from_slide_temp_name(name: str) -> Optional[int]:
    remainder = name.removeprefix(SLIDE_TEMP_PREFIX)
    token = remainder.split("-", 1)[0]
    return int(token) if token.isdigit() else None


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
