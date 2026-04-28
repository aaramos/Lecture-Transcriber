import json
from pathlib import Path
from typing import Iterable, Optional, Set

from .writers import write_json_atomic


CONTROL_FILE_NAME = ".lecture_processor_control.json"


class ProcessingControl:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path

    def skipped_files(self) -> Set[str]:
        return self._read_set("skip")

    def stopped_files(self) -> Set[str]:
        return self._read_set("stop")

    def should_skip(self, filename: str) -> bool:
        return filename in self.skipped_files()

    def should_stop(self, filename: str) -> bool:
        data = self._read()
        return filename in _string_set(data.get("stop")) or filename in _string_set(data.get("skip"))

    def _read_set(self, key: str) -> Set[str]:
        return _string_set(self._read().get(key))

    def _read(self) -> dict:
        if not self.path or not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}


def default_control_file(output_dir: Path) -> Path:
    return output_dir / CONTROL_FILE_NAME


def update_control_file(path: Path, *, skip: Iterable[str] = (), stop: Iterable[str] = ()) -> None:
    payload = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                payload = existing
        except (OSError, json.JSONDecodeError):
            payload = {}

    skip_set = _string_set(payload.get("skip"))
    stop_set = _string_set(payload.get("stop"))
    skip_set.update(name for name in skip if name)
    stop_set.update(name for name in stop if name)
    write_json_atomic(path, {"skip": sorted(skip_set), "stop": sorted(stop_set)})


def _string_set(value) -> Set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str) and item}
