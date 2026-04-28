import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from .artifacts import LECTURE_ARTIFACT_NAME, load_json
from .ai.providers.gemini import build_manual_test_prompt, response_schema_for_prompt
from .writers import write_text_atomic


def export_gemini_test_package(
    lecture_path: Path,
    *,
    output_path: Optional[Path] = None,
    max_slides: int = 40,
) -> Path:
    lecture_dir = _resolve_lecture_dir(lecture_path)
    lecture_json_path = lecture_dir / LECTURE_ARTIFACT_NAME
    artifact = load_json(lecture_json_path) if lecture_json_path.exists() else _legacy_artifact(lecture_dir)

    selected_slides = _selected_slides(artifact, max_slides)
    prompt = build_manual_test_prompt(artifact, selected_slides=selected_slides)

    if output_path is None:
        output_path = lecture_dir / "gemini_test_package.zip"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    with tempfile.TemporaryDirectory(prefix="gemini-test-package-") as tmp:
        package_dir = Path(tmp)
        write_text_atomic(package_dir / "README.md", _readme(artifact, max_slides))
        write_text_atomic(package_dir / "prompt.md", prompt)
        write_text_atomic(
            package_dir / "expected-response-schema.json",
            json.dumps(response_schema_for_prompt(), indent=2) + "\n",
        )
        write_text_atomic(package_dir / "lecture.json", json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        for name in ["transcript.txt", "transcript.srt", "processing_log.txt"]:
            _copy_if_exists(lecture_dir / name, package_dir / name)

        slides_package_dir = package_dir / "slides"
        slides_package_dir.mkdir()
        for slide in selected_slides:
            source = lecture_dir / slide.get("relative_path", "")
            if source.exists():
                shutil.copy2(source, slides_package_dir / source.name)

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(package_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(package_dir))
    return output_path


def _resolve_lecture_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file() and path.name == LECTURE_ARTIFACT_NAME:
        return path.parent
    if path.is_dir():
        return path
    raise FileNotFoundError(f"Lecture path not found: {path}")


def _selected_slides(artifact: Dict, max_slides: int) -> List[Dict]:
    slides = list(artifact.get("slides") or [])
    if max_slides < 1:
        return []
    return slides[:max_slides]


def _legacy_artifact(lecture_dir: Path) -> Dict:
    transcript_text = _read_optional(lecture_dir / "transcript.txt")
    segments = _legacy_segments(lecture_dir / "transcript.srt")
    slides = _legacy_slides(lecture_dir / "slides", segments)
    return {
        "schema_version": "legacy-export",
        "lecture_id": lecture_dir.name,
        "source": {"filename": lecture_dir.name},
        "media": {"duration_seconds": segments[-1]["end"] if segments else 0.0},
        "transcript": {
            "text": transcript_text,
            "segments": segments,
        },
        "slides": slides,
        "processing": {},
        "enrichment": None,
    }


def _legacy_segments(srt_path: Path) -> List[Dict]:
    if not srt_path.exists():
        return []
    lines = srt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    segments = []
    index = 0
    cursor = 0
    while cursor < len(lines):
        if "-->" not in lines[cursor]:
            cursor += 1
            continue
        timing = lines[cursor]
        cursor += 1
        text_lines = []
        while cursor < len(lines) and lines[cursor].strip():
            text_lines.append(lines[cursor].strip())
            cursor += 1
        try:
            start_raw, end_raw = [part.strip() for part in timing.split("-->", 1)]
            start = parse_srt_timestamp(start_raw)
            end = parse_srt_timestamp(end_raw)
        except ValueError:
            continue
        segments.append(
            {
                "id": index,
                "start": start,
                "end": end,
                "text": " ".join(text_lines).strip(),
            }
        )
        index += 1
    return segments


def _legacy_slides(slides_dir: Path, segments: List[Dict]) -> List[Dict]:
    if not slides_dir.exists():
        return []
    slides = []
    for index, path in enumerate(sorted(slides_dir.glob("*.png")), start=1):
        timestamp = _slide_timestamp(path.name)
        slides.append(
            {
                "id": index,
                "filename": path.name,
                "relative_path": f"slides/{path.name}",
                "timestamp_seconds": timestamp,
                "linked_segment_ids": _linked_segment_ids(segments, timestamp),
            }
        )
    return slides


def _linked_segment_ids(segments: List[Dict], timestamp: float) -> List[int]:
    window_start = max(0.0, timestamp - 45.0)
    window_end = timestamp + 120.0
    linked = []
    for segment in segments:
        if float(segment.get("end", 0.0)) >= window_start and float(segment.get("start", 0.0)) <= window_end:
            linked.append(int(segment.get("id", 0)))
    return linked[:12]


def _slide_timestamp(filename: str) -> float:
    import re

    match = re.search(r"_(\d{2})-(\d{2})-(\d{2})(?:\D|$)", filename)
    if not match:
        return 0.0
    hours, minutes, seconds = (int(part) for part in match.groups())
    return float(hours * 3600 + minutes * 60 + seconds)


def _read_optional(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def parse_srt_timestamp(value: str) -> float:
    hours, minutes, rest = value.split(":", 2)
    seconds, millis = rest.split(",", 1)
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis[:3].ljust(3, "0")) / 1000.0
    )


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        shutil.copy2(source, destination)


def _readme(artifact: Dict, max_slides: int) -> str:
    lecture_id = artifact.get("lecture_id", "lecture")
    slide_count = len(artifact.get("slides") or [])
    included_count = min(slide_count, max(0, max_slides))
    return f"""# Gemini Manual Test Package

Lecture: `{lecture_id}`

## How to use this

1. Open Gemini.
2. Upload this zip file if the Gemini UI accepts zip uploads.
3. If zip uploads are unavailable, upload the files inside this package or paste `prompt.md`.
4. Ask Gemini to follow `prompt.md` exactly.
5. Save Gemini's JSON response for review before we lock the in-app integration.

## Package contents

- `prompt.md`: the exact prompt to test.
- `expected-response-schema.json`: the JSON shape we want back.
- `lecture.json`: the app's structured lecture artifact.
- `transcript.txt` and `transcript.srt`: transcript files when available.
- `processing_log.txt`: processing diagnostics.
- `slides/`: first {included_count} of {slide_count} extracted slides.

## What to judge

- Is the title useful?
- Is the summary accurate and not too long?
- Is the formatted transcript readable while still faithful to the original lecture?
- Do slide notes match the transcript and images?
- Are tags useful for search?
- Does the JSON parse without manual cleanup?
"""
