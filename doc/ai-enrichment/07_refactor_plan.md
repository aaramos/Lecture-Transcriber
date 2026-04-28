# 07 — Refactor Plan (Pre-Work)

These refactors land **before** AI enrichment is introduced. They're not optional. Skipping them creates technical debt that the AI work will inherit and pay interest on for the life of the product.

Each item lists: the problem today, the change, the test that proves it landed, and an estimate.

---

## R1. Extract The Artifact Writer

### Problem
Today, `pipeline.py::_process_file` writes transcripts via `writers.write_transcript`, slides via `SlideExtractor`, and logs via `writers.write_processing_log`. There's no single place that owns the lecture's output state. Adding `lecture.json` by sprinkling more write calls inside `_process_file` would compound the mess.

### Change
Create `lecture_processor/artifacts.py` with:

```python
class LectureArtifactWriter:
    def __init__(self, output_dir: Path): ...
    def write_initial(self, source_info: dict, media_info: MediaInfo) -> None: ...
    def update_transcript(self, transcript: TranscriptResult, engine: str, model: str) -> None: ...
    def update_slides(self, slides: list[dict]) -> None: ...
    def update_processing(self, started: datetime, finished: datetime, timings: dict, host: dict) -> None: ...
    def update_enrichment(self, enrichment: dict) -> None: ...
    def read(self) -> dict: ...
```

Each `update_*` reads the current artifact, modifies it, validates against the schema, and atomically writes it back. The pipeline calls these methods at stage boundaries instead of writing files directly.

The legacy `transcript.txt`, `transcript.srt`, `processing_log.txt` writers are kept and called by the artifact writer (`write_transcript` becomes a private function called from `update_transcript`). This preserves backward-compat on disk.

### Test
A new `test_artifacts.py`:
- Round-trip a synthetic lecture artifact through `write_initial`, all `update_*`, `read`. Assert schema validity at every step.
- Assert that re-running `update_transcript` doesn't lose previously-written `media` info (no clobbering).
- Assert the legacy txt/srt/log files are still produced.

### Estimate
2 days.

---

## R2. Persist Transcript JSON Alongside `.txt` / `.srt`

### Problem
Transcript segments only exist in memory during a run. AI enrichment needs `transcript.segments` from disk to operate on already-processed batches. Without this refactor, you'd have to re-transcribe to enrich.

### Change
The transcript becomes part of `lecture.json` (already specified in document 01). The standalone "save transcript JSON" file is **not** added — `lecture.json` is the source of truth. `transcript.txt` and `transcript.srt` become derived views.

Concretely: the existing `write_transcript` is moved out of `pipeline.py` into the artifact writer. It still produces the same .txt and .srt for backward compatibility, but the canonical source is `lecture.json`.

### Test
- Existing `test_writers.py` still passes (txt and srt unchanged).
- New test: load `lecture.json` from a fixture, derive .txt and .srt from it, byte-compare to known good output.

### Estimate
0.5 days (mostly piggybacks on R1).

---

## R3. Widen File Discovery Beyond `.mov`

### Problem
`pipeline.py::discover_mov_files` and `lib.rs::scan_folder` both hardcode `.mov`. The PRD calls for `.mp4` and broader support. Adding this after AI enrichment is in place means touching two languages and three places.

### Change
Replace the hardcoded `.mov` filter with a configurable extension list. Default: `.mov`, `.mp4`, `.m4v`. Behind the scenes:

- New constant `SUPPORTED_VIDEO_EXTENSIONS` in `lecture_processor/media.py`.
- Rename `discover_mov_files` → `discover_video_files`.
- Rename Rust `mov_count` / `mov_files` → `video_count` / `video_files` (and the `FolderScan` struct fields).
- Update the frontend's "{N} MOV files" copy to "{N} video files".

Why include this in the refactor phase: the current product won't process the user's `.mov` Kellogg lectures plus, say, an `.mp4` from a different course recording without this. Punting it to "after MVP" creates a UX bug that AI enrichment can't fix.

### Test
- Update `test_pipeline.py::test_discover_*` to assert all supported extensions are found, in alpha order.
- Update `test_cli.py` to verify the CLI accepts mixed-extension folders.
- Add a Rust test in `src-tauri/src/lib.rs` (currently no tests there — add one) for `scan_folder` with mixed extensions.

### Estimate
1 day. Keep the old name as a deprecated alias for one release.

---

## R4. Settle Slide Naming Policy

### Problem
The PRD asks for AI-generated descriptive slide filenames. The current naming is `slide_NNNN_HH-MM-SS.png`. Renaming the actual files on disk has cascading effects: HTML renderer needs to track them, transcript references break, the user's mental model of "the slides are right there in slides/" gets disrupted, and re-enriching becomes destructive.

### Change
Decide explicitly: **the original PNG filenames never change.** AI-generated descriptive filenames are stored in `lecture.json::slides[].analysis.descriptive_filename` and shown in the HTML. Users who want files renamed on disk can use a future "Export with descriptive names" action that copies/symlinks rather than renaming.

This is the simplest decision that satisfies the PRD without breaking idempotency. Document the choice in code comments and in the user-facing release notes.

### Test
- Existing slide tests pass unchanged.
- Add a test in document 09's HTML render tests that asserts descriptive names appear in HTML even when the PNG keeps its sequence-based filename.

### Estimate
0 days of code (it's a non-decision); 0.25 day to write the docs/notes.

---

## R5. Surface Apple Silicon Detection As A Single Source Of Truth

### Problem
`is_apple_silicon` lives in Rust (`lib.rs::is_apple_silicon`) and in Python (`cli.py::_detect_apple_silicon`). The Rust version sets a CLI flag that Python re-derives. This is fine today but the AI enrichment flow has nothing to do with Apple Silicon — except the HTML thumbnail generation uses Pillow's Accelerate path, which we do want optimized.

### Change
- Keep both detectors (they answer different questions in different processes).
- Document explicitly in `media.py` and `lib.rs` that the source of truth for "are we on Apple Silicon" is the runtime detection in each language.
- Stop passing `--apple-silicon` from Rust to Python. Python detects it itself. The CLI flag stays for testing override only.

This isn't strictly required for AI enrichment, but it's the right time to do it because the new HTML renderer will want the same detection logic and we shouldn't add a third call site.

### Test
- Update `test_cli.py` to verify `_detect_apple_silicon` works on linux (returns False).
- Manually verify on a real M1 Max that the existing whisper.cpp / VideoToolbox paths still light up.

### Estimate
0.5 day.

---

## R6. Move Logging To `processing_log.txt` Through One Function

### Problem
`pipeline.py::_process_file` builds a log line list inline and `writes_processing_log` at end. Adding enrichment events to that log (`Enrich   OK  152.0s  ...`) means appending to a log file that was atomically replaced by a different stage. Atomic-write-then-append is messy.

### Change
Introduce `lecture_processor/logging.py::ProcessingLog`:

```python
class ProcessingLog:
    def __init__(self, output_dir: Path): ...
    def append(self, line: str) -> None: ...
    def step(self, name: str, status: str, elapsed_seconds: float, detail: str = "") -> None: ...
    def flush(self) -> None: ...   # Atomic replace.
```

Inside, hold lines in memory and write them via `write_text_atomic` on `flush`. The pipeline calls `flush` at the end of `_process_file`. Enrichment calls `append`/`step` and `flush` independently — its log block goes into the existing file without racing the local pipeline (which has already finished by the time enrichment runs for that lecture).

### Test
- New `test_logging.py` covers append, step, flush, and re-flush (idempotent).
- Existing `test_pipeline.py` updated to assert log file contents are identical to before this refactor (byte-for-byte for completed paths).

### Estimate
0.5 day.

---

## R7. Add `lecture-processor render` And `lecture-processor enrich` CLI Subcommands (Stubs)

### Problem
The Rust bridge will spawn the Python CLI for enrichment and rendering as new subcommands. The CLI surface needs to exist before the Rust bridge can wire to it, otherwise the front-end and back-end teams can't work in parallel.

### Change
Add stub subcommands in `cli.py`:

```python
enrich = subparsers.add_parser("enrich", help="Run AI enrichment on a processed batch")
enrich.add_argument("output_dir", type=Path)
enrich.add_argument("--provider", default="anthropic")
enrich.add_argument("--model", default="")
enrich.add_argument("--include-slide-images", action="store_true")
enrich.add_argument("--concurrent", type=int, default=1)
enrich.add_argument("--read-key-from-stdin", action="store_true")
enrich.add_argument("--only-missing", action="store_true")
enrich.add_argument("--json-events", action="store_true", help=argparse.SUPPRESS)

render = subparsers.add_parser("render", help="Render HTML for a processed batch")
render.add_argument("output_dir", type=Path)
render.add_argument("--force", action="store_true")
render.add_argument("--json-events", action="store_true", help=argparse.SUPPRESS)
```

Both stubs print a "not implemented" event and exit non-zero. They unblock front-end work because the Rust bridge can be wired to them and tested with the stub.

### Test
- `test_cli.py` asserts both subcommands parse without error and produce the documented "not implemented" exit code.

### Estimate
0.5 day.

---

## Order And Total

The refactors should land in this order:

1. **R3** (file discovery) — 1 day. Independent. Land first to clear UX debt.
2. **R5** (Apple Silicon detection cleanup) — 0.5 day. Independent.
3. **R1 + R2** (artifact writer + transcript JSON) — 2.5 days combined. Deepest change.
4. **R6** (logging module) — 0.5 day. Depends on R1.
5. **R4** (slide naming decision) — 0.25 day. Documentation only.
6. **R7** (CLI stubs) — 0.5 day. Independent.

**Total: ~5.25 engineering days for one developer.** This is roughly one sprint. Treat it as Sprint 0 of the AI work — Phase 1 in the implementation plan starts after these land.

---

## What's Explicitly NOT Refactored

- **`gui.py` (the Tkinter app):** It's a parallel UI we don't ship. Don't add AI features to it. Eventually delete it; not in this scope.
- **The Tauri Rust command surface:** Add new commands additively, don't restructure.
- **Whisper transcription stack:** It works. Leave it alone.
- **FFmpeg/OpenCV slide extraction:** Same. Leave alone.
- **The `temp_cleanup` module:** Already atomic and well-tested.
- **The lock file mechanism (`OUTPUT_LOCK_FILE`):** Already correct. Enrichment uses the same lock.

The principle here is: only refactor what touches AI enrichment's path. The rest of the codebase has earned its spot.
