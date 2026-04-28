# 01 — Data Model & Artifact Schema

This is the most important document in the set. The `lecture.json` artifact is the contract between every stage. Every other document depends on this one.

---

## 1. Why An Explicit Artifact Layer

Today, output is a folder of files (transcript.txt, slides/*.png, processing_log.txt). Stages communicate through Python objects in memory and through filenames. That works for a single linear pipeline, but it doesn't survive:

- Re-running AI enrichment on a previously-processed batch.
- Skipping or retrying individual stages.
- Validating outputs against a schema.
- Generating HTML deterministically from a known-good input.

The fix is one canonical JSON document per lecture, written by the local pipeline and read/augmented by every later stage. The HTML renderer never reads a `.txt` file or scans a folder. It reads `lecture.json`.

## 2. Artifact Files

Two files are added to the existing output structure:

```
OutputFolder/
├── batch.json                    ← NEW. Top-level batch artifact.
├── batch_summary.txt             ← Existing. Kept for backward compat.
└── Lecture_Name/
    ├── lecture.json              ← NEW. Per-lecture canonical artifact.
    ├── Lecture_Name.mp4          ← Existing.
    ├── transcript.txt            ← Existing. Generated FROM lecture.json.
    ├── transcript.srt            ← Existing. Generated FROM lecture.json.
    ├── slides/
    │   └── slide_0001_00-05-30.png
    ├── ai/                       ← NEW. AI-generated artifacts (when enriched).
    │   ├── enrichment.json       ← AI output, embedded into lecture.json on success.
    │   └── prompts/              ← Audit trail (kept iff debug flag set).
    ├── html/                     ← NEW. Rendered HTML output (when enriched).
    │   ├── index.html
    │   └── assets/
    │       ├── lecture.css
    │       └── slide_0001_thumb.jpg  ← Downsized for fast HTML rendering.
    └── processing_log.txt        ← Existing.
```

The existing files are NOT removed. The new `lecture.json` becomes the source of truth and the legacy text files are derived views written for backward compatibility and external tools (e.g., users who still want a plain transcript.txt for grep).

## 3. Schema Versioning

Every artifact carries `"schema_version": "1.0.0"`. SemVer applies:

- **Patch** (1.0.x): purely additive optional fields. Old readers ignore unknown fields.
- **Minor** (1.x.0): new required fields with documented defaults that older readers can fall back to.
- **Major** (2.0.0): breaking change. Requires a migration script in `lecture_processor/migrations/`.

Readers MUST check `schema_version`, MUST accept any 1.x.y, and MUST refuse to read 2.x.x without an explicit migration step. This rule prevents the silent-corruption failure mode where a newer writer produces an artifact that an older reader half-understands.

## 4. `lecture.json` Schema

This is the canonical schema. Field-level rules are normative.

```json
{
  "schema_version": "1.0.0",
  "lecture_id": "Module_6_Video_6_2_AI_Human_Pairing",
  "source": {
    "filename": "Module 6_Video 6.2_AI-Human Pairing.mov",
    "absolute_path": "/Users/adrian/Lectures/Module6/...",
    "byte_size": 23349378,
    "sha256": "9f4a...e2",
    "modified_at": "2026-04-26T21:35:14Z"
  },
  "media": {
    "duration_seconds": 612.4,
    "intended_frame_rate": 29.97,
    "is_vfr": false,
    "has_audio": true,
    "recording_speed_input": "2x",
    "recording_speed_output": "1x",
    "normalized_path": "Module_6_Video_6_2_AI_Human_Pairing.mp4",
    "normalized_duration_seconds": 1224.8
  },
  "transcript": {
    "engine": "whisper-cpp",
    "model": "large-v3",
    "coreml_used": true,
    "language": "en",
    "word_count": 1834,
    "text": "In examining the role of AI in creative industries, ...",
    "segments": [
      { "id": 0, "start": 0.0,   "end": 4.32, "text": "In examining the role of AI in creative industries," },
      { "id": 1, "start": 4.32,  "end": 9.18, "text": "it is important for us to first understand..." }
    ]
  },
  "slides": [
    {
      "id": 1,
      "filename": "slide_0001_00-00-00.png",
      "relative_path": "slides/slide_0001_00-00-00.png",
      "thumbnail_path": "html/assets/slide_0001_thumb.jpg",
      "timestamp_seconds": 0.0,
      "image": {
        "width": 1920,
        "height": 1080,
        "byte_size": 412334,
        "sha256": "ab12...ff"
      },
      "linked_segment_ids": [0, 1, 2, 3, 4]
    }
  ],
  "processing": {
    "started_at": "2026-04-27T15:02:11Z",
    "finished_at": "2026-04-27T15:08:43Z",
    "elapsed_seconds": 392.0,
    "host": {
      "platform": "darwin",
      "machine": "arm64",
      "apple_silicon": true,
      "chip": "Apple M1 Max",
      "ram_gb": 32
    },
    "stage_timings": {
      "probe": 0.4,
      "normalize": 89.1,
      "transcribe": 248.7,
      "slides": 53.8
    },
    "warnings": []
  },
  "enrichment": null
}
```

When AI enrichment runs, it replaces `enrichment: null` with:

```json
{
  "enrichment": {
    "schema_version": "1.0.0",
    "provider": "anthropic",
    "model": "claude-opus-4-7",
    "started_at": "2026-04-27T15:11:02Z",
    "finished_at": "2026-04-27T15:13:34Z",
    "elapsed_seconds": 152.0,
    "input_token_estimate": 12480,
    "output_token_estimate": 3120,
    "title": "AI-Human Pairing in Creative Work: Friend or Foe?",
    "executive_summary": "Professor Sawhney introduces three modes of human-AI pairing in creative work — collaborator, catalyst, and competitor — and argues the collaborator mode will dominate because it best aligns with how creative industries actually work...",
    "outline": [
      { "id": 1, "heading": "Three modes of human-AI pairing",   "slide_ids": [1, 2] },
      { "id": 2, "heading": "Collaborator mode in detail",        "slide_ids": [3, 4, 5] },
      { "id": 3, "heading": "Catalyst mode and ideation",         "slide_ids": [6, 7] },
      { "id": 4, "heading": "Competitor mode and where it fits",  "slide_ids": [8] }
    ],
    "slide_analysis": [
      {
        "slide_id": 1,
        "descriptive_filename": "slide_0001_three_modes_intro.png",
        "summary": "Introduces the three-C framework: AI as Collaborator, Catalyst, and Competitor.",
        "tags": ["framework", "human-ai-pairing", "introduction"],
        "instructor_commentary": "Professor Sawhney opens by saying these three modes all start with C — collaborator, catalyst, competitor — and frames them as the central organizing idea for the rest of the lecture."
      }
    ],
    "resources": [
      {
        "title": "Combinatorial and Transformational Creativity (Boden)",
        "url": "https://example.org/...",
        "summary": "Margaret Boden's foundational distinction between combinatorial and transformational creativity, referenced implicitly in the lecture's framing.",
        "source_quality": "academic"
      }
    ],
    "warnings": []
  }
}
```

## 5. Field-Level Rules

### 5.1 Required Fields

These must always be present in any 1.x.y artifact:

- `schema_version`, `lecture_id`, `source.filename`, `source.absolute_path`, `source.sha256`
- `media.duration_seconds`, `media.has_audio`
- `transcript.text`, `transcript.segments`, `transcript.engine`, `transcript.model`
- `slides` (array, may be empty)
- `processing.started_at`, `processing.finished_at`
- `enrichment` (may be `null`; presence of the key is required)

### 5.2 `lecture_id` Generation

`lecture_id` is derived from the source filename via the same `safe_folder_name()` function used today (`pipeline.py`). It MUST be stable across re-runs of the same source file and MUST be a valid filesystem-safe identifier on macOS, Windows, and Linux. Collision handling: append `_2`, `_3`, etc., as today.

### 5.3 `source.sha256`

Computed once during the probe stage. Required because:

- It uniquely identifies the lecture across runs even if the file is moved or renamed.
- Re-enrichment can verify the source hasn't changed.
- Cache keys for AI calls (Phase 3+) will be keyed on this hash.

Hash the first 16 MB of the file plus the byte size, not the whole file. Lecture videos are large; full-file hashing adds 5–15 seconds per file with no real benefit. Document this choice in code comments.

### 5.4 Transcript Segments

Segment `id` is a sequential integer assigned by the pipeline. AI enrichment references segments by `id` only — never by index, never by start time. This makes outline/commentary references stable even if a downstream tool re-orders segments.

`start` and `end` are seconds-from-start, floating point, scaled per the existing `normalized_timestamp_scale` logic. Whatever scaling rule applies to the saved video applies here.

### 5.5 Slides

`linked_segment_ids` is computed by the **slide-transcript linker** (a new pre-AI stage; see 05). For each slide, it lists the transcript segments that fall within that slide's display window — the period from this slide's `timestamp_seconds` to the next slide's `timestamp_seconds`, or to end of media for the last slide.

This linkage happens before AI enrichment and is provider-agnostic. The AI provider receives slides AND their linked segments; it does not have to figure out the timing itself.

### 5.6 `enrichment` Sub-Object

`enrichment` is `null` when AI has not run. It contains its own `schema_version` because the enrichment schema can evolve independently of the host artifact (e.g., adding a `flashcards` field) without forcing a major version bump on `lecture.json`.

`provider` and `model` are recorded so users can tell which model produced which artifact. This is non-negotiable: outputs without provider attribution are useless for trust and reproducibility.

`warnings` lists non-fatal issues encountered during enrichment (e.g., "resource discovery skipped: provider returned no results"). Warnings should be visible to the user in the HTML output and in the UI.

## 6. `batch.json` Schema

```json
{
  "schema_version": "1.0.0",
  "batch_id": "2026-04-27T15-02-11_module6",
  "input_dir": "/Users/adrian/Lectures/Module6",
  "output_dir": "/Users/adrian/Lectures/Module6_processed",
  "started_at": "2026-04-27T15:02:11Z",
  "finished_at": "2026-04-27T15:42:08Z",
  "config": {
    "recording_speed": "2x",
    "audio_quality": "fast",
    "transcription_engine": "whisper-cpp",
    "whisper_model": "large-v3",
    "slide_sensitivity": "medium",
    "concurrent_files": 4
  },
  "lectures": [
    {
      "lecture_id": "Module_6_Video_6_1_AI_and_Creativity_Foundations",
      "source_filename": "Module 6_Video 6.1_AI and Creativity Foundations.mov",
      "status": "completed",
      "elapsed_seconds": 312.4,
      "word_count": 1622,
      "slide_count": 14,
      "enriched": true,
      "lecture_json_path": "Module_6_Video_6_1_AI_and_Creativity_Foundations/lecture.json"
    }
  ],
  "summary": {
    "attempted": 9,
    "completed": 9,
    "failed": 0,
    "skipped": 0,
    "enriched": 9
  }
}
```

`batch.json` is the index that the HTML renderer uses to build `index.html`. It deliberately duplicates the small amount of per-lecture data needed to render the index page so the renderer doesn't have to open every `lecture.json` to build a summary list.

## 7. JSON Schema Files

A formal JSON Schema (Draft 2020-12) MUST live at:

```
src/lecture_processor/schemas/
├── lecture-v1.0.0.json
├── batch-v1.0.0.json
└── enrichment-v1.0.0.json
```

These files are committed and used by:

- A new `lecture_processor.artifacts.validate` module that validates every artifact before write and after read.
- The test suite (see 09) for fixture validation.
- Future tooling: a `lecture-processor validate <folder>` CLI command (post-MVP).

Use the `jsonschema` Python package, already available on PyPI. Validation failures during write are FATAL — the file is not written. Validation failures during read are FATAL — the operation aborts with a clear error. There is no "best-effort" mode. The schema IS the contract.

## 8. Backward Compatibility With Existing Outputs

A user with existing `_processed` folders generated before this change has no `batch.json` or `lecture.json`. The new code MUST:

- Detect the absence of artifacts.
- Treat such folders as legacy outputs.
- Refuse to enrich them and tell the user to re-process the source folder.

This is intentional. Building a migration that reconstructs `lecture.json` from txt/srt/slides is doable but error-prone (timestamps, segment boundaries, hashing) and the source files are still on disk. Cheaper to re-process. Document this clearly in release notes and in the UI when an unknown folder is selected for enrichment.

## 9. Why Not Use SQLite Or A Database

A database was considered. Rejected because:

- The product is currently file/folder oriented and users expect to be able to copy a lecture folder somewhere and have it work. SQLite ties data to a single file outside the lecture folder.
- JSON artifacts diff cleanly in version control if a user wants to track changes.
- HTML rendering benefits from per-folder portability — drop the folder on a USB stick, open `html/index.html`, it works.
- The data volume per batch is tiny (low MBs of JSON). No query performance argument applies.

Reconsider this if a future feature (full-text search across many batches) demands it. Until then, files.

## 10. Migration Strategy For Future Schema Changes

When 2.0.0 ships:

1. Bump `schema_version` to `2.0.0` in writers.
2. Add `src/lecture_processor/migrations/v1_to_v2.py` exporting `def migrate(d: dict) -> dict`.
3. Readers detect 1.x.y and call the migration, then proceed.
4. The migrated artifact is written back to disk if and only if the user is performing a write operation; reads do not silently mutate disk.

This pattern is well-trodden and it's worth setting it up now even though we don't need it yet — adding migration support after the fact is much harder than baking it in from version 1.0.0.
