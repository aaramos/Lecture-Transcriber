# 08 — Phased Implementation Plan

This is the plan engineers execute against. Each phase has explicit acceptance criteria, dependencies, and a definition of "done." Don't move to the next phase until the current one is signed off.

---

## Phase 0 — Refactor Sprint

**Goal:** Land the seven refactors in document 07 without changing user-visible behavior.

**Dependencies:** None.

**Deliverables:**

- R1 artifact writer module + tests.
- R2 lecture.json written by every successful local run, schema-validated.
- R3 `.mp4` and `.m4v` discovered alongside `.mov`.
- R4 slide naming policy committed to `doc/decisions/`.
- R5 Apple Silicon detection cleaned up.
- R6 ProcessingLog module replaces inline log line list.
- R7 CLI stubs for `enrich` and `render`.

**Acceptance criteria:**

- All existing tests pass without modification (except updated R3 tests).
- New `test_artifacts.py` and `test_logging.py` pass.
- Running an existing lecture batch produces all existing output files PLUS `lecture.json` and `batch.json`.
- Existing `lecture.json` artifacts validate against `lecture-v1.0.0.json` schema.
- `lecture-processor enrich` and `lecture-processor render` parse args and exit with documented "not implemented" message.

**Estimate:** 1 sprint (5–6 working days).

---

## Phase 1 — Provider Adapter & Local Enrichment Path

**Goal:** Make `lecture-processor enrich --provider mock` work end-to-end. No real network calls. Proves the orchestrator, the artifact write-back, and the event stream all work.

**Dependencies:** Phase 0.

**Deliverables:**

- `ai/providers/base.py` with the full ABC and exception types from document 02.
- `ai/providers/mock.py` returning canned schema-conformant responses.
- `ai/providers/registry.py`.
- `ai/enrichment.py` orchestrator with retry, cancel, progress.
- `ai/linker.py` slide-transcript linker (pure function).
- `ai/prompts/registry.py` and stub prompt files.
- `ai/tokens.py` (token counting via `tiktoken` for now; replace with provider-native counts later).
- Wired into `cli.py enrich` subcommand.
- Schema files in `schemas/` directory committed.

**Acceptance criteria:**

- `lecture-processor enrich /path/to/output --provider mock --json-events` populates `enrichment` field in every lecture's artifact.
- Cancel via SIGTERM stops cleanly, leaves completed lectures enriched and in-flight lecture's artifact unchanged.
- All emitted events match the taxonomy in document 05.
- Mock provider tests cover: success, rate limit retry, transient retry, auth fail, response parse fail.
- Tests for `link_slides_to_segments` cover edge cases from document 05 §2.1.

**Estimate:** 1.5 sprints (7–9 working days).

---

## Phase 2 — Anthropic Provider, Keychain, HTML Renderer, MVP UI

**Goal:** Ship the MVP. Real network calls, real keys, real HTML.

**Dependencies:** Phase 1.

**Deliverables:**

### Backend
- `ai/providers/anthropic.py` implementing the contract using the `anthropic` Python SDK.
- The MVP version of the prompt (`ai/prompts/v1/analyze_lecture.txt`).
- `html/renderer.py`, templates, static CSS/JS.
- `html/thumbnails.py` using Pillow with the Apple Silicon Accelerate path when available.
- `cli.py render` subcommand fully implemented.

### Tauri / Rust
- `keychain_set`, `keychain_get`, `keychain_delete`, `test_provider`, `list_providers`, `enrich_batch`, `render_batch` commands.
- macOS Keychain implementation via `security-framework` crate.
- `--read-key-from-stdin` wiring (key sent via single stdin line then closed).
- New events in the Rust event passthrough (already passes through unchanged; verify with logged samples).

### Frontend
- AI Enrichment section in settings dialog.
- Privacy disclosure dialog (first-use modal).
- Main-screen toggle.
- Active video list updated to show enrichment progress.
- Post-batch summary updated to show enrichment counts and Open HTML button.

**Acceptance criteria:**

- A user can install the app fresh, paste an Anthropic API key in settings, click Test Connection, see a green check, run a batch, and open `html/index.html` to see enriched content.
- The local-only path (AI off, HTML off) still produces the same outputs as today byte-for-byte (compare via test fixture).
- Cancelling mid-enrichment produces the documented partial-output state.
- API key never appears in any log file, environment variable, or process command line. Verified by grepping the test outputs and `ps` output during a test run.
- Privacy dialog appears on first AI enable per provider, never again unless provider is switched.
- HTML renders correctly on Safari, Chrome, Firefox, mobile Safari (visual smoke test).
- The schemas validate every artifact written.

**Estimate:** 2.5 sprints (12–15 working days).

**Risk callout:** This phase has the most cross-cutting concerns. If something slips, slip the **prompt iteration**, not the keychain or schema work. Keychain and schema are platform foundations; prompts can be refined post-MVP via a prompt-only release.

---

## Phase 3 — Multi-Provider, Resource Discovery, Polish

**Goal:** Add Gemini and Grok adapters. Add external resource discovery as an optional enrichment step. Polish everything that bit-rotted in Phase 2 crunch.

**Dependencies:** Phase 2.

**Deliverables:**

- `ai/providers/gemini.py`, `ai/providers/grok.py` implementing the same contract.
- `ai/prompts/v1/discover_resources.txt` and orchestrator wiring.
- `enrichment.resources` field populated when the user opts in.
- UI: provider picker actually shows three live providers.
- Concurrent AI calls setting wired through (max 3).
- Bulk operations: "Retry failed enrichments only" works.

**Acceptance criteria:**

- Switching providers in settings, running a batch, switching back, and re-running shows two distinct `provider`/`model` fields on the same lecture's enrichment.
- Resource discovery produces 3–5 reasonable links for a representative lecture (manual eval; no automated quality gate).
- All three providers pass the contract test suite.

**Estimate:** 2 sprints (8–10 working days).

---

## Phase 4+ — Future Work (Architectural Slots Reserved)

Items the architecture supports but doesn't ship. Each is a future ticket with its slot in the design.

| Feature | Architectural slot |
|---------|---|
| AI-renamed slide files on export | New CLI subcommand `export-renamed` reads `enrichment.slide_analysis[].descriptive_filename` and copies/symlinks. No artifact change. |
| Embedded XMP/EXIF metadata | Add to `html/thumbnails.py` step. Use `piexif` or `exiftool`. No schema change. |
| Per-lecture reprocess UI | New Tauri command `reprocess_lecture` with stage flags. Backend already supports per-lecture enrichment. |
| Quiz / flashcard generation | New prompt `v2/generate_flashcards.txt`. New artifact field `enrichment.flashcards`. Optional schema bump 1.1.0 (additive). |
| Full-text search across batches | Add SQLite index built from `lecture.json` files. Separate side-table; no artifact change. |
| PDF export | Headless browser (Playwright) reads the rendered HTML and prints PDF. No core change. |
| Non-English lectures | Whisper already supports them; AI prompt becomes language-aware. Schema's `transcript.language` field is already there. |
| Windows port | Implement `WindowsCredentialStore`, swap a couple `cfg!(target_os)` paths in Rust, validate FFmpeg/whisper.cpp installs. Schema and enrichment stay identical. |

---

## Critical Path

```
Phase 0  →  Phase 1  →  Phase 2  →  Phase 3
  5d         8d         13d         9d
```

**Total to MVP: Phase 0 + Phase 1 + Phase 2 = ~26 working days = 5–6 weeks for one engineer.**

**Total to full feature set: ~35 working days = 7–8 weeks.**

This assumes:
- One full-time engineer, or two engineers running roughly in parallel where possible.
- Anthropic API is reachable from the dev environment.
- No major prompt-quality issues (always a risk; budget +20% for prompt iteration).
- The `pywhispercpp` + CoreML stack is stable on the dev M1 Max (already validated per existing code).

---

## Parallelization Opportunities

If two engineers are available:

- Engineer A: Phase 0 → Phase 1 backend → Phase 2 backend (provider, prompt, renderer).
- Engineer B (starts at Phase 0 done): Phase 2 Tauri/keychain + Phase 2 frontend.

Coordination: agree on the Tauri command signatures (already specified in document 02–06) before starting Phase 2 in parallel.

If three engineers: split Phase 2 frontend off as a third stream and delete two days from the schedule.

---

## "Done" Definition For Each Phase

A phase is done when:

1. All acceptance criteria above are met.
2. Test coverage hasn't regressed (CI green).
3. The relevant docs are updated (`README.md`, schema docs, release notes draft).
4. A 5-minute demo to the product owner runs on the M1 Max without surprises.

If any of those four is false, the phase isn't done. Don't ship MVP without all four. Don't move to Phase 3 without all four for Phase 2.
