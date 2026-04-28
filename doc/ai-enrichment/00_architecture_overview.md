# AI Lecture Processor — Architecture Overview

**Document set version:** 1.0
**Date:** 2026-04-27
**Status:** Approved for implementation
**Audience:** Engineering team implementing the AI enrichment phase

---

## 1. Purpose Of This Document Set

The current product is a working local batch processor: Tauri shell, Rust bridge, Python core, local Whisper transcription, FFmpeg/OpenCV slide extraction, plain-text outputs. This document set defines the architecture for adding the AI study-material enrichment layer described in the PRD without destabilizing the working local path.

The deliverable answers all ten questions in the architect handoff brief and produces concrete contracts that engineers can build to.

## 2. Document Map

| # | Document | What it defines |
|---|---|---|
| 00 | Architecture Overview *(this doc)* | High-level architecture, principles, phased plan, MVP boundary |
| 01 | Data Model & Artifact Schema | The `lecture.json` and `batch.json` schemas that flow between every stage |
| 02 | Provider Abstraction Contract | The `AIProvider` interface, prompt contracts, and per-provider notes for Anthropic, Gemini, Grok |
| 03 | Credential Storage Design | How API keys are stored on macOS now and Windows later |
| 04 | HTML Output Specification | Layout, asset rules, accessibility, offline behavior, templating approach |
| 05 | Pipeline & Progress Events | New stages, event taxonomy, cancellation, retry, partial-output rules |
| 06 | UI Changes Required | Settings additions, progress surfaces, per-lecture reprocess flow |
| 07 | Refactor Plan (Pre-Work) | Code changes that must land before AI enrichment is introduced |
| 08 | Phased Implementation Plan | Phase-by-phase breakdown with acceptance criteria and dependencies |
| 09 | Test Strategy | Unit, integration, contract, golden-file, and manual test coverage |
| 10 | Risks & Open Decisions | What's not yet decided and what could go wrong |

Read 00 → 08 in order before starting work. 01–07 are reference documents you'll return to during implementation.

## 3. Architectural Principles

Five rules govern every design choice in this set.

### 3.1 Local-First Is Non-Negotiable
The local processing path (probe → normalize → transcribe → slides → text outputs) must continue to work with zero AI provider configuration. AI enrichment is a strictly additive layer. A user with no API key, no internet, or a failed API call still gets the current outputs.

### 3.2 The Artifact Is The Interface
Every stage produces or augments a single canonical JSON artifact (`lecture.json`). HTML rendering reads only from that artifact. AI enrichment writes only to that artifact. This decouples the AI layer from the HTML layer and makes both independently testable, swappable, and reproducible.

### 3.3 Providers Are Replaceable Adapters
No code outside `lecture_processor.ai.providers` knows whether the active provider is Anthropic, Gemini, or Grok. Adding a fourth provider requires implementing one interface and registering it; nothing else in the codebase changes.

### 3.4 AI Enrichment Is A Separate Pass
AI runs as a second pipeline phase after local processing completes, not interleaved. This isolates network failures, supports re-running enrichment without redoing transcription, and keeps the local pipeline's runtime characteristics unchanged.

### 3.5 Privacy Is Explicit, Minimization Is Default
The app never sends data to a third party without a configured provider, an explicit user toggle for the current batch, and a one-screen disclosure of what will be transmitted. Default transmission is the transcript text plus downsized slide thumbnails — never the full video, never the full-resolution slide PNGs unless the user opts in.

## 4. Target Architecture (One Diagram)

```
┌─────────────────────────────────────────────────────────────────┐
│                         Tauri Frontend                           │
│   index.html / main.js / styles.css   (no framework, vanilla)    │
└──────────────────────────────┬──────────────────────────────────┘
                               │  Tauri commands + events
┌──────────────────────────────▼──────────────────────────────────┐
│                         Rust Bridge                              │
│   src-tauri/src/lib.rs                                           │
│   - process_batch          - cancel_batch                        │
│   - enrich_batch  (NEW)    - cancel_enrichment  (NEW)            │
│   - test_provider (NEW)    - render_html  (NEW)                  │
│   - keychain_set/get/delete  (NEW, macOS keychain)               │
└──────────────────────────────┬──────────────────────────────────┘
                               │  CLI subprocess + JSON events
┌──────────────────────────────▼──────────────────────────────────┐
│                  Python Processing Core                          │
│  ┌──────────────────┐    ┌──────────────────────────────────┐   │
│  │  Local Pipeline  │───▶│       Artifact Store             │   │
│  │  (existing)      │    │   <out>/<lecture>/lecture.json   │   │
│  │  probe→norm→     │    │   <out>/batch.json               │   │
│  │  transcribe→     │    └──────────────────┬───────────────┘   │
│  │  slides          │                       │                   │
│  └──────────────────┘                       │                   │
│                                             ▼                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           AI Enrichment Pipeline (NEW)                   │   │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐  ┌────────┐ │   │
│  │  │ Slide-   │   │ Lecture  │   │ Resource │  │ Slide  │ │   │
│  │  │ Transcript│──▶│ Analysis │──▶│ Discovery│─▶│ Rename │ │   │
│  │  │ Linker   │   │ (title/  │   │          │  │        │ │   │
│  │  │          │   │ summary/ │   │          │  │        │ │   │
│  │  │          │   │ outline) │   │          │  │        │ │   │
│  │  └──────────┘   └─────┬────┘   └──────────┘  └────────┘ │   │
│  │                       │                                  │   │
│  │             ┌─────────▼─────────┐                        │   │
│  │             │ Provider Adapter  │  (Anthropic|Gemini|    │   │
│  │             │  (one interface)  │   Grok)                │   │
│  │             └───────────────────┘                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                             │                   │
│                                             ▼                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           HTML Renderer (NEW)                            │   │
│  │  Reads lecture.json + batch.json, writes static HTML +   │   │
│  │  CSS + minimal JS. Offline-friendly, no external CDNs.   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 5. Answer Summary For The Ten Architect Questions

| # | Question | Short answer | Detail in |
|---|---|---|---|
| 1 | Where does AI enrichment live? | Separate pipeline phase after local processing, reading and writing the artifact | 05 |
| 2 | Intermediate data model? | `lecture.json` per lecture + `batch.json` per batch, schema-versioned | 01 |
| 3 | Multi-provider support? | `AIProvider` ABC, registry-based discovery, unified prompt contract | 02 |
| 4 | Credential storage? | macOS Keychain via Tauri Rust today; Windows Credential Manager later, behind same trait | 03 |
| 5 | Long-running call UX? | Stage-level events, exponential backoff, partial-success persistence, cooperative cancel | 05 |
| 6 | Optional pass or integrated? | **Optional separate pass.** Always opt-in per batch; never blocks local outputs | 05, 08 |
| 7 | HTML structure? | Static HTML5, semantic landmarks, embedded CSS, local image refs, Jinja2 templates | 04 |
| 8 | Refactors needed first? | Extract artifact writer; widen file discovery; settle slide naming policy; persist transcript JSON | 07 |
| 9 | MVP vs later? | MVP: Anthropic only, title/summary/outline/slide-summaries, index+lecture HTML. Defer: resources, slide rename, multi-provider | 08 |
| 10 | Tests before implementation? | Schema fixtures, provider contract tests, golden-file HTML tests, mock-provider integration | 09 |

## 6. MVP Boundary

**MVP = Phase 1 + Phase 2 in document 08.** It includes:

- The artifact schema (`lecture.json`, `batch.json`) and the artifact writer.
- One provider adapter: **Anthropic Claude** (chosen because it's the most capable on long context for lecture transcripts and has the cleanest streaming API).
- AI-generated lecture title, executive summary, slide-level summaries, and outline.
- Index HTML and lecture HTML pages with embedded CSS, semantic structure, alt text on slide images.
- macOS Keychain credential storage via a small Rust helper.
- New "AI Enrichment" UI section: provider toggle, model picker, "test connection," and a privacy disclosure dialog before the first network call.
- New CLI subcommand `enrich` with its own progress events.

**Explicitly deferred to post-MVP** (documented in 08, Phase 3+):

- Gemini and Grok adapters (the contract supports them, but only Anthropic ships in MVP).
- AI-generated descriptive slide filenames (slides ship with current sequence-based names, descriptive name shown in HTML only).
- External resource recommendations.
- Embedded image metadata (XMP/EXIF tags).
- Per-lecture reprocess UI.
- Full input format expansion beyond `.mov` (handled in refactor phase, see 07).

This boundary keeps Phase 1+2 to roughly 4–6 engineering weeks for one full-time developer and ships a usable product.

## 7. What Engineers Read Next

**Before writing any code:** documents 01, 02, 04, 07. These define the contracts you'll be building to and the refactors that must land first.

**During Phase 1 implementation:** documents 05, 08, 09.

**As reference during Phase 2+:** documents 03, 06, 10.

## 8. Conventions Used Across This Document Set

- Schema field names use `snake_case` to match existing Python code.
- Tauri command names use `snake_case` to match existing Rust code (e.g., `process_batch`, not `processBatch`).
- Events emitted to the frontend use `camelCase` field names because the frontend is JavaScript and the existing Rust serde config does this conversion (`#[serde(rename_all = "camelCase")]`).
- File paths in examples use POSIX separators; the implementation must use `pathlib.Path` and accept both.
- "MUST", "SHOULD", "MAY" follow RFC 2119 meaning. Don't soften them.
