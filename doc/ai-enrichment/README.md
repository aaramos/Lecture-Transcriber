# AI Lecture Processor — Architect Deliverable

**Date:** 2026-04-27
**Status:** Approved for implementation
**Audience:** Engineering team building the AI enrichment phase

---

## What This Is

A complete engineering documentation set covering everything needed to build the AI study-material enrichment layer onto the existing AI Lecture Processor desktop app. Read in numeric order; refer back as needed during implementation.

## Files

| File | What it covers | Read when |
|------|---------------|-----------|
| `00_architecture_overview.md` | High-level architecture, principles, MVP boundary, doc map | First |
| `01_data_model_and_schema.md` | `lecture.json` and `batch.json` schemas (the contract) | Before any coding |
| `02_provider_abstraction.md` | `AIProvider` interface, prompts, per-provider notes | Before AI work |
| `03_credential_storage.md` | macOS Keychain design, Tauri commands | Before keychain work |
| `04_html_output_spec.md` | HTML structure, CSS approach, accessibility, templates | Before renderer work |
| `05_pipeline_and_events.md` | New stages, event taxonomy, cancellation, retry | Before orchestrator work |
| `06_ui_changes.md` | Settings, dialogs, progress, post-batch summary | Before frontend work |
| `07_refactor_plan.md` | Pre-AI refactors that must land first (Sprint 0) | Before Sprint 0 |
| `08_phased_implementation_plan.md` | Phase-by-phase breakdown, acceptance criteria, estimates | At the start of each phase |
| `09_test_strategy.md` | Test layers, fixtures, coverage targets, CI | Before writing tests |
| `10_risks_and_open_decisions.md` | What might go wrong, what's still up to debate | Before final design review |
| `schemas/lecture-v1.0.0.json` | JSON Schema for the lecture artifact | At implementation time |
| `schemas/batch-v1.0.0.json` | JSON Schema for the batch artifact | At implementation time |
| `schemas/enrichment-v1.0.0.json` | JSON Schema for the AI enrichment block | At implementation time |

## TL;DR

**The product:** existing local batch processor + a new optional AI enrichment phase + new optional HTML rendering phase. AI never blocks local outputs. Local processing keeps working unchanged.

**The data contract:** every stage reads and writes a single `lecture.json` per lecture, validated against a committed JSON Schema. HTML is rendered from the artifact. AI enrichment writes to the artifact. Both can be re-run without re-doing earlier stages.

**The provider model:** one ABC, three eventual implementations (Anthropic ships in MVP; Gemini and Grok in Phase 3). Adding a fourth provider is a one-file change.

**The credentials:** macOS Keychain via Rust today, abstraction supports Windows later. Keys never reach a CLI arg, env var, log line, or localStorage.

**The MVP scope:** Phase 0 (refactors) + Phase 1 (mock provider) + Phase 2 (Anthropic + keychain + HTML renderer + UI). About 5–6 weeks for one engineer.

**The deferred-to-later list:** Gemini/Grok adapters, resource discovery, AI slide rename on disk, embedded image metadata, per-lecture reprocess UI, multi-profile keys, telemetry. The architecture supports each; none ship in MVP.

## Implementation Sequence

```
Phase 0    →    Phase 1    →    Phase 2    →    Phase 3
Refactor       Provider ABC     Anthropic +     Multi-provider +
sprint         + Mock + e2e     keychain +      resources +
(~5 days)      (~8 days)        HTML + UI       polish
                                (~13 days)      (~9 days)
```

Total to MVP ship: **~26 working days.** Total to full feature set: **~35 working days.**

## Top Three Things An Engineer Reads First

1. **`00_architecture_overview.md` §3 (Architectural Principles)** — five rules that drove every other decision. Internalize them.
2. **`01_data_model_and_schema.md` §4 (`lecture.json` Schema)** — the contract every stage reads and writes. Memorize the field names.
3. **`07_refactor_plan.md`** — these refactors land first. Don't start the AI work until they're done.

## Top Three Risks

1. **Prompt quality drift.** The first prompt won't be the last; budget time for iteration before declaring MVP shipped.
2. **Provider rate limits.** Sequential default, retry with backoff, per-lecture failure isolation. Test ahead with realistic batch sizes.
3. **Scope creep.** The MVP is intentionally tight. Push every "while we're at it" suggestion into Phase 3+.

## Sign-off

This document set covers all ten architect-handoff questions and produces concrete contracts for engineers. After product owner review, it becomes the source of truth for the next two months of work.

When prompts, schemas, or provider behaviors change, update the relevant doc here and version-bump where applicable. The docs are part of the codebase, not separate from it.
