# Changelog

All notable changes to Lecture Processor are documented here.

## [0.7.2] - 2026-08-17

### Added

- **In-app update checker** — on startup the app checks the latest GitHub
  release (`aaramos/Lecture-Transcriber`), compares it against the running
  version with semver, and prompts the user when an update is available. A
  macOS menu-bar **"Check for Updates…"** command and an **"Update Now"** prompt
  open the release page for download. Includes release-integrity/DMG
  verification (HEAD checksum) before the packaged app is trusted.

### Security

- **Secret-hygiene hardening** — `.gitignore` now carries explicit credential
  exclusions (`.env` / `*.pem` / `*.key` patterns) as defense-in-depth. API keys
  continue to be read exclusively from the macOS Keychain with an env-var
  fallback — no secrets are committed or pushed.

## [0.7.1] - 2026-08-17

### Added

- **Configurable minimum file length** — the short-file skip threshold is now a
  config setting instead of a hardcoded 60s. Default changed to **30 seconds**.
  Wired through `BatchConfig.min_duration_seconds`, CLI `--min-duration`, a new
  GUI spinbox, the web `minDuration` setting, and the Rust/Tauri boundary
  (serde default 30s). Files shorter than the threshold are skipped + logged as
  before.
- **Ollama Cloud model option** — Lecture Transcriber now supports cloud models
  served by Ollama (`https://ollama.com/v1`) alongside the existing oMLX-served
  models. Users can enter an Ollama API key (stored in the macOS Keychain, never
  plaintext) and pick oMLX or Ollama cloud models for the overview/transcript/
  slides/resources AI steps. Includes cloud model refresh and URL config.

### Fixed

- `minDuration` setting was not persisted in `saveCurrentSettings`, so the
  threshold reset to 30s on every app restart. Now saved and restored.
- **Ollama Cloud route provider rejected** — the Rust `run_process_batch`
  validator rejected `ollama-cloud` as a model route provider (it only allowed
  `mlx-text`/`mlx-vision`/`local-stub`/`off`), so cloud batches errored at
  launch. Validation is extracted into `validate_route_provider()` which now
  accepts `ollama-cloud`, with new unit tests covering accept/reject cases.

## [0.7.0] - 2026-08-16

### Fixed

- **Critical:** Packaged app failed at startup with "Could not locate the Lecture
  Processor project root" and wrote a `batch_error.txt` instead of running any
  batch. The Tauri bundle placed project files under
  `Contents/Resources/bundle-resources/project/...`, but the Rust project-root
  resolver only checked `Contents/Resources/project/...`, so it could never
  find `pyproject.toml` + `src/lecture_processor`. Dev builds worked because the
  current directory / executable path resolved to the repo root, which masked
  the bug. The bundle config now targets `Contents/Resources/project/` directly,
  and the resolver was hardened to also check the legacy nested path so future
  config drift cannot silently reintroduce the failure.
- Supersedes the broken **v0.6.1** release; users on v0.6.1 should upgrade.

### Added

- **Notebook export** (`--export-notebook`): renders a markdown notebook
  (per-slide instructor transcript links, headings matched to slide count) from
  the lecture JSON, gated behind a `config.export_notebook` flag with an
  optional `notebook_course` override.

### Removed

- Dropped the orphaned `saga.py` async saga orchestrator and its tests. The
  module was never imported by the application or CLI (only by its own test
  suite), so it shipped as dead code. Removing it keeps the bundle lean.

## [0.6.1] - 2026-07-31

### Known Issues

- **Broken release.** Packaged app cannot locate the project root and fails
  every batch with `batch_error.txt`. Superseded by v0.7.0; do not use.
