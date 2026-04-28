# Lecture Processor Performance Plan

Date: 2026-04-28
Branch: `codex/dmg-installer-progress`

## Goal

Make Lecture Processor faster and more reliable without quietly lowering transcript quality.

Product direction:

- Accuracy matters.
- Users should get clear choices, not hidden behavior changes.
- Performance work should be benchmarked before deeper rewrites.
- The installer should remain understandable: show what it is doing and why.

## Current State

The app is a Tauri macOS desktop shell around a Python processor.

Current processing flow:

1. Find `.mov` files.
2. Probe media with `ffprobe`.
3. Optionally normalize 2x recordings to 1x with FFmpeg.
4. Extract clean 16 kHz mono WAV for transcription.
5. Transcribe with Whisper.
6. Extract slide images.
7. Write transcript, SRT, JSON, logs, and offline HTML.
8. Optionally enrich output with mock AI or Gemini.

Current default transcription:

- Engine: `faster-whisper`
- Model: `large-v3`
- Decode settings: accuracy-oriented
- Concurrency: UI currently allows high file concurrency

Important correction: the current branch already drains the Python child process stdout and stderr in Rust. That earlier stall risk should remain a verification checklist item, not a future implementation item.

## Implementation Status

Phase 1 is implemented on this branch:

- `TranscriptionQuality` config and CLI setting.
- Desktop UI selector for `Balanced`, `Accurate`, and `Fast`.
- Model selector includes `large-v3` and `medium.en`.
- faster-whisper `compute_type` is applied when creating `WhisperModel(...)`.
- Desktop UI concurrency now defaults to 2 and caps normal users at 3.

The remaining phases below are still planned work.

## Accepted Product Decisions

### Decision 1: Keep The Recommendations Doc, But Fix It First

Recommendation: accepted.

Why: the previous doc mixed current code, proposed code, and stale findings. A reviewer needs clean instructions.

### Decision 2: Keep Stdout/Stderr Drain As Verification, Not A New Fix

Recommendation: accepted.

Why: the branch already has reader threads for both pipes. We should verify this stays true, but not spend time re-implementing it.

### Decision 3: Split Benchmark Commands Into Current vs Future

Recommendation: accepted.

Why: several previous benchmark commands used flags that do not exist yet. Current commands must run today; future commands should be clearly labeled as proposed.

### Decision 4: Put `compute_type` In The Right Place

Recommendation: accepted.

Why: for `faster-whisper`, `compute_type` belongs when creating `WhisperModel(...)`, not inside the transcribe options.

Implementation note:

```python
self._model = module.WhisperModel(model_name, compute_type=compute_type)
```

Do not put `compute_type` into the dictionary passed to `self._model.transcribe(...)`.

### Decision 5: Add A Transcript Quality Toggle

Recommendation: add three modes.

- `Accurate`: closest to today's behavior.
- `Balanced`: faster, intended to be nearly as accurate as today.
- `Fast`: fastest option, acceptable when speed matters more than maximum accuracy.

Recommended default: `Balanced`.

Why: the user prefers accuracy, but also wants meaningful speed improvement. `Balanced` should be the normal default because it protects quality while avoiding the most expensive current settings. `Accurate` remains one click away.

Proposed behavior:

| Mode | Purpose | Suggested settings |
| --- | --- | --- |
| Accurate | Maximum quality | current `beam_size=5`, `best_of=5`, fallback temperatures |
| Balanced | Nearly current accuracy, faster | lower beam/search cost, likely `beam_size=2`, `best_of=2`, `temperature=0.0` |
| Fast | Maximum throughput | `beam_size=1`, `best_of=1`, `temperature=0.0` |

Exact values should be benchmarked with real lecture files before finalizing.

### Decision 6: Add A Model Selector

Recommendation: add a model selector with at least:

- `large-v3`: most accurate, slower.
- `medium.en`: faster for English lectures, likely good enough for many use cases.

Recommended default: `large-v3` for now.

Why: the user prefers accuracy. We should not silently move everyone to a smaller English-only model. Let the user pick `medium.en` when speed matters.

Product copy should be plain:

- `large-v3`: Best accuracy, slower.
- `medium.en`: Faster for English lectures.

### Decision 7: Do Not Rework The Whole Concurrency Model In The Same Pass

Recommendation: accepted.

Why: changing concurrency can create subtle batch bugs. First add safer controls and benchmarks, then decide if the deeper pipeline rewrite is worth it.

Important correction: current code builds one transcriber and wraps it in `LockedTranscriber`, so it does not create four separate Whisper model objects just because `--concurrent 4` is selected. Memory pressure can still happen from one model plus concurrent FFmpeg/slide/temp-frame work, but the old explanation overstated model duplication.

### Decision 8: Cap Normal UI Concurrency Lower

Recommendation: accepted.

UI behavior:

- Default: 2 files.
- Normal max: 3 files.
- Do not show 8 as a normal-user option.

Why: high concurrency sounds faster but can make the Mac slower through memory pressure. Reliable completion is more important than a big number in a control.

CLI behavior:

- Keep the existing CLI limit for advanced testing unless benchmark data says to lower it.
- The UI should be safer than the CLI.

### Decision 9: Do Not Make whisper.cpp/CoreML Default Yet

Recommendation: accepted.

Why: whisper.cpp/CoreML may become the Apple Silicon speed path, but setup and model handling need to be reliable first. For now, expose it as an optimized option, not the default.

### Decision 10: Add A Slide Extraction Method Toggle

Recommendation: add a toggle/select with:

- `Stable`: current sampled-frame plus Python image-diff behavior.
- `Fast Scene Detection`: FFmpeg scene detection using a threshold.

Recommended default: `Stable` until benchmarked.

Why: FFmpeg scene detection should be faster because it avoids writing and diffing lots of temporary frames, but slide styles vary. Keep the current reliable path available.

Future CLI option:

```bash
--slide-method stable
--slide-method scene
```

### Decision 11: Fuse FFmpeg Passes Later

Recommendation: accepted, but not first.

Why: combining audio extraction, slide detection, and normalization may save time, but it makes FFmpeg commands harder to debug. Do it after quality toggles and slide-method benchmarking exist.

### Decision 12: Harden First-Run Setup For Real Distribution

Recommendation: accepted.

Why: first-run setup works for local testing, but public distribution needs stronger dependency handling.

Next hardening items:

- Verified downloads.
- SHA-256 checksums.
- Clear progress per dependency.
- Retry/resume where practical.
- Better failure messages.
- Decision on whether to bundle Python/runtime assets instead of installing from package indexes at first launch.

## Implementation Plan

### Phase 1: Clean Settings And Safe Defaults

Goal: give users clear speed/accuracy choices without changing the pipeline architecture.

Changes:

1. Add `TranscriptionQuality` config enum:
   - `accurate`
   - `balanced`
   - `fast`

2. Add CLI flag:

```bash
--transcription-quality accurate
--transcription-quality balanced
--transcription-quality fast
```

3. Add UI setting:

```text
Transcription Quality:
  Balanced
  Accurate
  Fast
```

4. Update faster-whisper construction so model-level options, including `compute_type`, are applied at model creation.

5. Add `medium.en` to the model selector.

6. Change UI concurrent files:
   - default `2`
   - max `3`

7. Keep CLI `--concurrent` as-is for advanced benchmarking.

Recommended order:

1. Config and CLI.
2. Python transcriber settings.
3. UI settings.
4. Tests.

### Phase 2: Benchmark The New Settings

Goal: prove which settings are actually faster and still accurate enough.

Use a fixed sample folder and compare:

- `large-v3` vs `medium.en`
- `accurate` vs `balanced` vs `fast`
- `--concurrent 1`, `2`, and `3`
- slide backend `auto`, `ffmpeg`, and `opencv` where available

Current commands that work today:

```bash
# Baseline current behavior
/usr/bin/time -l .venv/bin/lecture-processor process ./sample \
  --output /tmp/lecture_perf_baseline \
  --concurrent 4 \
  --transcription-engine faster-whisper \
  --whisper-model large-v3

# Lower concurrency using today's CLI
/usr/bin/time -l .venv/bin/lecture-processor process ./sample \
  --output /tmp/lecture_perf_concurrent_2 \
  --concurrent 2 \
  --transcription-engine faster-whisper \
  --whisper-model large-v3

# Model comparison using today's CLI
/usr/bin/time -l .venv/bin/lecture-processor process ./sample \
  --output /tmp/lecture_perf_medium_en \
  --concurrent 2 \
  --transcription-engine faster-whisper \
  --whisper-model medium.en

# Media-only smoke test
/usr/bin/time -l .venv/bin/lecture-processor process ./sample \
  --output /tmp/lecture_perf_no_transcription \
  --concurrent 2 \
  --transcription-engine none
```

Future commands after Phase 1:

```bash
# Balanced mode
/usr/bin/time -l .venv/bin/lecture-processor process ./sample \
  --output /tmp/lecture_perf_balanced \
  --concurrent 2 \
  --transcription-engine faster-whisper \
  --whisper-model large-v3 \
  --transcription-quality balanced

# Fast mode with faster English model
/usr/bin/time -l .venv/bin/lecture-processor process ./sample \
  --output /tmp/lecture_perf_fast_medium_en \
  --concurrent 2 \
  --transcription-engine faster-whisper \
  --whisper-model medium.en \
  --transcription-quality fast
```

Measure:

- total wall time
- peak memory
- transcript quality spot-check
- slide count quality
- whether UI progress stays alive

### Phase 3: Add Slide Extraction Method Toggle

Goal: let users choose between reliable current behavior and faster FFmpeg scene detection.

Changes:

1. Add config enum:

```text
SlideMethod:
  stable
  scene
```

2. Add CLI flag:

```bash
--slide-method stable
--slide-method scene
```

3. Add UI select:

```text
Slide Detection:
  Stable
  Fast Scene Detection
```

4. Keep current behavior as `stable`.

5. Add FFmpeg scene detection as `scene`.

Suggested FFmpeg direction:

```bash
-vf "select='gt(scene,0.15)',scale=1280:-2" -vsync vfr
```

Benchmark thresholds:

- `0.10`
- `0.15`
- `0.20`
- `0.25`

Recommendation: ship `scene` only after we confirm it does not miss too many real slides.

### Phase 4: Heartbeats And Stall Visibility

Goal: make long-running steps feel alive and easier to debug.

Changes:

1. Emit heartbeat/progress events during transcription.
2. Emit heartbeat/progress events during slide extraction.
3. UI warns if an active step has no heartbeat for 120 seconds.

Plain UI message:

```text
Still working on transcription. This can take a while for long recordings.
```

Only show stronger warnings if the process truly appears stuck.

### Phase 5: Evaluate Pipeline Rewrite

Goal: decide whether a deeper concurrency rewrite is worth it.

Do not start here.

Possible future design:

- Keep 1 transcription active at a time.
- Let FFmpeg-bound work overlap around transcription.
- Keep output rendering cheap and parallel.

Reason to wait:

- The existing `LockedTranscriber` already protects against multiple simultaneous transcriptions.
- The main immediate gains likely come from quality/model settings and slide extraction.

### Phase 6: Fuse FFmpeg Passes

Goal: reduce repeated video decoding.

Possible future design:

- Combine audio extraction and slide-frame extraction into one FFmpeg invocation.
- For 2x normalization, evaluate whether normalized video, audio, and slides can be produced from one pass.

Recommendation: implement behind a feature flag or method toggle first.

Why: the performance upside is real, but debugging a multi-output FFmpeg command is harder.

### Phase 7: First-Run Installer Hardening

Goal: make the DMG safer for real distribution.

Changes:

1. Add explicit dependency manifest.
2. Download tools/models from fixed URLs.
3. Verify SHA-256 checksums.
4. Show progress by dependency.
5. Preserve existing working installs when a download fails.
6. Add retry/recheck flow.

Recommended dependency status rows:

- Python runtime
- Processor package
- FFmpeg / ffprobe
- Whisper backend
- Whisper model
- Optional Gemini support

## What Not To Do Yet

Do not:

- Make whisper.cpp/CoreML the default before setup is reliable.
- Remove the current slide extraction path before scene detection is benchmarked.
- Replace the pipeline concurrency model before measuring the simpler changes.
- Hide speed/accuracy changes behind the scenes.
- Ship public DMGs without signing/notarization review.

## Recommended Follow-Up PR

After this branch, build one focused PR:

**PR: Slide Detection Benchmark Toggle**

Include:

1. `SlideMethod` config/CLI setting.
2. UI selector: Stable vs Fast Scene Detection.
3. FFmpeg scene-detection implementation behind the new setting.
4. Threshold benchmark notes.
5. Tests that stable mode remains available.

Do not include:

- pipeline rewrite
- FFmpeg fusion
- public-release signing/notarization

Why: this is the next smallest useful slice. It can improve slide extraction speed while keeping the current reliable method available.
