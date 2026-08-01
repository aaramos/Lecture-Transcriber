# Lecture Processor

Local app for processing lecture recording folders.

The implementation has two layers: a Python processing core and a Tauri desktop shell. The desktop app is the main user path; the CLI remains available for automation and debugging.

Version 0.6.1 adds Parakeet MLX transcription and hardens long-running batches: staged AI and HTML failures are isolated per lecture, desktop log retention is bounded, changed source files are reprocessed, and Parakeet uses bounded inference chunks. See the [0.6.1 code review](doc/code-review-2026-07-31.md) for findings, fixes, verification, and remaining operational limits.

## What It Does

- Scans a supported lecture file or folder. Video files get the full transcript, slides, and study-page flow; audio and transcript files get transcript-first study pages without slide extraction.
- Skips files shorter than 60 seconds.
- Optionally normalizes 2x recordings to 1x playback with FFmpeg.
- Extracts a temporary clean 16 kHz mono WAV for transcription, then deletes it after use.
- Runs transcription through a local speech-to-text profile.
- Extracts slide images from visual scene changes.
- Writes per-file output folders, transcripts, SRT files, slides, and processing logs.
- Writes structured `lecture.json` and `batch.json` artifacts for study-page rendering.
- Generates offline HTML study pages for each lecture and a batch index page.
- Optionally enriches study pages with mock AI output or LM Studio-generated titles, summaries, outlines, slide notes, and resources.
- Continues the batch if one file fails.
- Prints a final attempted/completed/failed/skipped summary.
- Provides a desktop UI for selecting or dropping folders, speed, output location, and processing settings.
- Streams live per-file progress from the Python processor into the desktop app.

## Requirements

Local dependency bootstrap has been completed for this workspace. In a new terminal, load the project-local tools with:

```bash
. ./scripts/dev-env.sh
```

That puts these local tools on your shell path:

- Python virtual environment: `.venv`
- FFmpeg/ffprobe: `.tools/darwin_arm64`
- Rust/Cargo/Tauri CLI: `.tools/cargo`
- npm via the Codex Node binary: `.tools/npm`

The Python environment can be recreated with:

```bash
python3 -m pip install -e .
python3 -m pip install -e ".[transcription,slides]"
```

Apple Silicon acceleration adds one more optional setup step:

```bash
. ./scripts/dev-env.sh
scripts/setup-whisper-cpp-coreml.sh large-v3
```

The CoreML setup script builds `pywhispercpp` with CoreML support, downloads the matching `ggml-*.bin` model, and generates the matching `*-encoder.mlmodelc` encoder bundle. Keep the `.bin` file and `*-encoder.mlmodelc` folder together in the same model directory. `faster-whisper` is the default transcription engine for compatibility, and the new optional `Parakeet MLX` profile is available in the transcription profile dropdown for Apple Silicon setups.

FFmpeg is installed locally in `.tools/darwin_arm64`; the app auto-discovers that path when system `ffmpeg` and `ffprobe` are not available.

Without installing, you can run the module from the repo with:

```bash
PYTHONPATH=src python3 -m lecture_processor --help
```

For tests only, no external media tools are required:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Desktop App

Build the desktop app:

```bash
. ./scripts/dev-env.sh
npm install
npm run build
cargo tauri build
```

The verified macOS app bundle is created at:

```text
src-tauri/target/release/bundle/macos/Lecture Processor.app
```

For development:

```bash
. ./scripts/dev-env.sh
cargo tauri dev
```

Useful desktop shortcuts:

- `Command+,` opens Settings.
- `Command+R` starts the selected batch.
- `Escape` clears the current folder when no dialog is open.

## CLI Usage

Launch the lightweight Python desktop wrapper:

```bash
lecture-processor-app
```

Or run the processing core directly from the command line.

For normal-speed recordings:

```bash
lecture-processor process /path/to/lectures --recording-speed 1x
```

For 2x recordings:

```bash
lecture-processor process /path/to/lectures \
  --recording-speed 2x \
  --confirm-normalization
```

Useful options:

```bash
lecture-processor process /path/to/lectures \
  --output /path/to/output \
  --concurrent 2 \
  --transcription-quality balanced \
  --whisper-model large-v3 \
  --transcription-engine faster-whisper \
  --ffmpeg-hwaccel auto \
  --slide-backend auto \
  --slide-sensitivity medium
```

Transcription quality modes:

- `accurate`: closest to the original higher-accuracy defaults.
- `balanced`: faster while staying close to the original accuracy target.
- `fast`: fastest local mode for drafts or quick review.

Whisper model choices now include `large-v3` for best accuracy and `medium.en`
for faster English lecture transcription.

To generate local HTML only, use the default `--ai-provider none`. To preview the AI study-note flow without an API call:

```bash
lecture-processor process /path/to/lectures \
  --transcription-engine none \
  --ai-provider mock
```

To use oMLX / LM Studio enrichment from the CLI, start the local server and pick the same models you use in the desktop app:

```bash
lecture-processor process /path/to/lectures \
  --ai-provider lm-studio \
  --ai-overview-model your-text-model \
  --ai-transcript-model your-text-model \
  --ai-slides-model your-vision-model \
  --ai-resources-model your-resource-model \
  --mlx-text-url http://192.168.86.22:1234/v1 \
  --mlx-vision-url http://192.168.86.22:1234/v1
```

The desktop app uses one oMLX / LM Studio server URL for text, vision, and resources. If your server requires a token, Settings stores it in macOS Keychain and passes it only to the processor process.

## Vision Model Test Harness

`harness.py` compares local LM Studio vision models and prompt files against the same image folder over time. It appends each run to `results.json` and regenerates `report.md` from the full history.

Install the small harness extra if this workspace has not already been bootstrapped:

```bash
python3 -m pip install -e ".[vision-harness]"
```

Prompts live in `prompts/` and are referenced by filename without `.txt`:

```text
prompts/classify_v1.txt
```

First run, including the reusable baseline:

```bash
python harness.py \
  --model outlier-vision-35b-a3b-mlx \
  --prompt classify_v1 \
  --batch ~/Desktop/batchImageTest \
  --codex-model qwen2.5-vl-7b-instruct
```

Later runs reuse the baseline automatically:

```bash
python harness.py --model qwen2.5-vl-32b-instruct --prompt classify_v1 --batch ~/Desktop/batchImageTest
python harness.py --model qwen2.5-vl-32b-instruct --prompt classify_v2 --batch ~/Desktop/batchImageTest
```

Profiles tune LM Studio load and inference settings for a run. `VISION_HQ` is the default quality-focused profile with a 32k context on the M3 Ultra; `VISION_TURBO` uses a 16k context and shorter generation budget for faster screening:

```bash
python harness.py --model qwen2.5-vl-32b-instruct --prompt classify_v2 --batch ~/Desktop/batchImageTest --profile VISION_TURBO
```

The same profile layer also defines `SUMMARIZATION_HQ` and `SUMMARIZATION_TURBO` for text-summary workloads. Those use non-greedy sampling, repetition controls, and longer text contexts; the Vision Web UI keeps its dropdown scoped to the two vision profiles.

Useful read-only commands:

```bash
python harness.py --list-runs
python harness.py --report-only
```

The default oMLX / LM Studio URL is `http://192.168.86.22:1234`. Use `--base-url` if the server moves.

To use the local Web UI:

```bash
python harness_web.py
```

Then open `http://127.0.0.1:8765`, choose or drag an image folder, refresh LM Studio models, select one or more vision models, choose a profile, and run the test. The Web UI unloads the current LM Studio model before loading the next selected model so each run is isolated.

Use the Vision Test selector when you want a new prompt experiment without mixing results. `Vision Test #1` keeps the original `results.json` and `report.md`; new tests get their own files under `.harness_tests/`, reuse the same uploaded image folder and Codex baseline by default, and start with no model runs recorded.

The selected profile is saved per Vision Test, so each experiment keeps using its chosen profile until you change it.

If a model crashes or LM Studio reports that no model is loaded, the Web UI retries the load once. If the model still cannot answer, the remaining images for that model are marked as errors without sending the same failing request dozens of times.

Codex agreement is shown only when real baseline verdicts exist. Empty baseline placeholders are ignored so the Results tab does not imply that agreement scoring is available when it is not.

If LM Studio returns `401`, paste the LM Studio API token into the Web UI and click Save Token. The token is saved to macOS Keychain under the `Vision Harness` service and is not written to `results.json`, `report.md`, browser storage, or a plain text file. You can also start the Web UI with `LM_STUDIO_API_KEY` or `LM_API_TOKEN` in the environment.

`--audio-quality high` uses FFmpeg's `rubberband` filter when the installed FFmpeg build includes it. The bundled local FFmpeg does not, so the processor automatically falls back to `atempo` instead of failing the batch.

## Transcription Reliability

Before Whisper runs, the processor creates a hidden temporary `.transcription_audio.wav` in the lecture output folder. That file is mono, 16 kHz PCM audio with basic speech cleanup/normalization when the installed FFmpeg build supports it. The file is removed immediately after transcription, and startup cleanup removes stale copies from interrupted runs.

`faster-whisper` uses English transcription settings with voice-activity filtering, previous-text conditioning disabled, and mild anti-repetition controls. Those settings are meant to reduce the repeated ending loops found in Module 6 while keeping the output faithful to the lecture audio.

## Performance Backends

On Apple Silicon, the Tauri shell detects the platform and passes `--apple-silicon` to the Python processor. That uses FFmpeg VideoToolbox hardware decode for normalization and FFmpeg for slide-frame extraction instead of OpenCV. Transcription still defaults to `faster-whisper`; select `whisper.cpp` explicitly when testing CoreML/Metal acceleration.

Useful speed controls:

- `--transcription-engine whisper-cpp` forces the whisper.cpp backend.
- `--transcription-quality accurate|balanced|fast` controls the faster-whisper accuracy/speed tradeoff.
- `--require-whisper-cpp-coreml` fails fast unless `pywhispercpp` reports CoreML support.
- `--ffmpeg-hwaccel auto` uses VideoToolbox on Apple Silicon and falls back to software decode if hardware decode is not accepted for a file.
- `--slide-backend ffmpeg` forces FFmpeg frame sampling; `auto` falls back to OpenCV if FFmpeg frame extraction is unavailable.
- `--concurrent N` controls how many files are processed at once. The processor still uses `ThreadPoolExecutor`; switch to `ProcessPoolExecutor` only after benchmarking with the installed CoreML whisper.cpp model because process workers would each need their own model/runtime state.

Optional Pillow source rebuild:

```bash
. ./scripts/dev-env.sh
scripts/rebuild-pillow-source.sh
```

If you want to test video normalization and slide extraction before installing Whisper, disable transcription explicitly:

```bash
lecture-processor process /path/to/lectures \
  --recording-speed 2x \
  --confirm-normalization \
  --transcription-engine none
```

## Output

Each source file gets its own output folder:

```text
OutputFolder/
├── index.html
├── batch.json
└── Lecture1/
    ├── Lecture1.mp4
    ├── lecture.json
    ├── html/
    │   └── index.html
    ├── transcript.txt
    ├── transcript.srt
    ├── slides/
    │   └── slide_0001_00-05-30.png
    └── processing_log.txt
```

If `--no-save-normalized-video` is used, transcript and slide timestamps are converted to match the original playback-speed video.
