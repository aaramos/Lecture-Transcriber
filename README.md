# Lecture Processor

Local app for processing lecture recording folders.

The implementation has two layers: a Python processing core and a Tauri desktop shell. The desktop app is the main user path; the CLI remains available for automation and debugging.

## What It Does

- Scans a folder for `.mov` files.
- Skips files shorter than 60 seconds.
- Optionally normalizes 2x recordings to 1x playback with FFmpeg.
- Runs transcription through one controlled Whisper lane.
- Extracts slide images from visual scene changes.
- Writes per-file output folders, transcripts, SRT files, slides, and processing logs.
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

The CoreML setup script builds `pywhispercpp` with CoreML support, downloads the matching `ggml-*.bin` model, and generates the matching `*-encoder.mlmodelc` encoder bundle. Keep the `.bin` file and `.mlmodelc` folder together in the same model directory. The desktop app automatically uses `.models/whisper-cpp` when the selected Whisper model is present there, and it disables whisper.cpp flash attention by default because that path can crash inside Metal on Apple Silicon.

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
  --concurrent 4 \
  --whisper-model large-v3 \
  --transcription-engine auto \
  --ffmpeg-hwaccel auto \
  --slide-backend auto \
  --slide-sensitivity medium
```

`--audio-quality high` uses FFmpeg's `rubberband` filter when the installed FFmpeg build includes it. The bundled local FFmpeg does not, so the processor automatically falls back to `atempo` instead of failing the batch.

## Performance Backends

On Apple Silicon, the Tauri shell detects the platform and passes `--apple-silicon` to the Python processor. That makes `auto` mode prefer `whisper.cpp` through `pywhispercpp` when it is installed, use FFmpeg VideoToolbox hardware decode for normalization, and use FFmpeg for slide-frame extraction instead of OpenCV.

Useful speed controls:

- `--transcription-engine whisper-cpp` forces the whisper.cpp backend.
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
└── Lecture1/
    ├── Lecture1.mp4
    ├── transcript.txt
    ├── transcript.srt
    ├── slides/
    │   └── slide_0001_00-05-30.png
    └── processing_log.txt
```

If `--no-save-normalized-video` is used, transcript and slide timestamps are converted to match the original playback-speed video.
