# Lecture Processor

Local MVP for processing lecture recording folders.

The first implementation is a Python CLI. It is designed to become the processing core for the desktop app described in `doc/lecture-processor-prd.html`.

## What It Does

- Scans a folder for `.mov` files.
- Skips files shorter than 60 seconds.
- Optionally normalizes 2x recordings to 1x playback with FFmpeg.
- Runs transcription through one controlled Whisper lane.
- Extracts slide images from visual scene changes.
- Writes per-file output folders, transcripts, SRT files, slides, and processing logs.
- Continues the batch if one file fails.
- Prints a final attempted/completed/failed/skipped summary.

## Requirements

System tools:

```bash
brew install ffmpeg
```

Optional Python dependencies:

```bash
python3 -m pip install -e .
python3 -m pip install -e ".[transcription,slides]"
```

Without installing, you can run the module from the repo with:

```bash
PYTHONPATH=src python3 -m lecture_processor --help
```

For tests only, no external media tools are required:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Usage

Launch the lightweight desktop wrapper:

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
  --slide-sensitivity medium
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
    ├── normalized_video.mp4
    ├── transcript.txt
    ├── transcript.srt
    ├── slides/
    │   └── slide_0001_00-05-30.png
    └── processing_log.txt
```

If `--no-save-normalized-video` is used, transcript and slide timestamps are converted to match the original playback-speed video.
