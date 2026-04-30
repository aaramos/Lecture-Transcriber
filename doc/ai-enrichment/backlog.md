# Enhancement Backlog

## Flexible Video Import

Status: Deferred by product decision on 2026-04-27.

The app continues to prioritize the existing `.mov` workflow. Support for additional file types such as `.mp4` and `.m4v` remains useful, but it is not part of the first AI study-notes build.

End-user value when revisited:

- Users can process more common lecture video formats without converting files first.
- Folder scanning can describe "video files" instead of ".mov files."
- The UI can accept mixed-format lecture folders.

## DeepFilterNet Speaker-Focused Audio Enhancement

Status: Backlog request from product on 2026-04-29.

Add an optional audio enhancement step that uses the same DeepFilterNet-based approach from the ClearVoice app (`https://github.com/aaramos/Clear-Voice-`) to clean lecture audio before transcription. The goal is to improve speaker focus and reduce background noise before Whisper/MLX transcription runs.

End-user value when revisited:

- Users with noisy lecture recordings can improve transcript quality without running a separate cleanup app first.
- The app can keep a simple setting: `None`, `Hybrid`, or `Strong`.
- The default stays `None` so existing processing behavior, runtime, and audio artifacts do not change unless the user opts in.

Proposed user-facing configuration:

- `None`: skip DeepFilterNet enhancement and keep the current audio pipeline.
- `Hybrid`: use ClearVoice-style FFmpeg cleanup plus DeepFilterNet for balanced speaker cleanup.
- `Strong`: use the stronger DeepFilterNet speaker-isolation path for harder-to-hear recordings.

Implementation notes:

- Reuse the ClearVoice DeepFilterNet dependency approach where practical:
  - managed `deep-filter` binary
  - architecture-specific Apple Silicon and Intel assets
  - SHA-256 verification before use
  - local-only processing, no cloud dependency
- Insert the enhancement step before transcription audio is handed to the selected transcription profile.
- Preserve the final transcription input contract: `16 kHz`, mono, `pcm_s16le`, `.wav`.
- Log which enhancement mode was used in `processing_log.txt` and `lecture.json`.
- Add the setting to CLI/config/schema/Tauri UI so the desktop app and command line agree.

Acceptance criteria:

- User can choose `None`, `Hybrid`, or `Strong` in settings before starting a batch.
- `None` produces the same audio/transcription path as today.
- `Hybrid` and `Strong` run locally and fail gracefully if DeepFilterNet is unavailable.
- Quality is verified on at least one noisy lecture against the current pipeline.
- Runtime impact is reported honestly, since audio cleanup will add processing time.
