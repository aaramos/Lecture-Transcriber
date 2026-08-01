# Lecture Processor code review — 2026-07-31

## Scope

This review covered the Python processing pipeline, transcription backends, slide extraction, AI enrichment, Tauri process management, frontend build, dependency bootstrap, and release metadata. The primary risk scenario was a long-running batch containing many large lecture files.

## Confirmed issues fixed in 0.6.1

| Severity | Problem | User impact | Resolution |
| --- | --- | --- | --- |
| Critical | A staged AI exception was handled as one batch-wide failure. | One lecture or provider error could mark every lecture's enrichment failed. | The staged path now falls back to isolated per-lecture retries, preserves successful artifacts, and continues after an individual failure. |
| Critical | Deferred HTML rendering was not isolated per lecture. | One malformed artifact or rendering error could abort the batch after transcription had completed. | Rendering failures now fail only the affected lecture while the batch summary and remaining HTML pages complete. |
| High | The desktop shell retained all processor stdout and stderr for the lifetime of a run. | Verbose ML/native logs could grow memory without a bound during a large batch. | Each stream now keeps a bounded 1 MiB tail and explicitly marks omitted earlier output. |
| High | Parakeet processed a whole long recording as one inference unit. | Long audio could create a large inference-memory spike and destabilize later files. | Parakeet 0.5.1 is pinned and inference is split into 10-minute chunks with 15-second overlap. The model remains single-threaded because MLX binds work to its initialization thread. |
| High | Completed output was accepted based only on the source filename. | Replacing or moving a recording under the same filename could silently reuse stale output. | New artifacts store a resolved source path; resume checks also compare path, byte size, and modification time before skipping work. Legacy artifacts remain readable. |
| High | The documented Node bootstrap kept stale wrappers after the Codex runtime moved. | `npm test`, lint, frontend build, and the desktop release build could not start. | The environment script now discovers an available ARM-native Node binary and refreshes its local wrappers every time it is loaded. |
| Medium | Desktop local-AI discovery parsed only numeric socket addresses. | Valid settings such as `http://localhost:1234/v1` failed before a connection was attempted. | Hostnames are now resolved to socket addresses while preserving the five-second connection/read limits. |
| Medium | OpenCV accepted corrupt/extreme frame-rate metadata and ignored image-write failure. | Slide extraction could skip frames incorrectly or report success without saved slides. | Frame rates outside 1–240 fps fall back to 30 fps, and a failed image write now produces a per-file processing error. |
| Medium | Timestamp scaling could create negative or reversed transcript segments. | Invalid SRT timing and slide/transcript links could be emitted. | Scaled timestamps are clamped to zero and each end is clamped to its start. |
| Medium | Local AI URL validation and progress callback failures were silent or late. | Invalid server settings reached runtime, and a broken event sink was invisible during diagnosis. | URLs are validated before processing; callback failures are reported to stderr without stopping the batch. |
| Medium | Release versions disagreed between package manifests and the npm lockfile. | Builds could identify themselves as different releases. | Python, npm, npm lock, Cargo, Cargo lock, and Tauri metadata now agree on 0.6.1. |
| Medium | The local release environment lacked Pillow and OpenCV. | Video slide processing and four image-path tests were unavailable in the app bundle. | The ARM64 environment now includes Pillow 12.3.0 and OpenCV 4.11.0; the complete image-path suite runs. |
| Medium | The desktop resource wildcard copied untracked files, caches, and Finder metadata into release builds. | A local prototype or stale bytecode could be distributed accidentally and make builds non-reproducible. | The bundle is now assembled from Git-tracked processor files plus the two required media tools, and the local app is ad-hoc signed before the DMG is created. |

## Verification completed

- Full tracked Python suite: 256 tests passed with no skips.
- Rust unit tests: 7 passed.
- Rust Clippy: clean with warnings treated as errors.
- Frontend unit test, JavaScript/JSON lint, and production frontend build: passed.
- Python dependency consistency (`pip check`) and bytecode compilation: passed.
- Real Apple Silicon Parakeet smoke test: loaded the cached 0.6B model, transcribed generated speech correctly, and returned sentence timestamps.
- Packaged macOS app and DMG: ARM64 build, valid DMG checksum, verified ad-hoc signature, clean tracked-resource manifest, and live UI launch including the Parakeet setting.

## Remaining operational limits

- No representative multi-hour lecture corpus is stored in this repository, so an end-to-end hours-long media batch was not run as part of this review. The failure and continuation paths are covered with deterministic multi-file tests.
- External AI enrichment still depends on the selected local server, model availability, and credentials. Requests have bounded timeouts, and failures are now isolated, but the app cannot make an unavailable external model succeed.
- The local macOS build is not Apple-notarized. It is suitable for this Mac and source-controlled release distribution, but public distribution would require signing/notarization credentials.
