# Lecture Processor 0.6.1

This release adds Parakeet MLX transcription and makes long batch runs substantially more resilient.

## Highlights

- Isolates AI enrichment and HTML rendering failures to the affected lecture instead of aborting the batch.
- Limits retained processor logs to prevent unbounded desktop memory growth.
- Processes long Parakeet recordings in bounded 10-minute chunks.
- Detects changed or moved source recordings before treating prior output as complete.
- Supports hostnames such as `localhost` during local AI discovery.
- Rejects invalid frame-rate metadata, failed slide writes, invalid timestamps, and invalid local AI URLs with useful errors.
- Packages only tracked processor sources and produces a verified ARM64, ad-hoc-signed local app and DMG.

## Verification

- 256 tracked Python tests
- 7 Rust tests and warning-free Clippy
- Frontend tests, lint, and production build
- Real Parakeet MLX transcription smoke test
- Live macOS app launch and Settings check

See `doc/code-review-2026-07-31.md` for the complete findings and remaining operational limits.
