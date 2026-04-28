# 05 — Pipeline & Progress Events

## 1. Pipeline Stages

### 1.1 Existing (preserved unchanged)

```
Probe → Normalize → Transcribe → Slides → Write artifact + legacy text files
```

The local pipeline always runs. Its outputs are unchanged from today except for the new `lecture.json` written at the end (legacy `transcript.txt`, `transcript.srt`, `processing_log.txt` are still written from the artifact).

### 1.2 New (additive, opt-in)

```
[local pipeline finishes] → Link slides to segments → AI Enrichment → Render HTML
```

The three new stages run only when:

- AI enrichment is toggled on for the batch, AND
- A valid provider key is in keychain, AND
- The local pipeline produced a complete `lecture.json` for this lecture.

If any of those is false, the lecture's enrichment is skipped, its `enrichment` field stays `null`, and a fallback HTML page is still generated (showing transcript + slides without AI summaries) IF the user opted into HTML rendering.

### 1.3 HTML rendering as a separate toggle

HTML rendering is its own toggle, independent of AI enrichment. Combinations:

| Local | AI | HTML | Result |
|-------|----|----- |--------|
| ✅    | ❌ | ❌   | Today's behavior. Plain text outputs. |
| ✅    | ❌ | ✅   | HTML pages with transcript and slides, no AI sections. |
| ✅    | ✅ | ❌   | `lecture.json` with `enrichment` populated, no HTML rendered. Useful for users who want the data and roll their own output. |
| ✅    | ✅ | ✅   | Full study material. The default once configured. |

This matrix matters because it forces clean separation: enrichment and rendering must each work in isolation.

## 2. Where The New Stages Live

### 2.1 Slide-transcript linker (`lecture_processor/ai/linker.py`)

A pure function. No I/O, no AI. Takes `slides[]` and `segments[]`, returns slides with `linked_segment_ids` populated.

```python
def link_slides_to_segments(
    slides: list[dict],
    segments: list[dict],
) -> list[dict]:
    """For each slide, find segments that fall in its display window."""
    # ...
```

Edge cases (must be tested):

- Last slide owns segments from its timestamp through end of media.
- A segment that spans a slide boundary (start before, end after the next slide) is linked to the slide it started in.
- A slide with no segments in its window has `linked_segment_ids: []` — this is valid and the AI prompt handles it.

This stage runs as part of the artifact write, not as part of enrichment. The artifact always has linked segments, so re-running enrichment doesn't redo the linking.

### 2.2 AI enrichment orchestrator (`lecture_processor/ai/enrichment.py`)

```python
class EnrichmentOrchestrator:
    def __init__(
        self,
        provider: AIProvider,
        progress_callback: Callable[[dict], None],
        cancel_event: threading.Event,
    ): ...

    def enrich_batch(self, batch_dir: Path) -> EnrichmentBatchSummary: ...
    def enrich_lecture(self, lecture_dir: Path) -> EnrichmentResult: ...
```

The orchestrator is the only thing that calls providers. It owns retry, cancel, progress, and writing the enriched artifact back to disk atomically.

### 2.3 HTML renderer (`lecture_processor/html/renderer.py`)

Already specified in document 04. Pure read-from-artifact, write-to-disk.

## 3. Concurrency

### 3.1 Local pipeline

Existing `ThreadPoolExecutor` with `concurrent_files` (1–8). No change.

### 3.2 AI enrichment

Default: **sequential** (concurrency = 1). Reasons:

- API rate limits: every provider rate-limits by tokens/minute and requests/minute. Three concurrent calls hit those limits fast.
- Cost predictability: a sequential run with a known per-lecture cost is easier to reason about than parallel calls that might trigger 429s and retry storms.
- Cancellation: cancelling a sequential run stops cleanly. Cancelling N parallel calls means tracking N cancel handles.

A `concurrent_ai_calls` setting (default 1, max 3) is exposed for users who know what they're doing. Document the rate-limit risk in the UI helper text. Anything above 3 should be considered out of scope; users with that need go to the API directly.

## 4. Event Taxonomy

The frontend already consumes JSON events from the Python CLI via the `__LECTURE_PROCESSOR_EVENT__ ` line prefix mechanism. Extend it.

### 4.1 Existing events (kept)

```
batch_started    { attempted, output_dir }
batch_finished   { attempted, completed, failed, skipped }
file_started     { source }
file_finished    { source, status, message, completed, failed, skipped, attempted, word_count, slide_count }
step_started     { source, step }
step_finished    { source, step, elapsed_seconds }
```

### 4.2 New events for AI enrichment

```
enrich_batch_started      { attempted, provider, model }
enrich_batch_finished     { attempted, completed, failed, skipped }
enrich_lecture_started    { lecture_id, source }
enrich_lecture_progress   { lecture_id, fraction, stage }
                          # stage is one of: "preparing", "uploading",
                          # "waiting", "streaming", "parsing", "validating", "writing"
enrich_lecture_finished   { lecture_id, status, message,
                            input_tokens, output_tokens, elapsed_seconds }
enrich_retry              { lecture_id, attempt, reason, retry_after_seconds }
enrich_warning            { lecture_id, message }
```

### 4.3 New events for HTML rendering

```
render_batch_started      { attempted }
render_lecture_started    { lecture_id }
render_lecture_finished   { lecture_id, status, message, html_path }
render_batch_finished     { attempted, completed, failed }
```

Field naming: snake_case in Python events, the existing Rust serde rename rule is **only on the request structs**, not on pass-through JSON. Re-check `serde_json::Value` is used in the streaming path so events pass through unchanged. (See `lib.rs` line ~330; today the events are deserialized as `serde_json::Value` and emitted to the webview without renaming. Keep that.)

### 4.4 Why per-stage progress fractions

Long AI calls (60–120 seconds for a long lecture) need movement on the progress bar or users assume the app froze. The provider's streaming response supplies natural progress signals (each chunk received, parsing started, validation started). The frontend uses `enrich_lecture_progress.fraction` to drive a per-lecture progress bar.

If the provider doesn't support streaming, emit:

- `progress 0.05 stage="preparing"` once
- `progress 0.10 stage="waiting"` immediately after the request is sent
- `progress 0.95 stage="parsing"` when the response arrives
- `progress 1.00 stage="writing"` after the artifact is written

So progress moves a little to confirm the call is in flight, then sits at "waiting" — that's honest.

## 5. Cancellation

### 5.1 Existing model

The Rust bridge spawns Python in its own process group and SIGTERMs the group on cancel. This works for the local pipeline because every step is either fast or interruptible (FFmpeg respects SIGTERM, Whisper is in a transcribe loop the OS can kill).

### 5.2 New model for AI calls

The same SIGTERM works for AI calls — Python catches it and aborts. **But:** the in-flight HTTP request will be cancelled mid-stream, which provider SDKs may handle ungracefully (open sockets, lingering retries).

Better: cooperative cancellation via a `threading.Event`.

- Rust cancellation continues to set the existing flag in its `ActiveProcess` state and SIGTERM the process group as a last resort.
- Python registers a `SIGTERM` handler that sets `cancel_event.set()`.
- The orchestrator passes a `cancel_check` lambda (`lambda: cancel_event.is_set()`) to the provider's `analyze_lecture` call.
- The provider's adapter checks `cancel_check()` between streaming chunks and aborts when it returns True.
- After the lecture's adapter call returns or raises, the orchestrator checks the event and exits the loop without enriching the next lecture.
- Already-enriched lectures keep their results. Partially-enriched in-flight lectures lose their work.

If the user cancels and the SIGTERM handler doesn't fire within 5 seconds (network library hung in C code), Rust falls back to SIGKILL. We accept losing the in-flight request in that case.

### 5.3 What "cancel" looks like to the user

- Cancel during local processing: same as today (already-completed lectures keep their outputs).
- Cancel during AI enrichment: completed lectures keep their full enriched outputs. The lecture in-flight loses its enrichment. The user sees the enrichment progress bar stop and a banner: "Cancelled. {N} lectures enriched, {M} remaining were not processed."

## 6. Retry Policy

Owned by the orchestrator. Per-lecture retry budget = 3 attempts. Backoff: 1s, 4s, 16s. Exponential factor 4 (not 2) to give rate limits time to clear.

```python
attempts = 0
while True:
    try:
        response = provider.analyze_lecture(request, cancel_check=...)
        break
    except ProviderRateLimitError as e:
        wait = e.retry_after_seconds or (4 ** attempts)
        emit("enrich_retry", lecture_id=..., attempt=attempts, reason="rate_limit", retry_after_seconds=wait)
        time.sleep(wait)
        attempts += 1
    except ProviderTransientError:
        emit("enrich_retry", lecture_id=..., attempt=attempts, reason="transient")
        time.sleep(4 ** attempts)
        attempts += 1
    if attempts >= 3:
        # Mark this lecture failed, emit enrich_lecture_finished status="failed", continue.
        break
```

Auth errors and request errors abort. Response errors get one retry then fail (see error matrix in document 02).

## 7. Partial Output

The orchestrator writes the lecture artifact incrementally:

1. After local processing: artifact has `enrichment: null`.
2. Before AI call: nothing changes on disk.
3. On AI success: rewrite artifact with `enrichment: {...}`.
4. On AI failure: artifact stays at step 1 form. The lecture's HTML (if enabled) renders without AI sections.
5. The batch finishes; `batch.json` records which lectures were enriched.

There is **no** "half-enriched" state on disk. The artifact write is atomic (existing `write_text_atomic` pattern). If the process dies mid-call, restarting the enrichment re-tries from scratch for that lecture; all completed lectures keep their results.

## 8. Re-enrichment Idempotency

A user can re-run enrichment on an already-enriched batch. Behavior:

- For each lecture, the artifact's `source.sha256` is verified against the current source file (if available) — flag mismatches as warnings, do not abort.
- If `enrichment` is already populated, prompt: skip / overwrite / overwrite-changed-only. MVP supports skip and overwrite. "Changed only" requires comparing prompt versions and provider/model — defer to Phase 3.
- On overwrite, the previous enrichment is replaced. We don't keep history in MVP. (Adding history is a future schema change: `enrichment_history: [...]` field.)

## 9. Logging

Every enrichment writes a per-lecture log line to `processing_log.txt` (existing file):

```
Enrich    OK  152.0s  (anthropic claude-opus-4-7  in:12480 out:3120)
```

On failure:

```
Enrich    ERROR  Rate limit exceeded after 3 retries
```

This keeps existing log inspection workflows working.

## 10. Telemetry & Metrics

**MVP: none.** No usage metrics, no provider call counts sent anywhere. The user sees their own counts in the UI; we don't.

If we add telemetry in Phase 3+, it MUST be opt-in, MUST default off, and MUST never include any lecture content or filenames — only counts and durations. Document this in privacy policy at the time of implementation.
