# Handoff: LM Studio AI Performance Pass

Date: 2026-05-03
App: Lecture Transcriber
Scope: Local LM Studio AI enhancement pipeline

## Summary

The current AI enhancement pipeline is functionally correct but too slow, mainly because slide analysis sends one request per slide. In the latest reviewed `Test_processed` run, AI enhancement took about 421 seconds. Slide analysis accounted for most of that time because a 17-slide lecture produced 17 sequential vision requests.

The recommended next pass is to keep the current one-model-at-a-time safety rule, but reduce repeated work:

1. Batch slide images into small groups, defaulting to 4 slides per request.
2. Retry missing or low-quality slide results one slide at a time.
3. Cache recent LM Studio loaded-model verification during a stage so each request does not re-check the model list twice.

Do not add AI concurrency. The goal is fewer requests, not multiple simultaneous local model calls.

## Opus Recommendation Coverage

This handoff is based on the Opus review of the AI pipeline artifacts and code. It intentionally carries forward these recommendations:

| Opus recommendation | Captured here | Implementation note |
| --- | --- | --- |
| Batch slide analysis instead of one request per slide | Yes | Default to 4 slides per request, bounded by `LECTURE_SLIDE_BATCH_SIZE`. |
| Keep one-model-at-a-time behavior | Yes | Do not add AI concurrency or model pre-warming. |
| Retry missing slide IDs and preserve fallback | Yes | Batch first, retry missing slides, then solo retry, then local fallback. |
| Reduce repeated `/v1/models` checks | Yes | Add short-lived loaded-model verification cache. |
| Clear cache at stage transitions | Yes | Clear before each routed single-file step and each staged batch phase. |
| Scale output budget for batches | Yes | Use `min(8192, max(2048, 1024 * len(batch)))`. |
| Add tests for batching and model-check caching | Yes | See Required Tests. |
| Keep Resource workflow unchanged in this pass | Yes | Resource changes are out of scope. |
| Prefer reversible tuning | Yes | Environment variable first, no new UI surface in this pass. |

The doc is meant to be implementation-ready. A next agent should not need to reread the original Opus review to start work.

## Product Guardrails

These are product requirements, not implementation preferences:

- The app must not intentionally use more than one LM Studio model at the same time.
- The app must not intentionally keep two LM Studio models loaded at once.
- Batch processing should still do non-AI work even if AI is skipped.
- If a selected LM Studio model is unavailable, skip only the affected AI step and write a clear warning.
- Preserve the existing `lecture.json` output shape.
- Do not reintroduce Gemini support in the app or UI.
- Do not add hidden web behavior. Resource discovery remains app-owned and explicit.

## Current Pipeline

For a batch, the app currently does non-AI work first, then AI work in staged order:

1. Overview, using Text Model
2. Transcript Cleanup, using Text Model
3. Slide Analysis, using Vision Model
4. Resources, using Resource Model

The staged design is correct. It prevents LM Studio from unloading or loading multiple models mid-batch. Keep it.

## Evidence From Latest Reviewed Run

The clean `Test_processed` run showed this shape:

| Stage | Requests | Average total tokens | Min total tokens | Median total tokens | Max total tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overview | 1 | 2229 | 2229 | 2229 | 2229 |
| Transcript Cleanup | 1 | 1843 | 1843 | 1843 | 1843 |
| Slide Analysis | 17 | 717 | 584 | 704 | 838 |
| Resource Query Planner | 1 | 429 | 429 | 429 | 429 |
| Resource Formatter | 1 | 1394 | 1394 | 1394 | 1394 |

Total model requests: 21

Total model-list checks observed: 24

AI elapsed time in artifact log: about 420.8 seconds

Key interpretation:

- Slide requests are small, but there are too many of them.
- The model is being asked to do fresh prompt and image processing for every slide.
- The app also calls LM Studio model-list endpoints too often while enforcing the one-loaded-model rule.

## Main Problem

In `src/lecture_processor/ai/providers/mlx_openai.py`, slide analysis currently uses:

```python
SLIDE_BATCH_SIZE = 1
```

There is a real quality reason for this comment in the code: some vision models mix details across images when multiple slides are sent together. That was a correct safety choice for weaker or unstable models.

However, the target machine is a Mac Studio M3 Ultra with 96GB unified memory. That hardware can run better local vision models and makes small-batch slide analysis practical. The right fix is not raw concurrency. The right fix is controlled batching with validation and fallback.

## Hardware And Model Guidance

The target hardware is a Mac Studio M3 Ultra with 96GB unified memory. This changes what is practical, but it does not remove the one-model-at-a-time product rule.

Recommended vision-model trial order:

1. Qwen2.5-VL-32B-Instruct, MLX 4-bit if available.
2. InternVL3-38B, MLX 4-bit if available.
3. Current Gemma-style vision model as fallback.

Why this matters:

- The current one-slide setting was a defensive choice because some vision models mix details between images.
- Better vision models on the M3 Ultra should handle 3 to 5 slide images per request more reliably.
- If a chosen vision model blends slide details, set `LECTURE_SLIDE_BATCH_SIZE=1` and rerun. That restores the old safer behavior.

Do not choose a second simultaneous model just because the M3 Ultra has enough memory. The performance fix is fewer requests, not overlapping local models.

## Required Changes

### 1. Make slide batch size configurable

Replace the hard-coded one-slide setting with a bounded setting.

Recommended first version:

- Default batch size: 4
- Minimum: 1
- Maximum: 8
- Environment variable: `LECTURE_SLIDE_BATCH_SIZE`
- No UI control in this pass unless the product owner asks for one

Reason: this keeps the change reversible. If a model behaves badly with multiple images, set `LECTURE_SLIDE_BATCH_SIZE=1` and the old behavior returns.

### 2. Batch slide analysis safely

Current behavior:

- One vision call per slide.
- Missing slide IDs get one retry.
- Missing slides are filled with local fallback.

New behavior:

- Send up to `LECTURE_SLIDE_BATCH_SIZE` slides in each vision call.
- Keep the existing required JSON output shape:

```json
{
  "slide_analysis": [
    {
      "slide_id": 1,
      "descriptive_filename": "string",
      "caption": "string",
      "summary": "string",
      "tags": ["string"],
      "instructor_commentary": "string"
    }
  ],
  "warnings": []
}
```

- Validate that every requested `slide_id` appears in the response.
- If any slide IDs are missing, retry the missing slides once.
- If the batch still misses slide IDs and the batch size is greater than 1, retry those missing slides individually.
- If an individual slide still fails, preserve the existing local fallback behavior and write a warning.

Important: do not duplicate slides in output. Normalize only after all batch, retry, and solo attempts have been gathered.

### 3. Strengthen the slide prompt

Keep the current prompt structure, but add one instruction:

```text
For each attached image, use only the visual content for that same slide_id. Do not mix visual details between slide_ids.
```

Reason: this directly addresses the known failure mode where a model blends details across nearby slides.

### 4. Scale slide max tokens with batch size

Current slide requests use:

```python
max_tokens=4096
```

Recommended:

```python
max_tokens = min(8192, max(2048, 1024 * len(batch)))
```

Reason: a 4-slide batch needs more output budget than a 1-slide request. This avoids clipped JSON without allowing unlimited output.

### 5. Cache loaded-model verification within a stage

Current behavior:

- Each LM Studio chat checks the loaded model.
- `_loaded_chat_model()` calls the model list endpoint.
- `_verify_single_loaded_model()` calls the model list endpoint again.
- The reviewed run showed more model-list checks than useful model calls.

Recommended behavior:

- Add a short verification cache inside `_OpenAICompatibleClient`.
- Cache key: `(native_base_url, model)`.
- Cache value: verified instance ID plus timestamp.
- TTL: 30 seconds.
- If the same model was just verified as the only loaded model, reuse that verification inside the same stage.
- Clear the cache when:
  - a model is unloaded,
  - a model is loaded,
  - the app moves from one AI stage to the next,
  - a model request fails in a way that suggests server state changed.

This reduces repeated model-list calls while preserving the one-model-at-a-time guardrail.

### 6. Clear verification cache at stage boundaries

In `src/lecture_processor/ai/model_routing.py`, clear the verification cache before each AI stage starts in both flows:

- `routed_analyze_lecture(...)`
- `routed_analyze_lectures_staged(...)`

Reason: inside a stage, caching saves repeated checks. Between stages, the app should re-check because a different model may be needed.

### 7. Keep model offload behavior

Do not remove `_offload_model_between_steps(...)`.

Expected behavior:

- If the next step uses the same model as the previous step, keep it loaded.
- If the next step uses a different local model, unload the previous model before moving on.
- If the next step uses no local model, no extra load should happen.

## Files To Change

Primary:

- `src/lecture_processor/ai/providers/mlx_openai.py`
  - Replace hard-coded `SLIDE_BATCH_SIZE = 1`.
  - Refactor `MLXVisionProvider.analyze_slides(...)` to support safe batches.
  - Add a helper such as `_request_slide_batch(...)` so each model call is isolated and testable.
  - Add a helper such as `_slide_batch_max_tokens(...)` so output budget scales with batch size.
  - Add prompt line that prevents cross-slide mixing.
  - Add model verification cache to `_OpenAICompatibleClient`.
  - Add a small public helper such as `clear_lm_studio_model_verification_cache()`.

- `src/lecture_processor/ai/model_routing.py`
  - Import the cache clear helper.
  - Clear verification cache at the start of each AI step or role stage.
  - Keep existing offload behavior.

Tests:

- `tests/test_mlx_openai_provider.py`
- `tests/test_model_routing.py`

Optional later:

- `src/lecture_processor/config.py`
- `src/main.js`
- `src/styles.css`
- `src-tauri/src/lib.rs`

Only touch UI/config if the batch size becomes user-facing. The recommended first pass is environment-variable-only.

## Suggested Implementation Shape

### Slide batch configuration

Use a helper instead of reading the environment everywhere:

```python
DEFAULT_SLIDE_BATCH_SIZE = 4
MAX_SLIDE_BATCH_SIZE = 8


def _configured_slide_batch_size() -> int:
    raw = os.getenv("LECTURE_SLIDE_BATCH_SIZE", "").strip()
    if not raw:
        return DEFAULT_SLIDE_BATCH_SIZE
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_SLIDE_BATCH_SIZE
    return max(1, min(value, MAX_SLIDE_BATCH_SIZE))
```

Update `_slide_batches(...)` to accept an explicit batch size:

```python
def _slide_batches(slides: List[Dict], batch_size: Optional[int] = None) -> List[List[Dict]]:
    size = max(1, int(batch_size or _configured_slide_batch_size()))
    return [slides[index : index + size] for index in range(0, len(slides), size)]
```

Add a max-token helper:

```python
def _slide_batch_max_tokens(slides: List[Dict]) -> int:
    return min(8192, max(2048, 1024 * max(1, len(slides))))
```

### Slide batching

Keep the existing normalize and fallback behavior, but gather raw items first:

```python
for batch in _slide_batches(request.slides, batch_size):
    raw_items = []
    raw_items.extend(call_model_for(batch))

    missing = _missing_slide_ids(raw_items, batch)
    if missing:
        raw_items.extend(call_model_for(_slides_with_ids(batch, missing)))

    still_missing = _missing_slide_ids(raw_items, batch)
    if still_missing and batch_size > 1:
        for slide in _slides_with_ids(batch, still_missing):
            raw_items.extend(call_model_for([slide]))

    normalized = _normalize_slide_analysis(raw_items, request, batch)
    slide_analysis.extend(normalized)
```

This avoids duplicate fallback rows.

Recommended refactor:

```python
def _request_slide_batch(
    self,
    request: AnalyzeLectureRequest,
    batch: List[Dict],
    *,
    context: str,
    temperature: float,
) -> Tuple[List[Dict], List[str], int, int]:
    prompt = _slide_batch_prompt(request, batch)
    payload, usage = self._client.chat_json_with_lm_studio_images(
        _vision_input_items(request, batch, prompt),
        system="You analyze lecture slide images and return compact JSON.",
        max_tokens=_slide_batch_max_tokens(batch),
        temperature=temperature,
        context=context,
    )
    return (
        list(payload.get("slide_analysis") or []),
        list(payload.get("warnings") or []),
        usage.input_tokens or _estimate_tokens(prompt),
        usage.output_tokens or _estimate_tokens(json.dumps(payload)),
    )
```

Then `analyze_slides(...)` should:

1. Read `batch_size = _configured_slide_batch_size()`.
2. Call `_request_slide_batch(...)` for each batch.
3. Retry missing slide IDs once as a smaller batch.
4. If still missing and original batch size was greater than 1, retry each missing slide individually.
5. Normalize once per original batch.
6. Let `_normalize_slide_analysis(...)` fill only the slides that still failed.

Important: do not append normalized fallback output before the solo retries complete. That would create duplicate slide rows.

### Model verification cache

Keep this cache intentionally short-lived:

```python
_verification_cache: Dict[Tuple[str, str], Tuple[str, float]] = {}
_verification_ttl_seconds = 30.0
```

When there is a valid, unexpired cache entry, `_loaded_chat_model(...)` can return the cached instance ID without another model-list call.

The key is:

```python
(native_base_url, selected_model)
```

The value is:

```python
(verified_instance_id, verified_at_monotonic_time)
```

Expected behavior inside `_loaded_chat_model(...)`:

1. Return immediately for Ollama-style URLs, as today.
2. If there is no `known_payload`, check the verification cache.
3. If the cache entry is younger than 30 seconds, return the cached instance ID.
4. Otherwise, fetch the model list and enforce the current one-loaded-model behavior.
5. When verification succeeds, remember the verified instance ID.

Expected invalidation:

- Clear the verification cache before `_load_model(...)` changes LM Studio state.
- Clear the verification cache after `_unload_instance(...)` changes LM Studio state.
- Clear the verification cache when `_post_lm_studio_native_chat(...)` receives a server-state error.
- Clear the verification cache at every AI stage boundary in model routing.

Add a module-level helper to avoid importing private cache internals elsewhere:

```python
def clear_lm_studio_model_verification_cache() -> None:
    _OpenAICompatibleClient.clear_model_verification_cache()
```

### Stage-boundary cache clearing

In `routed_analyze_lecture(...)`, clear the cache before each step in the ordered step loop.

In `routed_analyze_lectures_staged(...)`, clear the cache before each staged phase begins, not before every file. This keeps the speed benefit within a phase while forcing a fresh check before the next model role.

Do not clear the cache between files inside the same staged phase unless a load/unload happens or a request fails.

### Error and warning behavior

Preserve existing warnings and add only useful new ones:

- Include the configured slide batch size in the slide-stage warning.
- If a batch misses slide IDs, name the missing IDs.
- If a solo retry succeeds, do not leave a scary failure warning for that slide.
- If local fallback fills a slide, write a clear warning that names the slide ID.

Avoid vague warnings such as "AI failed" when the app knows the specific failed batch or slide ID.

## Required Tests

Add or update tests for:

1. `LECTURE_SLIDE_BATCH_SIZE=3` turns 7 slides into batches of `[3, 3, 1]`.
2. Invalid `LECTURE_SLIDE_BATCH_SIZE` falls back to default.
3. Slide batch response with all slide IDs is accepted.
4. Slide batch response with missing slide IDs retries missing slides.
5. Batched slide failure falls back to one-slide requests before local fallback.
6. Final slide output has one item per original slide ID and no duplicates.
7. Model verification cache prevents repeated model-list checks within a stage.
8. Model verification cache clears on unload/load.
9. Stage transition clears the verification cache.

## Manual Verification

Use the same `Test_processed` style batch that exposed the problem.

Before rerun:

- Clear existing AI enhancement fields from the processed folder.
- Confirm LM Studio is running.
- Confirm exactly one model is loaded at the end of each stage.

Recommended smoke test:

```bash
LECTURE_SLIDE_BATCH_SIZE=4
```

Expected result:

- Slide analysis for a 17-slide lecture should use about 5 primary vision requests instead of 17.
- If a model misses slide IDs, warnings should name the missing slide IDs and fallback path.
- `lecture.json` should still include exactly one `slide_analysis` entry per slide.
- AI elapsed time should improve materially.
- LM Studio should not show two models loaded at once.
- LM Studio logs should show fewer repeated model-list checks than before.
- The final loaded model should match the final AI stage model, not a stale earlier model.

## Performance Expectation

For a 17-slide lecture:

- Current slide stage: about 17 vision requests.
- Target slide stage with batch size 4: about 5 primary vision requests.
- Worst case with retries: more than 5 requests, but still usually fewer than 17.

Expected user-visible outcome:

- Faster AI enhancement.
- Less repeated LM Studio server chatter.
- Same output fields.
- Same one-model-at-a-time safety behavior.

## Acceptance Criteria

The feature is done when all of these are true:

1. A lecture with 17 slides uses about 5 primary vision calls at `LECTURE_SLIDE_BATCH_SIZE=4`.
2. `LECTURE_SLIDE_BATCH_SIZE=1` restores the old one-slide-per-call behavior.
3. A mocked missing-slide response retries missing slides and outputs one row per slide.
4. The app never issues parallel LM Studio chat requests.
5. The app never intentionally leaves two LM Studio models loaded.
6. Repeated calls in the same stage do not call the model-list endpoint twice per chat.
7. Stage changes still re-check loaded model state.
8. Existing `lecture.json` consumers continue to work without schema changes.

## Risks

### Cross-slide visual mixing

Risk: a model may describe slide 2 with details from slide 1.

Mitigation:

- Reinforce prompt with slide ID binding.
- Validate slide IDs.
- Retry missing slides individually.
- Keep `LECTURE_SLIDE_BATCH_SIZE=1` as an immediate rollback.

### Clipped JSON

Risk: batched slide output may exceed token budget.

Mitigation:

- Scale `max_tokens` with batch size.
- Keep a hard cap of 8192.

### Stale model cache

Risk: a user manually changes LM Studio loaded models during a stage.

Mitigation:

- Keep TTL short.
- Clear cache on load/unload/stage transitions.
- If LM Studio rejects the request, clear cache and fail with a clear warning.

## Out Of Scope

Do not include these in this pass:

- AI concurrency.
- Parallel LM Studio requests.
- Keeping text and vision models loaded at the same time.
- Streaming responses.
- Gemini support.
- New model selection UI.
- New Resource workflow changes.
- Reworking transcription or audio processing.

## Recommended Commit Message

```text
Improve LM Studio slide batching and model checks

- Batch slide analysis requests with safe per-slide validation and fallback.
- Add a reversible slide batch size setting for local testing.
- Cache short-lived LM Studio loaded-model verification within stages.
- Preserve one-model-at-a-time routing and existing lecture JSON output.
```

## Handoff Prompt For Next Agent

You are working in `/Users/macstudio/Apps/Lecture Transcriber`.

Implement the LM Studio AI performance pass described in `doc/lm-studio-ai-performance-handoff-2026-05-03.md`.

Focus on two changes:

1. Replace one-slide-per-request vision analysis with configurable safe slide batching.
2. Reduce redundant LM Studio model-list checks by adding a short-lived loaded-model verification cache.

Keep the product guardrails:

- Never run multiple LM Studio chat requests concurrently.
- Never intentionally keep two LM Studio models loaded at once.
- Preserve existing `lecture.json` output shapes.
- Do not reintroduce Gemini support.
- Do not add a new UI control unless explicitly requested.

Implementation expectations:

- Default slide batch size is 4.
- Support `LECTURE_SLIDE_BATCH_SIZE`.
- Bound slide batch size between 1 and 8.
- Retry missing slide IDs once.
- If a batched request still misses slide IDs, retry those slides individually.
- Preserve local fallback for any slide that still fails.
- Scale slide `max_tokens` with batch size.
- Add the slide prompt instruction that prevents mixing visual details between slide IDs.
- Cache LM Studio loaded-model verification for 30 seconds within a stage.
- Clear the cache on load, unload, and AI stage transitions.

Tests required:

- Batch size env var behavior.
- Missing slide retry behavior.
- Solo fallback after failed batch behavior.
- No duplicate slide analysis rows.
- Model-list check cache behavior.
- Cache invalidation on load/unload/stage transition.

Run lint and the Python test suite before finishing. If a live LM Studio run is not possible, say that clearly and provide the exact manual smoke test steps.
