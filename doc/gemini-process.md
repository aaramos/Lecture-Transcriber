# Gemini Enrichment Process

This document explains what Lecture Processor sends to Gemini, why each call exists, and what the app expects back.

## Plain-Language Summary

Gemini enrichment starts after the local lecture processing is done. By that point, the app already has a `lecture.json` with the transcript, transcript segments, slide image files, slide timestamps, and links between slides and nearby transcript segments.

The app does **not** send the whole job to Gemini as one giant request in the normal flow. It breaks enrichment into smaller calls:

1. Create the lecture overview.
2. Lightly clean transcript chunks.
3. Analyze slides in small batches.
4. Find external resources with Google Search grounding.

Each completed Gemini chunk is cached in `.enrichment_partial.json`, so if a run fails or is retried, already-completed chunks can be reused instead of spending the same API calls again.

## Source Data Used

| Data source | Where it comes from | What Gemini receives |
|---|---|---|
| `lecture_id` | `lecture.json` | Stable lecture identifier, usually the output folder name. |
| Transcript text | `lecture.json.transcript.text` | Full transcript for overview and fallback context. |
| Transcript segments | `lecture.json.transcript.segments` | Segment IDs, start/end timestamps, and text. Used for chunking and slide context. |
| Slides | `lecture.json.slides` | Slide IDs, filenames, timestamps, image paths, and linked segment IDs. |
| Slide images | `slides/*.png` under the lecture folder | Uploaded only during slide-batch calls, five slides at a time. |
| Duration | `lecture.json.media.duration_seconds` | Sent as minutes in overview/manual prompts. |
| Model | App setting, default `gemini-2.5-flash` | Used for all Gemini calls in a run. |

## Gemini Calls Matrix

| Step | Call type | How many calls | Data in | Gemini tools/images | Expected data out | Objective |
|---|---:|---:|---|---|---|---|
| 0 | Connection test | Optional, settings/test only | Text: `Reply with OK.` | No tools, no images, max output 4 tokens | Any short OK-style response | Confirm the saved API key and selected model work. |
| 1 | Overview | 1 per lecture | Lecture ID, duration, full transcript, slide IDs, timestamps, linked segment IDs | No Google Search, no images | `title`, `executive_summary`, `outline`, `warnings` | Build the top-level study structure before chunk-specific work. |
| 2 | Transcript chunk | 1 to N per lecture | One transcript chunk, built from timestamped segments. Max chunk target is 12,000 chars. | No Google Search, no images | `formatted_transcript`, `warnings` | Lightly clean the transcript while preserving the instructor's meaning. |
| 3 | Slide batch | 0 to N per lecture, 5 slides per batch | Slide IDs, filenames, timestamps, overview title/summary, nearby transcript for each slide, slide images when available | Images included on first attempt; no Google Search | `slide_analysis`, `warnings` | Produce slide captions, summaries, tags, and instructor commentary. |
| 4 | Resources | 1 per lecture | Lecture ID, overview title, overview summary, transcript fallback summary | Google Search grounding enabled, no images | `resources`, `warnings` | Find 3-4 grounded external resources with real URLs. |
| 5 | Final assembly | No Gemini call | Cached outputs from the steps above | None | App writes `lecture.json.enrichment` | Merge Gemini outputs into the final lecture artifact and render HTML. |

## Runtime Behavior

| Behavior | What happens |
|---|---|
| Progress events | After each chunk completes, the backend emits `enrichment_progress` with completed step count and cumulative token counts. |
| Token counts | The app totals Gemini `prompt_token_count` as tokens in and `candidates_token_count` as tokens out. |
| Cache file | `.enrichment_partial.json` stores completed overview, transcript chunks, slide batches, resources, model, lecture ID, and warnings. |
| Retry behavior | If the cache matches the lecture ID and model, completed chunks are reused. Missing chunks are submitted to Gemini. |
| JSON parsing | Gemini is instructed to return JSON only. The app accepts plain JSON, fenced JSON, or text containing one JSON object. |
| Google Search | Only the Resources step uses Google Search grounding. Other steps do not. |
| Image fallback | Slide-batch calls first try with images. If the response is empty or unusable, the app retries without images using the same prompt. |
| Failure fallback | Some response failures fall back to local transcript/slide-derived content with warnings, so one bad chunk does not always fail the whole lecture. |

## Prompt Templates

The templates below are the active prompts used by the current chunked Gemini flow. Braced values are filled from the processed lecture artifact.

### 1. Overview Prompt

Objective: create the lecture title, executive summary, and slide-aware outline.

Data sent:

| Field | Contents |
|---|---|
| `lecture_id` | Output folder / lecture ID. |
| `duration_minutes` | Lecture duration in minutes. |
| `transcript_text` | Full transcript text. |
| `slides` | One line per slide with ID, timestamp, and linked transcript segment IDs. |

Expected JSON:

```json
{
  "title": "string",
  "executive_summary": "string",
  "outline": [{"id": 1, "heading": "string", "slide_ids": [1, 2]}],
  "warnings": []
}
```

Prompt:

```text
You are creating study notes for a student from a processed lecture transcript.

Return JSON only. Do not include markdown fences.

Create the high-level study structure only:
- title
- executive_summary
- outline
- warnings

The outline should be slide-aware. Each item must include id, heading, and slide_ids.
Use the transcript for concepts and the slide list for the slide_ids.

Expected JSON keys:
- title
- executive_summary
- outline
- warnings

Lecture id: {lecture_id}
Duration minutes: {duration_minutes}

Transcript:
{transcript_text}

Slides:
- Slide {slide_id}: @{timestamp_seconds}s segments={linked_segment_ids}
```

### 2. Transcript Chunk Prompt

Objective: lightly edit transcript text for readability without changing the instructor's meaning.

Data sent:

| Field | Contents |
|---|---|
| `lecture_id` | Output folder / lecture ID. |
| `chunk_index` / `chunk_total` | Position of this transcript chunk. |
| `chunk_text` | Transcript segment lines in the form `[id] start-end: text`, grouped up to about 12,000 characters. |

Expected JSON:

```json
{
  "formatted_transcript": "string",
  "warnings": []
}
```

Prompt:

```text
You are lightly editing one chunk of a lecture transcript for readability.

Return JSON only. Do not include markdown fences.

Expected JSON keys:
- formatted_transcript
- warnings

Rules:
- Fix obvious grammar errors, typos, misspellings, capitalization, punctuation, and spacing.
- Add paragraph breaks where helpful.
- Do not summarize.
- Do not add new ideas.
- Do not remove substantive details.
- Preserve the instructor's meaning.

Lecture id: {lecture_id}
Chunk: {chunk_index} of {chunk_total}

Transcript chunk:
{chunk_text}
```

### 3. Slide Batch Prompt

Objective: create slide-by-slide study notes and visual captions.

Data sent:

| Field | Contents |
|---|---|
| `overview.title` | Title from the Overview step. |
| `overview.executive_summary` | Summary from the Overview step. |
| `slides` | Five slides per batch by default. |
| `slide image bytes` | PNG/JPEG/WebP image bytes when available. |
| `nearby transcript` | Text from linked transcript segments; if none, text within about 90 seconds of the slide timestamp, capped around 2,200 characters per slide. |

Expected JSON:

```json
{
  "slide_analysis": [
    {
      "slide_id": 1,
      "descriptive_filename": "string or null",
      "caption": "string or null",
      "summary": "string",
      "tags": ["string"],
      "instructor_commentary": "string"
    }
  ],
  "warnings": []
}
```

Prompt:

```text
You are creating slide-by-slide study notes for a lecture.

Return JSON only. Do not include markdown fences.

Expected JSON keys:
- slide_analysis
- warnings

Each slide_analysis item must include:
- slide_id
- descriptive_filename
- caption
- summary
- tags
- instructor_commentary

For each slide, inspect the uploaded image if available and write a factual caption describing exactly what is visible.
If the image is blank, a pre-roll frame, or a mid-animation transition with no instructional content, set caption to null.
Use the nearby transcript for what the instructor says about the slide.
Return one slide_analysis item for every slide listed below.

Lecture title: {overview_title}
Lecture summary: {overview_summary}

Slides in this batch:
- Slide {slide_id}: {filename} @{timestamp_seconds}s

Nearby transcript:
Slide {slide_id} nearby transcript:
{nearby_transcript}
```

### 4. Resources Prompt

Objective: use Gemini with Google Search grounding to find external resources.

Data sent:

| Field | Contents |
|---|---|
| `lecture_id` | Output folder / lecture ID. |
| `overview.title` | Title from the Overview step. |
| `overview.executive_summary` | Summary from the Overview step. |
| `fallback summary` | Transcript-derived fallback summary if overview summary is missing. |

Expected JSON:

```json
{
  "resources": [
    {
      "title": "string",
      "url": "string",
      "summary": "string",
      "source_quality": "high"
    }
  ],
  "warnings": []
}
```

Prompt:

```text
Use Google Search grounding to find 3-4 high-quality external resources relevant to this lecture.

Return JSON only. Do not include markdown fences.

Expected JSON keys:
- resources
- warnings

For each resource include:
- title
- url
- summary
- source_quality ("high" or "medium")

Prefer peer-reviewed papers, WEF/McKinsey/industry reports, or reputable educational sources.
Only include URLs confirmed by grounding. Do not invent or guess URLs.

Lecture id: {lecture_id}
Lecture title: {overview_title}
Lecture summary: {overview_summary_or_fallback}
```

## Final Enrichment Output

After the Gemini calls finish, the app writes one `enrichment` block into `lecture.json`.

| Output field | Source |
|---|---|
| `provider` | App config, normally `gemini`. |
| `model` | App config, normally `gemini-2.5-flash`. |
| `prompt_version` | Currently `v1`. |
| `started_at`, `finished_at`, `elapsed_seconds` | Enrichment orchestrator timing. |
| `input_token_estimate` | Sum of Gemini prompt token counts across calls. |
| `output_token_estimate` | Sum of Gemini output token counts across calls. |
| `title` | Overview call. |
| `executive_summary` | Overview call. |
| `formatted_transcript` | Combined transcript chunk calls. |
| `outline` | Overview call, normalized by the app. |
| `slide_analysis` | Combined slide batch calls, normalized by the app. |
| `resources` | Resources call. |
| `warnings` | Deduplicated warnings from all calls and fallback paths. |

## Manual / Legacy Single-Call Prompt

The normal app path uses the chunked process above. The code also keeps a manual test prompt used by `export-gemini-test` and a single-call helper. This is mainly for debugging and manual provider testing.

That prompt sends the full transcript, timestamped transcript segments, slide list, and optionally slide images in one request. It asks for:

| Expected top-level key | Purpose |
|---|---|
| `title` | Lecture title. |
| `executive_summary` | Overall lecture summary. |
| `formatted_transcript` | Lightly edited full transcript. |
| `outline` | Slide-aware outline. |
| `slide_analysis` | One item per slide. |
| `resources` | 3-4 grounded external resources. |
| `warnings` | Non-fatal issues. |

The manual prompt includes the same transcript-editing, slide-caption, and external-resource requirements, but it is heavier and less resilient than the chunked flow.

## Current Product Implications

| Product question | Current answer |
|---|---|
| Why can Gemini take a long time? | A lecture may trigger many calls: one overview, several transcript chunks, one call per five slides, and one resources call. A lecture with 300 slides can generate many slide-batch calls. |
| Why do token counts update during Gemini? | Each completed chunk emits cumulative token counts. |
| Why do re-runs sometimes skip work? | Completed chunks are cached in `.enrichment_partial.json` and reused when lecture ID and model match. |
| Which call uses the internet/search? | Only the Resources call uses Google Search grounding. |
| Are slide images sent? | Yes, only for slide-batch calls, five slides at a time by default. |
| Are videos or audio sent? | No. Gemini receives text artifacts and slide images, not the source video/audio. |
