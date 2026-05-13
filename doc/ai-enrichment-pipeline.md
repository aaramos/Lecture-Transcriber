# AI Enrichment Pipeline

Date: 2026-05-03
Scope: Current Lecture Transcriber AI enrichment behavior, with emphasis on the local LM Studio route.

## Instruction Confirmation

This document uses the requested matrix columns:

- Step
- What is submitted to the model
- Purpose
- Prompt / model instructions
- Output

## Product Summary

AI enrichment starts after the app has already created a `lecture.json` artifact with transcript text, transcript segments, slide metadata, media duration, and extracted slide images. The enrichment pipeline reads that artifact, builds an `AnalyzeLectureRequest`, runs the configured AI steps, then writes the result back into the `enrichment` section of the same `lecture.json`.

The current local LM Studio path is split into four user-facing outputs:

- Overview
- Transcript cleanup
- Slide analysis
- Resources

The app can run those as a single lecture flow or as a staged batch flow. In staged batch mode, it runs one AI role across all lectures before moving to the next role, which helps avoid LM Studio loading and unloading multiple local models during a batch.

The default local route is:

1. Overview with the text model
2. Transcript cleanup with the text model
3. Slide analysis with the same LM Studio model used for text, with the vision adapter loaded
4. Resources with the resource/text model

The router is dependency-aware. Resources depend on the overview, but not on slide analysis. If resources use the same text model as overview/transcript, the router may run resources before slides to avoid an unnecessary model switch. This does not change the final `lecture.json` shape.

Transcript cleanup note: the cleanup model does **not** receive the entire lecture in one request. The app splits the transcript into chunks of about 20,000 characters and sends one chunk at a time. Those cleaned chunks are joined afterward. The overview step is the whole-lecture summarization pass and can now receive up to 100,000 transcript characters.

Slide context note: each slide is linked to transcript segments spoken during that slide's display window, from the slide timestamp up to the next slide timestamp. This keeps adjacent slide prompts from carrying the same repeated segment list.

## Safety And Availability Rules

- The app checks whether selected LM Studio models are available before running local AI steps.
- If a selected local model is unavailable, only the affected AI role is skipped, and the artifact receives a clear warning.
- LM Studio requests are serialized by the app so local model calls are not intentionally run in parallel.
- The app no longer unloads a model just because the pipeline is moving to a new step.
- Loaded-model verification is cached briefly inside a stage to avoid repeated `/models` checks, then cleared at stage boundaries and when LM Studio load/unload state changes.
- Every pipeline system prompt includes `/no_think`. The app does not send LM Studio's optional `chat_template_kwargs` field by default because this LM Studio server rejects it as an unrecognized key.
- Non-AI processing still runs even if AI enrichment is skipped.

## Pipeline Matrix

The prompt cells below include the actual current prompt templates. Braced values such as `{lecture_id}` are filled in by the app at runtime.

<table>
  <thead>
    <tr>
      <th>Step</th>
      <th>What is submitted to the model</th>
      <th>Purpose</th>
      <th>Prompt / model instructions</th>
      <th>Output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Preflight: LM Studio availability</td>
      <td>No prompt is submitted to a model. The app calls the LM Studio model-list endpoint for the configured text, vision, and resource base URLs.</td>
      <td>Confirm the selected local models exist before running enrichment.</td>
      <td>None. This is an app-side availability check, not a generation prompt.</td>
      <td><code>RouteAvailability</code>: resolved model names, skipped roles, and warnings that are written to processing/enrichment when a role is unavailable.</td>
    </tr>
    <tr>
      <td>Overview</td>
      <td>Lecture id, duration in minutes, transcript trimmed to 100,000 characters, and slide index metadata.</td>
      <td>Create the study-note overview from as much of the lecture as practical so the summary is not based on an incomplete transcript.</td>
      <td>
        <strong>System</strong>
        <pre><code>You create concise study-note overviews from lecture transcripts. /no_think</code></pre>
        <strong>User</strong>
        <pre><code>Respond immediately with only the JSON object. Do not include planning, analysis, reasoning, or any text before or after the JSON.
Create a lecture overview with these fields:
- title: short, specific lecture title
- executive_summary: one concise paragraph
- outline: array of objects with id, heading, slide_ids
- warnings: array of strings

Lecture id: {lecture_id}
Duration minutes: {duration_minutes:.2f}

Transcript:
{transcript_text_trimmed_to_100000_chars}

Slides:
{slide_index_text}</code></pre>
      </td>
      <td><code>title</code>, <code>executive_summary</code>, <code>outline</code>, <code>warnings</code>, plus token estimates. If the model fails, the app fills a local overview fallback and records the failure warning.</td>
    </tr>
    <tr>
      <td>Transcript cleanup</td>
      <td>Transcript text split into chunks of up to about 20,000 characters. Each request includes only the current chunk plus the current chunk number and total chunk count.</td>
      <td>Convert raw transcript text into readable paragraphs without changing the substance.</td>
      <td>
        <strong>System</strong>
        <pre><code>You are a careful lecture transcript copy editor. Preserve meaning. /no_think</code></pre>
        <strong>User</strong>
        <pre><code>Respond immediately with only the JSON object. Do not include planning, analysis, reasoning, or any text before or after the JSON.
Return fields:
- formatted_transcript: polished transcript text with readable paragraph breaks
- warnings: array of strings, each noting an edit that changed wording beyond punctuation, capitalization, or spacing.

Format this lecture transcript chunk for a student reading it later.

Rules:
- Make light copy edits only: punctuation, capitalization, spacing, repeated words, and obvious transcription glitches.
- Add paragraph breaks every 2-5 sentences or whenever the topic shifts.
- Use blank lines between paragraphs so the result is not one large blob.
- Keep the instructor's voice and the original order of ideas.
- Preserve names, technical terms, examples, and substantive details.
- Do not summarize, add new ideas, or remove meaningful content.
- Never add headings. If the speaker introduces a section verbally, leave the title as plain text in the paragraph.

Chunk {chunk_index}/{chunk_count}:
{chunk_text}</code></pre>
      </td>
      <td>Cleaned transcript chunks are joined into one <code>formatted_transcript</code>, with <code>warnings</code> and token estimates. If a chunk fails, that chunk falls back to the raw transcript text and records a warning.</td>
    </tr>
    <tr>
      <td>Slide analysis</td>
      <td>Slide batches containing slide ids, timestamps, linked transcript segment ids, attached slide images as data URLs, and transcript context for the linked segments. Default batch size is 4, bounded by <code>LECTURE_SLIDE_BATCH_SIZE</code> from 1 to 8.</td>
      <td>Generate visual captions, slide summaries, tags, and instructor commentary for each slide while preserving one output row per slide and flagging low-value frames.</td>
      <td>
        <strong>System</strong>
        <pre><code>You analyze lecture slide images and return compact JSON. /no_think</code></pre>
        <strong>User</strong>
        <pre><code>Respond immediately with only the JSON object. Do not include planning, analysis, reasoning, or any text before or after the JSON.
Return field slide_analysis.
Return one slide_analysis item for every slide listed.
Each item must include: slide_id, descriptive_filename, caption, summary, tags, instructor_commentary.

Analyze the attached image for each listed slide_id. For caption and summary, describe only what is visible in
the image. Do not use transcript context to guess visual content.
For each attached image, use only the visual content for that same slide_id. Do not mix visual details between slide_ids.

If the image is a speaker-only frame, transition screen, video-player screen, or otherwise not an actual slide,
say that directly in caption and summary. Do not borrow the topic from nearby transcript.
Treat low-value frames as discard candidates: use tags such as "low-value", "speaker-only", "transition",
"video-player", "duplicate", or "not-slide" when they apply, and keep their summary brief instead of inventing
educational value.

=== VISUAL ANALYSIS INPUT (use for caption, summary, tags) ===
Slides:
{slide_index_text}

=== TRANSCRIPT CONTEXT (use ONLY for instructor_commentary, never for caption/summary/tags) ===
{batch_transcript_context}</code></pre>
      </td>
      <td><code>slide_analysis</code> array and <code>warnings</code>. Missing slide ids are retried. If a batch still misses slides, those slides are retried one at a time. If a slide still fails, local fallback fills that slide.</td>
    </tr>
    <tr>
      <td>Resource query planner</td>
      <td>Overview title, executive summary, and outline headings.</td>
      <td>Ask the text model to suggest focused web search queries before the app searches directly.</td>
      <td>
        <strong>System</strong>
        <pre><code>You plan high-quality resource searches for lecture study materials. /no_think</code></pre>
        <strong>User</strong>
        <pre><code>Respond immediately with only the JSON object. Do not include planning, analysis, reasoning, or any text before or after the JSON.
Create focused web search queries for finding external study resources for this lecture.

Rules:
- Return 3 to 6 web-search-ready queries.
- Prefer queries that find official docs, university pages, peer-reviewed papers, reputable educational sources, or strong industry reports.
- Do not include URLs.
- Keep each query concise and specific.

Lecture title: {title}
Executive summary: {executive_summary}
Outline headings:
{outline_headings_text}</code></pre>
      </td>
      <td>Query plan with <code>queries</code> and <code>warnings</code>. If planning fails or returns no usable queries, the app uses heuristic fallback queries.</td>
    </tr>
    <tr>
      <td>Direct resource search</td>
      <td>No model prompt. The app submits planned or fallback web queries to Brave Search, then checks candidate URLs directly.</td>
      <td>Gather real candidate resources without relying on the model to invent links.</td>
      <td>None. This is app-owned search and URL verification.</td>
      <td>Verified resource candidates and search warnings. If candidates are found, they are sent to the resource formatter. If not, the pipeline can try the LM Studio web-tool path.</td>
    </tr>
    <tr>
      <td>Resource formatter: direct candidates</td>
      <td>Overview title, executive summary, outline headings, and verified candidate resources from direct search.</td>
      <td>Rank and summarize the verified candidates into the final resources list.</td>
      <td>
        <strong>System</strong>
        <pre><code>You rank verified lecture study resources and return compact JSON only. /no_think</code></pre>
        <strong>User</strong>
        <pre><code>Respond immediately with only the JSON object. Do not include planning, analysis, reasoning, or any text before or after the JSON.
Return fields:
- resources: array of up to 6 objects with title, url, summary, source_quality
- warnings: array of strings

Rank and summarize these verified search candidates for a student.

Rules:
- Include only URLs from the candidate list.
- Keep URLs exactly as provided.
- Prefer official, university, peer-reviewed, or reputable educational sources.
- Each summary should explain why the resource helps with this lecture.
- source_quality:
  - "high" = official documentation, peer-reviewed papers, .edu/.gov domains, established textbooks
  - "medium" = reputable industry blogs, journalism from major outlets, well-cited tutorials

Lecture title: {title}
Executive summary: {executive_summary}
Outline headings:
{outline_headings_text}

Verified resource candidates:
{verified_resource_candidates_json}</code></pre>
      </td>
      <td>Final <code>resources</code> array with title, URL, summary, and source quality. If formatting returns too few usable links, the app fills remaining slots from verified search results. If formatting fails, it saves the verified candidates directly.</td>
    </tr>
    <tr>
      <td>Resource gather: LM Studio web tools fallback</td>
      <td>Overview title, executive summary, and outline headings. The model may call LM Studio web-search/fetch integrations when direct search did not produce candidates.</td>
      <td>Gather candidate resources through LM Studio's configured web tools as a fallback path.</td>
      <td>
        <strong>System</strong>
        <pre><code>You are a research assistant for lecture study resources. Search the web, verify URLs, and gather concise source notes. /no_think</code></pre>
        <strong>User</strong>
        <pre><code>Use LM Studio's web-search tools to gather candidate study resources for this lecture.

Find 4-6 high-quality external resources that help a student go deeper on the lecture's main topics.

Rules:
- Prefer official documentation, university pages, peer-reviewed papers, reputable educational sources, or
  well-known industry reports.
- Use fetch when it helps confirm a search result.
- Only keep URLs you verified through the web tools. Do not invent or guess URLs.
- Return concise notes for each candidate with title, URL, why it is relevant, and source quality.
- Run these searches in order: {planner_queries}. Stop early once you have 4-6 verified candidates.
- This is a research-gathering pass. Do not return JSON yet.

Lecture title: {title}
Executive summary: {executive_summary}
Outline headings:
{outline_headings_text}</code></pre>
      </td>
      <td>Gathered web context, tool-call names, token estimates, and warnings. If no tool calls are reported but links are present, the app tries to extract resources from the text.</td>
    </tr>
    <tr>
      <td>Resource formatter: LM Studio gathered context</td>
      <td>Overview title, executive summary, outline headings, and gathered web context from the LM Studio web-tool pass.</td>
      <td>Convert gathered web research into the final resources shape.</td>
      <td>
        <strong>System</strong>
        <pre><code>You format verified lecture resource notes as compact JSON only. /no_think</code></pre>
        <strong>User</strong>
        <pre><code>Respond immediately with only the JSON object. Do not include planning, analysis, reasoning, or any text before or after the JSON.
Return fields:
- resources: array of up to 6 objects with title, url, summary, source_quality
- warnings: array of strings

Format the gathered web research into study resources.

Rules:
- Include only resources from the gathered context.
- Every resource must include title, url, summary, and source_quality.
- source_quality:
  - "high" = official documentation, peer-reviewed papers, .edu/.gov domains, established textbooks
  - "medium" = reputable industry blogs, journalism from major outlets, well-cited tutorials
- Do not invent or guess URLs.
- If the gathered context does not contain useful verified URLs, return an empty resources array and explain the problem in warnings.

Lecture title: {title}
Executive summary: {executive_summary}
Outline headings:
{outline_headings_text}

Gathered web context:
{gathered_web_context_trimmed_to_12000_chars}</code></pre>
      </td>
      <td>Final <code>resources</code> array and <code>warnings</code>. If formatting fails but verified links can be extracted, the app uses those extracted links. Otherwise it returns an empty resources payload with warnings.</td>
    </tr>
    <tr>
      <td>Final artifact write</td>
      <td>No model prompt. The app combines step results into the enrichment payload.</td>
      <td>Preserve a stable <code>lecture.json</code> contract for the UI and downstream consumers.</td>
      <td>None. This is app-side assembly.</td>
      <td><code>enrichment</code> object containing schema version, provider, model, routing info when active, start/end timestamps, elapsed seconds, per-step timings, token estimates, title, summary, formatted transcript, outline, slide analysis, resources, and warnings.</td>
    </tr>
  </tbody>
</table>

## Output Shape Written To `lecture.json`

The final enrichment payload preserves the existing artifact shape:

```json
{
  "schema_version": "1.1.0",
  "provider": "experimental-routing",
  "model": "string",
  "prompt_version": "v1",
  "started_at": "ISO timestamp",
  "finished_at": "ISO timestamp",
  "elapsed_seconds": 0,
  "input_token_estimate": 0,
  "output_token_estimate": 0,
  "title": "string",
  "executive_summary": "string",
  "formatted_transcript": "string",
  "outline": [],
  "slide_analysis": [],
  "resources": [],
  "step_timings": {
    "overview": {"started_at": "ISO timestamp", "finished_at": "ISO timestamp", "elapsed_seconds": 0},
    "transcript_cleanup": {"started_at": "ISO timestamp", "finished_at": "ISO timestamp", "elapsed_seconds": 0, "chunk_count": 0},
    "slide_analysis": {"started_at": "ISO timestamp", "finished_at": "ISO timestamp", "elapsed_seconds": 0, "batch_count": 0, "retry_count": 0},
    "resource_planner": {"started_at": "ISO timestamp", "finished_at": "ISO timestamp", "elapsed_seconds": 0},
    "resource_search": {"started_at": "ISO timestamp", "finished_at": "ISO timestamp", "elapsed_seconds": 0},
    "resource_formatter": {"started_at": "ISO timestamp", "finished_at": "ISO timestamp", "elapsed_seconds": 0}
  },
  "warnings": []
}
```

When model routing is active, the payload also includes `model_routing`, showing which provider and model were configured for overview, transcript, slides, and resources.

## Fallback Behavior

The pipeline is designed to finish gracefully even when one model step fails.

- Overview failure uses a local overview fallback.
- Transcript chunk failure preserves the raw chunk for that section.
- Slide batch failure retries each slide one at a time before local fallback.
- Missing slide rows are filled locally so the final output still has one row per original slide.
- Resource planner failure uses app-generated search queries.
- Resource formatter failure saves verified candidates directly when available.
- Unavailable LM Studio roles are skipped with warnings instead of blocking the whole batch.

## Source Of Truth

The current implementation lives in:

- `src/lecture_processor/ai/enrichment.py`
- `src/lecture_processor/ai/model_routing.py`
- `src/lecture_processor/ai/providers/mlx_openai.py`
- `src/lecture_processor/ai/providers/base.py`
- `src/lecture_processor/config.py`
