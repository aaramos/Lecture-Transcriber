# 02 — Provider Abstraction Contract

This document defines the `AIProvider` interface that every provider adapter must implement, the prompt contracts they all share, and per-provider notes for Anthropic (MVP), Gemini, and Grok.

---

## 1. Goals

- Adding a new provider must require zero changes outside `lecture_processor/ai/providers/`.
- Provider-specific differences (auth, model names, rate limits, image input formats, streaming protocol) are absorbed inside the adapter. The pipeline never branches on provider.
- Prompts are versioned and live in their own files so they can be iterated on without code changes.
- Failures are typed so the pipeline can decide whether to retry, fall back, or surface a user-facing error.

## 2. Module Layout

```
src/lecture_processor/ai/
├── __init__.py
├── enrichment.py            # The orchestrator: takes a lecture artifact, calls a provider, writes back.
├── prompts/
│   ├── __init__.py
│   ├── v1/
│   │   ├── analyze_lecture.txt        # Title + summary + outline + slide_analysis prompt.
│   │   ├── discover_resources.txt     # Phase 3+. Resource discovery prompt.
│   │   └── system.txt                 # Shared system prompt.
│   └── registry.py                    # load_prompt("v1/analyze_lecture") -> str
├── providers/
│   ├── __init__.py
│   ├── base.py              # AIProvider ABC, request/response dataclasses, exception types.
│   ├── anthropic.py         # MVP.
│   ├── gemini.py            # Phase 3.
│   ├── grok.py              # Phase 3.
│   └── registry.py          # get_provider(name) -> AIProvider
└── tokens.py                # Token counting / chunking utilities.
```

## 3. The Interface

```python
# providers/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional


class ProviderCapability(str, Enum):
    TEXT = "text"
    IMAGE_INPUT = "image_input"
    STREAMING = "streaming"
    LONG_CONTEXT_200K = "long_context_200k"


@dataclass(frozen=True)
class ProviderInfo:
    name: str                    # "anthropic" | "gemini" | "grok"
    display_name: str            # "Anthropic Claude"
    available_models: list[str]
    default_model: str
    capabilities: set[ProviderCapability]
    docs_url: str


@dataclass(frozen=True)
class AnalyzeLectureRequest:
    """Input bundle for one lecture-analysis call."""
    lecture_id: str
    transcript_text: str
    segments: list[dict]                # [{id, start, end, text}, ...]
    slides: list[dict]                  # [{id, timestamp_seconds, linked_segment_ids, image_path}, ...]
    include_slide_images: bool
    target_token_budget: int            # informational; provider may chunk
    prompt_version: str = "v1"


@dataclass(frozen=True)
class AnalyzeLectureResponse:
    """Validated, schema-conformant analysis output."""
    title: str
    executive_summary: str
    outline: list[dict]                 # [{id, heading, slide_ids}, ...]
    slide_analysis: list[dict]          # see schema/enrichment-v1.0.0.json
    input_token_estimate: int
    output_token_estimate: int
    raw_response_id: Optional[str]      # Provider request ID, for support/debug
    warnings: list[str]


class AIProviderError(Exception):
    """Base class. Subclasses tell the orchestrator what to do next."""

class ProviderAuthError(AIProviderError):
    """API key invalid or expired. Surface to user; do not retry."""

class ProviderRateLimitError(AIProviderError):
    """Retry with backoff. retry_after_seconds may be set."""
    def __init__(self, msg: str, retry_after_seconds: Optional[float] = None):
        super().__init__(msg)
        self.retry_after_seconds = retry_after_seconds

class ProviderTransientError(AIProviderError):
    """Network blip, 5xx, timeout. Retry with backoff."""

class ProviderRequestError(AIProviderError):
    """4xx that isn't auth/rate. Bug in our request construction. Do not retry."""

class ProviderResponseError(AIProviderError):
    """Provider returned 2xx but the body did not match the schema we asked for."""


class AIProvider(ABC):
    @classmethod
    @abstractmethod
    def info(cls) -> ProviderInfo:
        """Static metadata. No API call."""

    @abstractmethod
    def __init__(self, api_key: str, model: str, *, http_client=None) -> None: ...

    @abstractmethod
    def test_connection(self) -> None:
        """
        Cheap call that validates the API key and model.
        Raises ProviderAuthError on bad key, ProviderRequestError on bad model.
        Returns None on success.
        """

    @abstractmethod
    def analyze_lecture(
        self,
        request: AnalyzeLectureRequest,
        *,
        cancel_check: Optional[callable] = None,
        progress_callback: Optional[callable] = None,
    ) -> AnalyzeLectureResponse:
        """
        Make the lecture-analysis call. Block until the response is ready.

        - cancel_check: called every few seconds. If it returns True, raise
          ProviderTransientError("cancelled") and stop. Implementations MUST
          attempt to call it during streaming.
        - progress_callback: called with float in [0.0, 1.0] when a meaningful
          milestone is reached (request constructed, first byte received,
          streaming chunk received, parsing complete). Implementations SHOULD
          call it but a single 0.0 / 1.0 pair is acceptable for non-streaming.
        """
```

## 4. Why Synchronous

The interface is synchronous. The pipeline already runs each lecture's enrichment on a worker thread — adding async would force an asyncio event loop into the existing `ThreadPoolExecutor` model and gain nothing.

If a provider's official SDK is async-only, the adapter wraps it with `asyncio.run()` internally. This is fine for our concurrency model (one in-flight call per lecture, single-digit concurrent lectures).

## 5. Cancellation Contract

Every adapter MUST honor `cancel_check`. The orchestrator passes a callable that returns `True` once the user clicks Cancel. Implementations:

- Streaming providers: call `cancel_check()` between chunks and abort the stream.
- Non-streaming providers: call `cancel_check()` before sending the request and after receiving the response. Mid-flight cancellation isn't possible without async, and we accept that limitation.
- On cancel, raise `ProviderTransientError("cancelled")`. The orchestrator uses the message string to distinguish cancel from real transient failures.

## 6. Prompt Contract

All providers receive the same prompt text. Providers differ only in how they wrap it in their respective SDKs (Anthropic uses `messages`, Gemini uses `contents`, Grok uses an OpenAI-compatible `messages` schema).

The prompt MUST instruct the model to return strict JSON. Each adapter is responsible for whatever JSON-mode flag its API offers (Anthropic's tool-use or "respond in JSON" instruction, Gemini's `response_mime_type="application/json"`, Grok's `response_format={"type": "json_object"}`).

The expected JSON shape is identical across providers and matches the `enrichment` portion of the lecture artifact. The adapter validates the response against `enrichment-v1.0.0.json` before returning. If validation fails, raise `ProviderResponseError` — do not silently coerce.

### 6.1 Prompt Files Live On Disk

```
ai/prompts/v1/analyze_lecture.txt       ← Hand-edited prompt template.
ai/prompts/v1/system.txt                ← Shared system instructions.
```

The template uses Python `string.Template` syntax (`$variable`). No Jinja, no f-strings — keep substitution dumb so prompts are easy to read and review without running the code.

Versioned subfolders (`v1/`, `v2/`, ...) let you ship a prompt iteration without breaking artifacts produced by older prompts. The artifact records `prompt_version` so reproductions work.

### 6.2 Prompt Inputs

The orchestrator constructs the prompt context:

```python
{
    "lecture_id": str,
    "transcript_full": str,                   # All transcript text.
    "segments_table": str,                    # Pre-formatted: "[id] start-end: text"
    "slides_table": str,                      # Pre-formatted: "[id] @MM:SS  segments: 12,13,14"
    "slide_count": int,
    "duration_minutes": float,
}
```

Prompts use these placeholders and nothing else. If a prompt iteration needs new context, add it explicitly here and bump the prompt version.

## 7. Token Budget & Chunking

A 60-minute lecture transcript averages 7,000–9,000 words ≈ 10,000–13,000 tokens. Plus slide images (if included), system prompt, schema instructions: realistic budget is ~25,000 tokens input.

All three providers in scope handle 100k+ context. **No chunking is required for MVP.** The interface accepts a `target_token_budget` field that the adapter MAY use to decide whether to chunk internally; for Phase 1 every adapter ignores it and sends the full transcript in one call.

Chunking will become necessary if the product expands to multi-hour content (graduate seminars, conference talks). At that point, chunking lives inside the adapter, not the orchestrator. The orchestrator's job is "hand the provider one lecture's worth of data and get one structured response back."

## 8. Image Input Policy

Slide images are **not sent by default in Phase 1**. The first prompt operates on transcript text and slide timestamps only. This:

- Cuts token cost by ~80% for image-heavy lectures.
- Eliminates a class of provider-specific image format bugs.
- Produces useful enrichment for typical lecture content (the transcript usually describes what's on the slide).

Image input is reserved for a future "deep analysis" mode (not in MVP). When added, the adapter will:

- Receive downsized JPEGs (max 1280px on the long edge, quality 75).
- Send them via the provider's native image input format (Anthropic content blocks, Gemini inline data, Grok image URL or base64).
- Document the per-image token cost in the provider's `info()` capabilities so users can predict spend.

## 9. Provider Notes

### 9.1 Anthropic Claude (MVP)

- **SDK:** `anthropic` Python package.
- **Auth:** `Authorization: Bearer ${ANTHROPIC_API_KEY}` (handled by SDK).
- **Default model:** `claude-opus-4-7` for best quality on long-form pedagogical content; offer `claude-sonnet-4-6` and `claude-haiku-4-5` in the model picker for speed/cost tradeoffs.
- **Streaming:** Use the streaming API. The orchestrator gets progress callbacks.
- **JSON output:** Use the "respond in JSON" prompt instruction and the `tool_use` mechanism for structured output. Tool use forces a schema-conformant response and is more reliable than asking for JSON in prose.
- **Test connection:** A 1-token completion against the configured model. Cost is effectively zero.
- **Known gotcha:** Anthropic rejects requests where the system prompt contains untrusted instructions. Our system prompt is fully under our control, so this isn't an issue, but never inject user-controlled content into the system message.

### 9.2 Google Gemini (Phase 3)

- **SDK:** `google-generativeai` Python package.
- **Auth:** API key in URL or header.
- **Default model:** `gemini-1.5-pro-latest` or successor.
- **JSON output:** Set `response_mime_type="application/json"` and `response_schema={...}`. This is more reliable than Anthropic's free-form JSON and lets us drop the schema-mismatch retry loop.
- **Streaming:** Supported.
- **Known gotcha:** Aggressive content filtering can refuse responses on lecture content discussing sensitive topics (medical, political). Catch the safety-block error type and surface a clear message.

### 9.3 xAI Grok (Phase 3)

- **SDK:** OpenAI-compatible Python package, base URL `https://api.x.ai/v1`.
- **Auth:** Bearer token.
- **Default model:** Whatever Grok's current best long-context model is at implementation time (verify via API at adapter init).
- **JSON output:** `response_format={"type": "json_object"}`.
- **Streaming:** Supported via OpenAI-compatible SSE.
- **Known gotcha:** Less mature than Anthropic/Gemini. Treat all unfamiliar errors as `ProviderTransientError` and retry; only treat documented 4xx codes as terminal.

## 10. Registry & Discovery

```python
# providers/registry.py

from .anthropic import AnthropicProvider
from .gemini import GeminiProvider
from .grok import GrokProvider

_REGISTRY = {
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "grok": GrokProvider,
}

def get_provider(name: str) -> type[AIProvider]:
    try:
        return _REGISTRY[name.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown provider: {name!r}. Available: {sorted(_REGISTRY)}"
        )

def all_providers() -> list[ProviderInfo]:
    return [cls.info() for cls in _REGISTRY.values()]
```

The frontend calls a Tauri command `list_providers` which calls `all_providers()` and returns the metadata. The UI renders the provider picker from this list — there's no hardcoded provider list in the frontend. Adding Gemini and Grok to the registry is sufficient to make them appear in the UI.

## 11. Error Handling Matrix

| Exception                  | Orchestrator action                        | User sees                              |
|----------------------------|--------------------------------------------|----------------------------------------|
| `ProviderAuthError`        | Stop. Mark all remaining lectures as failed-auth. | "API key for {provider} is invalid."   |
| `ProviderRateLimitError`   | Sleep `retry_after`, retry up to 3 times. After 3 failures, mark this lecture failed and continue with the next. | "Rate limited. Retrying…" then "Some lectures couldn't be enriched due to rate limits." |
| `ProviderTransientError`   | Exponential backoff (1s, 4s, 16s). 3 retries. After 3 failures, mark this lecture failed and continue. | "Network problem. Retrying…" |
| `ProviderRequestError`     | Stop. This is our bug. Log full request. | "An unexpected error occurred. See processing log." |
| `ProviderResponseError`    | Retry once with the same request, then mark failed. | "Couldn't parse the AI response for {lecture}." |
| `Exception` (unhandled)    | Stop. Log full traceback. | Generic crash dialog. |

Auth errors abort the whole batch because retrying with the same key won't help and we shouldn't waste the user's time on N lectures.

Other errors fail one lecture and proceed. The user gets partial enrichment, which is more useful than zero enrichment.

## 12. Testing Hooks

The base class includes an `http_client` injection seam. The default is the provider's official SDK; tests pass a mock that returns canned responses. This is how the contract tests in document 09 work without hitting real APIs.

A `MockProvider` class in `providers/mock.py` provides deterministic responses for development and CI. It's registered under the name `mock` and can be selected from the CLI via `--ai-provider mock`. It never makes network calls. This is the provider that integration tests use.
