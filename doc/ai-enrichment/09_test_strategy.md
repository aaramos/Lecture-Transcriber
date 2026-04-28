# 09 — Test Strategy

## 1. What Currently Exists

The repo has 8 unittest files under `tests/`: `test_cli.py`, `test_config.py`, `test_media.py`, `test_pipeline.py`, `test_temp_cleanup.py`, `test_timecode.py`, `test_transcription.py`, `test_writers.py`. Coverage is real but Python-only — there are no Rust tests and no UI tests.

Don't break what's there. Every existing test must pass throughout the implementation.

## 2. New Test Layers Required

| Layer | Where | What it covers |
|-------|-------|----------------|
| Schema fixtures | `tests/test_artifacts.py` | `lecture.json`, `batch.json`, `enrichment` validate against the JSON Schemas |
| Provider contract | `tests/test_provider_contract.py` | All providers obey the `AIProvider` ABC and produce schema-conformant responses |
| Mock provider integration | `tests/test_enrichment_e2e.py` | End-to-end enrich flow with the mock provider, including retry, cancel, partial failure |
| Slide-segment linker | `tests/test_linker.py` | Edge cases for the linker function |
| HTML rendering | `tests/test_html_renderer.py` | Golden-file tests for index and lecture pages |
| Keychain | `src-tauri/src/keychain_tests.rs` | Round-trip set/get/delete, key never logged |
| Logging | `tests/test_logging.py` | ProcessingLog atomic writes, multi-stage append |
| File discovery | `tests/test_pipeline.py` (extended) | Mixed-extension folders, alpha sort |

## 3. Tests Required Before Implementation Starts

These are the tests engineers write **first**, against fixtures that don't have implementations yet. The point is to have the contract codified before the code is written.

### 3.1 Schema validation tests

```python
# tests/test_artifacts.py
import json, jsonschema, pathlib

SCHEMA_DIR = pathlib.Path("src/lecture_processor/schemas")
FIXTURE_DIR = pathlib.Path("tests/fixtures/artifacts")

def _load_schema(name):
    return json.loads((SCHEMA_DIR / name).read_text())

def _load_fixture(name):
    return json.loads((FIXTURE_DIR / name).read_text())

def test_minimal_lecture_validates():
    jsonschema.validate(_load_fixture("lecture_minimal.json"), _load_schema("lecture-v1.0.0.json"))

def test_fully_enriched_lecture_validates():
    jsonschema.validate(_load_fixture("lecture_enriched.json"), _load_schema("lecture-v1.0.0.json"))

def test_lecture_missing_required_field_fails():
    artifact = _load_fixture("lecture_minimal.json")
    del artifact["transcript"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(artifact, _load_schema("lecture-v1.0.0.json"))

def test_batch_validates():
    jsonschema.validate(_load_fixture("batch.json"), _load_schema("batch-v1.0.0.json"))

def test_unknown_schema_version_rejected():
    artifact = _load_fixture("lecture_minimal.json")
    artifact["schema_version"] = "2.0.0"
    # Reader-level test, not schema-level (schema permits any string).
    from lecture_processor.artifacts import LectureArtifactWriter
    with pytest.raises(SchemaVersionError):
        LectureArtifactWriter._validate_for_read(artifact)
```

The fixtures `tests/fixtures/artifacts/*.json` are **the ground truth.** Any change to them is a contract change requiring review.

### 3.2 Provider contract tests

Every provider implementation must pass the same suite. The suite operates on a registered provider class:

```python
# tests/test_provider_contract.py
import pytest
from lecture_processor.ai.providers.registry import _REGISTRY

@pytest.fixture(params=list(_REGISTRY.keys()))
def provider_class(request):
    return _REGISTRY[request.param]

def test_info_has_required_fields(provider_class):
    info = provider_class.info()
    assert info.name
    assert info.display_name
    assert info.available_models
    assert info.default_model in info.available_models
    assert info.docs_url.startswith("http")

def test_provider_validates_schema(provider_class, mock_http_client_returning_canned):
    # Each provider has a fixture file: tests/fixtures/responses/<name>_canned.json
    p = provider_class(api_key="test", model=provider_class.info().default_model,
                       http_client=mock_http_client_returning_canned)
    response = p.analyze_lecture(_minimal_request())
    # The response is validated by the adapter before return.
    # Schema-mismatched canned responses must trigger ProviderResponseError.

def test_auth_error_raised_for_401(provider_class, mock_http_client):
    mock_http_client.set_response(status=401)
    p = provider_class(api_key="bad", model="...", http_client=mock_http_client)
    with pytest.raises(ProviderAuthError):
        p.test_connection()

def test_rate_limit_raises_with_retry_after(provider_class, mock_http_client):
    mock_http_client.set_response(status=429, headers={"retry-after": "30"})
    p = provider_class(api_key="...", model="...", http_client=mock_http_client)
    with pytest.raises(ProviderRateLimitError) as ei:
        p.analyze_lecture(_minimal_request())
    assert ei.value.retry_after_seconds == 30.0

def test_cancel_check_honored(provider_class, mock_http_client_with_streaming):
    p = provider_class(api_key="...", model="...", http_client=mock_http_client_with_streaming)
    cancelled = [False]
    def cancel_check():
        cancelled[0] = True
        return True
    with pytest.raises(ProviderTransientError, match="cancelled"):
        p.analyze_lecture(_minimal_request(), cancel_check=cancel_check)
    assert cancelled[0]
```

This is the contract. Adding a fourth provider in Phase 4 is just "make this suite pass with your adapter registered."

### 3.3 Slide-segment linker tests

Tests for `link_slides_to_segments`:

- Empty slides, non-empty segments → returns empty list.
- Empty segments, non-empty slides → each slide has `linked_segment_ids: []`.
- Single slide owns all segments.
- Two slides; segments split at the boundary.
- Segment that spans a slide boundary attaches to the slide it started in.
- Segment timestamp exactly at a slide boundary attaches to the new slide.
- Segments and slides not pre-sorted by time → linker handles it.

### 3.4 HTML golden-file tests

```python
# tests/test_html_renderer.py
def test_lecture_html_matches_golden(tmp_path):
    artifact = json.loads(FIXTURE_DIR.joinpath("lecture_enriched.json").read_text())
    rendered = render_lecture_to_string(artifact, asset_root="assets/")
    expected = (FIXTURE_DIR / "expected/lecture_enriched.html").read_text()
    # Normalize whitespace before compare.
    assert _normalize(rendered) == _normalize(expected)
```

Golden files are reviewed in PR. A renderer change that affects HTML must update the golden file in the same PR. This makes the diff visible to reviewers.

Also assert:

- `<img>` elements all have non-empty `alt`.
- `<a>` links to slide files use relative paths that exist on disk (mock the filesystem to verify).
- Heading hierarchy never skips levels (parse with html.parser, walk h1→h2→h3).
- No external URLs in `<script src>` or `<link href>` (offline-friendly).

## 4. Tests That Run Against Real APIs

A small set of **manual** tests, gated by env var, hit real providers. These don't run in CI.

```python
# tests/manual/test_real_anthropic.py
import os, pytest

@pytest.mark.skipif(
    not os.environ.get("RUN_REAL_API_TESTS"),
    reason="Set RUN_REAL_API_TESTS=1 to run"
)
def test_anthropic_real_short_lecture():
    """Sanity check against real API. ~$0.05 per run. Run before each release."""
    ...
```

Runbook: a checklist in `doc/release-checklist.md` that says "run `RUN_REAL_API_TESTS=1 pytest tests/manual/` before tagging a release." Don't try to gate CI on real API calls — flaky and expensive.

## 5. Rust Tests

The Tauri Rust layer currently has no tests. Add for AI work:

```rust
// src-tauri/src/keychain_tests.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_set_get_delete() {
        let store = MacOsKeychainStore::test_keychain();
        let service = "com.lecture-processor.test.foo";
        store.set(service, "secret-value").unwrap();
        assert_eq!(store.get(service).unwrap(), Some("secret-value".to_string()));
        assert!(store.delete(service).unwrap());
        assert_eq!(store.get(service).unwrap(), None);
        assert!(!store.delete(service).unwrap());
    }

    #[test]
    fn overwriting_a_key_updates_it() { ... }

    #[test]
    fn key_never_appears_in_debug_output() {
        let key = "supersecret-12345-abcdef";
        let store = MacOsKeychainStore::test_keychain();
        store.set("svc", key).unwrap();
        let debug = format!("{:?}", store);
        assert!(!debug.contains(key), "Debug impl leaked the key");
    }
}
```

Also: a Rust test for `scan_folder` returning mixed extensions in alphabetical order (refactor R3).

## 6. UI Tests

The repo notes Playwright was used for manual UI checks. Don't expand to full UI test automation in this scope — the cost-to-coverage ratio is bad for vanilla JS apps with no framework. Instead:

- A short Playwright smoke test that walks: enable AI → enter mock API key → run mock batch → verify HTML opens. One test, ~50 lines. Green or red, no analysis.
- Manual test checklist in `doc/manual-test-checklist.md` covering settings dialog, privacy disclosure, cancel button states, post-batch summary.

## 7. Coverage Targets

| Module | Target line coverage |
|--------|---------------------|
| `ai/providers/base.py` | 100% (it's mostly types) |
| `ai/providers/<provider>.py` | 90%+ |
| `ai/enrichment.py` | 95%+ |
| `ai/linker.py` | 100% |
| `artifacts.py` | 95%+ |
| `html/renderer.py` | 90%+ |
| Existing modules | unchanged from today |

Coverage is necessary but not sufficient. A 95%-covered module with weak assertions still ships bugs. Code review is the second gate.

## 8. CI Setup

`pytest -q` runs the Python suite. `cargo test` runs the Rust suite. Both run on every PR. Schema validation runs as part of the Python suite via the artifacts tests.

The Playwright smoke test runs on PR if frontend or backend code changed (file path filter). It runs against a built debug binary.

Real API tests **never** run in CI.

## 9. Test Data

`tests/fixtures/` gets two new directories:

```
tests/fixtures/
├── artifacts/                    # JSON fixtures for schema and renderer tests
│   ├── lecture_minimal.json
│   ├── lecture_enriched.json
│   ├── lecture_enrichment_failed.json   # has enrichment: null
│   └── batch.json
├── responses/                    # Canned provider responses for contract tests
│   ├── anthropic_canned.json
│   ├── gemini_canned.json
│   ├── grok_canned.json
│   └── invalid_schema_canned.json
├── transcripts/                  # Sample transcript text (not committed if large)
└── expected/                     # Golden HTML output
    ├── lecture_enriched.html
    └── batch_index.html
```

Keep fixtures small. A "lecture" fixture has 5 slides and 20 segments — enough to exercise edge cases, not so much that file diffs are unreadable.

## 10. What's Not Tested (And Why That's OK)

- **The exact wording of AI-generated content.** It's non-deterministic. Test that it's present, schema-valid, and the right shape. Don't assert "the summary contains the word 'Picasso'."
- **Real network behavior.** Covered by manual tests (§4).
- **Tkinter `gui.py`.** It's not on the ship path.
- **Visual rendering of HTML.** Manual smoke test on three browsers per release.
- **Cross-platform Windows behavior.** Out of scope for MVP. Tests get added when Windows support lands.

The principle: aim tests at contracts and at non-obvious behavior. Don't write tests that re-implement the function being tested.
