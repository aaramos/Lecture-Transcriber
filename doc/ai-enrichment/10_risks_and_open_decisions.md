# 10 — Risks & Open Decisions

## 1. Risks

Ranked by impact × likelihood. "Mitigation" is what's already designed for. "Trigger" is what to watch for that means the risk is materializing.

### 1.1 Prompt quality drift (high × high)

**Risk:** AI outputs are unpredictable. A prompt that produces excellent outlines for one course style produces mediocre output for another. Users blame the product.

**Mitigation:**
- Versioned prompts with explicit `prompt_version` recorded in artifacts.
- Schema-conformant outputs enforced by validation; bad outputs fail loud.
- A "regenerate" action per lecture (Phase 3) lets users re-run with a different model when one underperforms.
- Document expected prompt iteration cycles. Shipping prompt v1 doesn't mean it's done.

**Trigger:** First real-user feedback. Budget time in Phase 2.5 for at least one prompt iteration before declaring MVP shipped.

### 1.2 Provider rate limits crater the user experience (medium × high)

**Risk:** A user processes 30 lectures and the 4th one hits a 429. Default retries don't recover. User sees "8 of 30 enriched" and is frustrated.

**Mitigation:**
- Sequential default (concurrency = 1) gives the provider's token bucket time to refill.
- 3-attempt retry with exponential backoff (1, 4, 16s).
- "Retry failed" action surfaces clearly in the UI.
- Per-lecture failure isolation — one rate-limited lecture doesn't poison the rest.

**Trigger:** Real-user batches >20 lectures. Test ahead with an Anthropic Tier 1 account to know the limits.

### 1.3 Schema evolution pain (medium × medium)

**Risk:** We ship schema 1.0.0, then find we need a structural change in 6 months. Existing users have on-disk artifacts that don't match the new code.

**Mitigation:**
- Migration system designed in document 01 §10.
- Strict semver: additive changes go in 1.x.0, breaking changes require a migration script.
- Reader checks `schema_version` and refuses unknown majors.

**Trigger:** Any non-additive change request. Treat as a breaking change and budget the migration effort.

### 1.4 Apple Silicon CoreML stack regression (low × medium)

**Risk:** The current whisper.cpp + CoreML setup is sensitive to model file pairings (the `.bin` and `.mlmodelc` must match versions). A future whisper.cpp update could break it.

**Mitigation:**
- Existing setup script (`scripts/setup-whisper-cpp-coreml.sh`) is preserved.
- The `--require-whisper-cpp-coreml` flag fails loud when CoreML isn't actually active.
- Local processing falls back to Metal/CPU automatically.

**Trigger:** Whisper transcription stage logs `COREML = 0` when `apple_silicon=True`. Existing log inspection catches this.

### 1.5 macOS Keychain prompt fatigue (low × low)

**Risk:** macOS Keychain prompts the user for permission to access keychain items in some scenarios (notably when a different code-signed binary tries to read). Users may see confusing prompts.

**Mitigation:**
- The Tauri app is consistently code-signed across releases (already part of the build).
- Items use the app's bundle ID as the access group, so the same signed app reads them silently.
- Document the expected first-run prompt in release notes.

**Trigger:** User support tickets about repeated keychain prompts. Investigate code signing if it appears.

### 1.6 HTML output drift across browsers (low × medium)

**Risk:** A CSS feature works in Safari but not in Chrome. Users open lecture pages on different machines and see broken layouts.

**Mitigation:**
- Plain semantic HTML5, no fancy CSS.
- Manual smoke test on Safari, Chrome, Firefox before each release.
- Print stylesheet is the most browser-divergent feature; test print preview specifically.

**Trigger:** First user report. Browsers are stable; this should rarely trigger.

### 1.7 Cancellation race during AI streaming (medium × low)

**Risk:** User cancels mid-stream. The provider SDK's stream object isn't cleanly cancellable from another thread. Network connection lingers, state machine confused.

**Mitigation:**
- Cooperative `cancel_check` between chunks.
- 5-second SIGTERM grace period before SIGKILL fallback (existing behavior in Rust).
- Worst case: the in-flight lecture is marked failed; nothing else corrupts.

**Trigger:** User reports of "cancel didn't work" or hung enrichment. Add SDK-specific abort calls if necessary.

### 1.8 API costs surprise users (medium × medium)

**Risk:** A user processes a 9-lecture module and gets a $20 bill from Anthropic with no warning.

**Mitigation:**
- Per-lecture token estimates shown in post-batch summary.
- "Estimate cost before running" is a Phase 3 feature; for MVP, document per-call costs in the privacy disclosure.
- Anthropic and most providers have usage caps users can set on their accounts.

**Trigger:** First user complaint. Add cost estimation in Phase 3 if it becomes a pattern.

---

## 2. Open Decisions

Decisions that defer to the implementation team or product owner.

### 2.1 Resource discovery: web search vs model knowledge?

**Question:** When generating "Further reading," should the model rely on its training data, or call out to a web search?

**Why open:** Model-only is cheap, fast, and offline; results are stale. Web search is current but adds another API dependency, another cost, and another failure mode.

**Recommendation:** MVP defers resource discovery entirely (deferred per document 00). When implemented in Phase 3, start with model-only and consider adding optional web search behind a separate setting.

**Decider:** Product owner. Defer until Phase 3 design.

### 2.2 Should batch.json record the API key fingerprint?

**Question:** Should we hash the API key and record `key_fingerprint: "sha256:abc..."` in batch.json, so users can verify which key produced an artifact?

**Why open:** Useful for users with multiple keys / multiple Anthropic projects. But it's information leakage of a sort — anyone with read access to the artifact knows which key it was. And the hash is reversible if the keyspace is small enough.

**Recommendation:** No. Record provider name and model only. If users need multi-key audit, that's the multi-profile feature in Phase 3+.

**Decider:** Architect. Provisionally decided NO.

### 2.3 What happens if `enrichment` exists but `local pipeline` info is incomplete?

**Question:** A user manually edits a `lecture.json`, leaves `enrichment` populated but corrupts `transcript.segments`. What do we do?

**Why open:** The schema would catch the corruption, but the question is whether to refuse to render at all, render with degraded sections, or render and warn.

**Recommendation:** Render with degraded sections. Show a banner at the top of the lecture HTML: "Some lecture data was incomplete; some sections may be missing." Log details. Don't refuse outright.

**Decider:** Architect. Provisionally decided RENDER WITH WARNING.

### 2.4 Where does the user choose `concurrent_ai_calls`?

**Question:** Settings dialog or main screen?

**Why open:** Frequency of change vs. discoverability. Users probably set it once.

**Recommendation:** Settings dialog. Default 1, max 3. Add a help tooltip explaining the rate-limit risk.

**Decider:** Designer / Product. Provisionally decided SETTINGS DIALOG.

### 2.5 Should a user be allowed to mix providers across lectures in one batch?

**Question:** Could lecture A be enriched by Claude, lecture B by Gemini in the same batch run?

**Why open:** Conceivable use case (different content suits different models), but it complicates the orchestrator and the cost-tracking story.

**Recommendation:** No for MVP. One provider per batch. A user who wants per-lecture provider choice can run two batches and copy the artifacts together (the schema supports this — `provider` is per-enrichment).

**Decider:** Product / architect. Provisionally decided ONE PROVIDER PER BATCH.

### 2.6 What gets logged at debug level vs info level?

**Question:** A debug mode that logs full prompts and responses is useful for support. But it could leak transcript content to wherever the log ends up.

**Why open:** Users need a way to file bug reports about enrichment quality. We need a way to triage them.

**Recommendation:**

- Default log level: stage timings, errors, warnings, token counts. No transcript text, no AI output.
- `--debug-prompts` flag (Phase 3): writes full prompts and responses to `<lecture>/ai/prompts/` (already a path slot in the artifact layout).
- The flag is documented as "for support; contains full lecture text."
- Off by default. Never enabled by Tauri-launched runs without an explicit Tauri command.

**Decider:** Architect. Provisionally decided AS ABOVE.

### 2.7 Should HTML rendering produce a single self-contained file per lecture?

**Question:** Embed thumbnails as base64 data URIs into one `.html` file, vs. the current spec of HTML + assets folder.

**Why open:** Single-file is more portable (email it, drag it anywhere). Multi-file is faster to load and easier to edit.

**Recommendation:** Multi-file by default (per current spec). Add a `--inline-assets` flag in Phase 3 if users ask for single-file. The two modes can coexist; they read the same `lecture.json`.

**Decider:** Architect. Provisionally decided MULTI-FILE.

### 2.8 What's the max lecture length the prompt can handle reliably?

**Question:** A 4-hour lecture is 30,000+ words. Will Anthropic Claude handle it well in one shot?

**Why open:** Token capacity is fine, but quality degrades on very long inputs (model attention).

**Recommendation:** Test with the longest realistic Kellogg/Northwestern lecture (~90 minutes, ~10,000 words). If quality holds, ship MVP without chunking. Add chunking in Phase 3 if needed for academic-conference-length content.

**Decider:** Engineer during Phase 2 implementation. Set a hard limit (e.g., 50,000 words) and reject longer lectures with a clear error message. Document the limit.

---

## 3. Decisions Already Made (Recap)

These are closed; reopening them requires a new design doc:

- ✅ AI enrichment is a **separate pipeline phase**, not interleaved.
- ✅ The **canonical artifact** is `lecture.json`. HTML and AI both read/write it; legacy text files are derived views.
- ✅ Providers behind a **single `AIProvider` ABC**; one ships in MVP (Anthropic).
- ✅ **macOS Keychain only** for MVP credential storage; trait abstraction supports Windows later.
- ✅ **HTML rendering is independent** of AI enrichment.
- ✅ **`.mov`, `.mp4`, `.m4v`** are the supported video formats from MVP onward (R3).
- ✅ **Original slide PNG filenames don't change.** AI descriptive names live in the artifact and HTML only (R4).
- ✅ **One provider per batch run** in MVP.
- ✅ **No telemetry, no usage metrics** from MVP.
- ✅ **No chunking in MVP**; full transcript fits in modern context windows for typical lecture lengths.
- ✅ **JSON Schema files committed**; validation is mandatory on read and write.

---

## 4. Final Note

The primary risk the architect flags is not technical — it's **scope creep during implementation.** The MVP boundary in document 00 is tight on purpose. Every "while we're at it, let's also add..." conversation should be redirected to a Phase 3+ ticket.

The product is more useful with a tight MVP shipped in 6 weeks than with a perfect product shipped in 16 weeks.
