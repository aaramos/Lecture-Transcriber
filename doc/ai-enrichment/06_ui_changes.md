# 06 — UI Changes Required

## 1. Scope Of Frontend Work

The current UI lives in `index.html`, `src/main.js` (~990 lines), `src/styles.css` (~520 lines). It's vanilla JS, no framework. Keep it that way — don't introduce React/Vue/Svelte for this scope.

New surfaces required for MVP:

1. AI enrichment toggle on the main batch screen.
2. Provider settings dialog (key entry, model picker, test connection).
3. Privacy disclosure dialog — shown before the first AI call ever.
4. Per-lecture enrichment progress in the active-video list.
5. Output options: HTML rendering toggle.
6. Post-batch summary updated to show enrichment counts and "Open HTML" button.

Deferred (post-MVP):

- Per-lecture reprocess UI (full screen with stage selector).
- Batch input format display when non-`.mov` files are supported.

## 2. Settings Dialog Additions

The existing settings dialog (audio quality, transcription engine, Whisper model, slide sensitivity, concurrent files, save normalized video) gains a new section: **AI Enrichment**.

```
┌─ AI Enrichment ──────────────────────────────────────────┐
│                                                          │
│  Provider     [Anthropic Claude    ▼]                    │
│  Model        [claude-opus-4-7      ▼]                   │
│                                                          │
│  API key      ●●●●●●●●●●●●●●●●●●●●●●  [Test connection] │
│                                            [Remove]      │
│                                                          │
│  ☑ Render HTML study materials after enrichment          │
│  ☐ Include slide images in AI analysis (slower, costs   │
│     more, only enable for visually-heavy lectures)       │
│                                                          │
│  Concurrent AI calls  [1 ▼]                              │
│                                                          │
│  Privacy: transcripts and slide thumbnails are sent      │
│  to your selected provider. The full video is never      │
│  uploaded. Read the full disclosure ›                    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

Behavior:

- **Provider** dropdown: populated from `list_providers` Tauri command. Disabling a provider means it's still in the list but disabled with "Coming soon" — that's how Gemini/Grok appear in MVP UI.
- **Model** dropdown: populated from the selected provider's `info().available_models`.
- **API key** input: masked. The placeholder shows the last 4 chars of the stored key when one exists ("●●●● 9q2v"). When empty, "Enter your API key".
- **Test connection** button: calls `test_provider` Tauri command. Shows a green check or red error for 3 seconds.
- **Remove** button: confirms, then calls `keychain_delete`.
- **Privacy disclosure ›**: opens the privacy dialog (next section).
- Settings saved to localStorage **excluding the API key** (key is in keychain only).

## 3. Privacy Disclosure Dialog

Shown:

- The first time a user enables AI enrichment for any provider.
- Whenever they switch providers.
- Always reachable from "Read the full disclosure ›" link in settings.

Content:

```
What gets sent when AI enrichment is on

  • The full transcript text (1,000–10,000 words per lecture).
  • Slide timestamps and slide order.
  • Slide thumbnails ONLY if "Include slide images" is enabled.
  • The lecture's filename.

What is NOT sent

  • The original video.
  • The normalized video.
  • The full-resolution slide PNGs (unless image analysis is on).
  • Your folder paths.
  • Any other lectures in the batch except the one being processed.

Where it goes

  Anthropic (https://api.anthropic.com)
  Subject to Anthropic's privacy policy and data retention.
  This app stores your API key locally in macOS Keychain.

[ Cancel ]              [ I understand — enable enrichment ]
```

Click "I understand" sets a per-provider flag in localStorage. Switching providers clears the flag. There's no "don't show this again" — it's only shown on first-use and on switch.

## 4. Main Screen Toggle

A new compact strip above the **Start** button:

```
☑ AI Enrichment with Anthropic Claude (claude-opus-4-7)         Configure ›
```

If no key is configured: the checkbox is disabled and the strip says "Configure AI enrichment to enable" with a Configure link that opens settings to the AI section.

If checked: enrichment runs after local processing. If unchecked: today's behavior.

## 5. Active Video List Changes

Today's active video list shows a per-file progress bar with the current step name. Add an enrichment phase:

```
Module 6_Video 6.2_AI-Human Pairing.mov
[████████████████████████░░░░] Enriching with Claude · 45%
```

The progress bar uses `enrich_lecture_progress.fraction`. The status label shows the current stage from the same event.

Enriched lectures show a small ✨ icon next to the completed status. HTML-rendered lectures show a 🔗 icon. Clicking either icon opens the relevant artifact (the lecture HTML page or the lecture folder).

## 6. Post-Batch Summary

Today's post-batch summary shows: completed / failed / skipped counts and an "Open output folder" button. Add:

```
Batch finished. 9 of 9 lectures processed.

  Local processing      9 ✓
  AI enrichment         8 ✓   1 failed (rate limit)
  HTML rendered         9 ✓

[ Open output folder ]    [ Open batch HTML ]
```

The "Open batch HTML" button is only present when HTML rendering is enabled and at least one lecture rendered successfully. It opens `<output>/index.html` in the user's default browser.

Failed enrichments are listed below the counts with the reason. A "Retry failed enrichments" action runs `enrich_batch` again with `--only-missing`.

## 7. Cancel Behavior

Today's Cancel button kills the local pipeline. It should also kill in-flight AI calls and stop the queue without cancelling already-completed enrichment.

UI states:

- Before any enrichment: "Cancel" stops the local pipeline (existing behavior).
- During AI enrichment: "Cancel" changes its label to "Stop enrichment". On click, finishes the in-flight call's lecture if it's near complete (>90% progress), or aborts immediately. The user sees a confirmation dialog explaining that completed enrichments are kept.
- During HTML rendering: rendering finishes for all enriched lectures. Cancel is disabled during rendering; it's fast (under a second per lecture).

## 8. Settings Persistence

Settings (excluding API keys) are stored in localStorage as today. New settings:

```js
{
  // ... existing settings ...
  aiProvider: "anthropic",          // null when not configured
  aiModel: "claude-opus-4-7",
  aiEnrichEnabled: false,           // the main-screen checkbox
  aiIncludeSlideImages: false,
  aiConcurrentCalls: 1,
  htmlRenderEnabled: true,
  privacyAcknowledgedProviders: ["anthropic"],
  // ...
}
```

API keys: never in localStorage. Keychain only.

## 9. Error Surfacing

| Error | Where shown | What user can do |
|-------|-------------|------------------|
| No API key for selected provider | Inline below provider picker | "Add a key" link |
| API key invalid (auth error) | Toast + settings dialog opens | Re-enter key |
| Rate limit exceeded after retries | Active video list status | Retry failed enrichments after a wait |
| Network down | Active video list status | Retry failed enrichments |
| Provider response unparseable | Active video list status | Open processing log |
| HTML render error | Per-lecture status with details | Open output folder |

Errors never disappear silently. Every failure is recorded in the lecture's `processing_log.txt` AND in the active video list (until cleared) AND in the batch summary.

## 10. Accessibility

- All interactive elements keyboard-reachable.
- Tab order: settings → main controls → start/cancel.
- Modal dialogs trap focus when open and restore focus on close.
- Status updates announced via `aria-live="polite"` regions for screen readers.
- Color is not the sole indicator of status (✓ / ✗ / ⚠ symbols accompany color).

The current UI is mostly already accessible. Don't regress it.

## 11. What Doesn't Change

- Folder picker, drag-and-drop, output suggestion: same.
- Recording speed picker, normalization confirmation: same.
- Existing whisper model picker, slide sensitivity, concurrent files: same.
- Active video list rendering pattern: same component, just a new progress phase.
- Settings dialog visual style: same; AI section is one new collapsible group.

Avoid the temptation to redesign. The existing UI works; the goal is to add new surfaces in a way that feels native to it.

## 12. Estimated Scope

Frontend changes for MVP:

- ~150 lines added to `main.js` (event handlers for new commands, new settings persistence).
- ~80 lines added to `index.html` (new dialogs, new strip, new active-list states).
- ~60 lines added to `styles.css`.
- New module: ~120 lines for the privacy dialog component (vanilla JS class).

Total frontend: under 500 lines net. Doable in a sprint.
