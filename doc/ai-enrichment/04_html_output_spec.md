# 04 — HTML Output Specification

## 1. Goals

- Open the file, learn from it. The HTML is a study tool, not a marketing page.
- Works fully offline. No CDN, no external fonts, no analytics, no tracking pixels.
- Accessible: semantic HTML5, heading hierarchy, alt text on every image, sufficient contrast, keyboard-navigable.
- Portable: copy the lecture folder to USB / cloud drive / phone, open in any browser, it works.
- Trivial to render: pure templating, no client-side framework. JS is for progressive enhancement only.

## 2. Output Layout

For each lecture:

```
Lecture_Name/
├── lecture.json
├── html/
│   ├── index.html              ← The study page for this lecture.
│   ├── assets/
│   │   ├── lecture.css
│   │   ├── lecture.js          ← Optional: collapsible sections, no required JS.
│   │   ├── favicon.svg
│   │   └── slide_0001_thumb.jpg ← 1280px max edge, JPEG q75, ~50–150 KB each.
│   └── data/
│       └── lecture.json        ← Symlink or copy. Lets the page link to raw data.
└── slides/
    └── slide_0001_*.png        ← Originals stay in their existing location.
```

Per batch:

```
OutputFolder/
├── batch.json
├── index.html                  ← Top-level batch index.
├── assets/
│   ├── batch.css
│   └── batch.js
└── Lecture_Name/...
```

### 2.1 Why Thumbnails

Slide PNGs are typically 1920×1080 PNGs at 200–600 KB each. A 60-slide lecture is 25–50 MB of slides. Loading those inline crushes the rendering. Thumbnails at 1280px JPEG q75 are ~50–150 KB and visually indistinguishable in the layout. The HTML links the thumbnail with `<a href="../slides/slide_0001_*.png">` so clicking opens the original.

The thumbnail generation step runs as part of the HTML render stage, not during slide extraction. This keeps the local pipeline unchanged and means re-rendering HTML can produce new thumbnails (e.g., if we change quality settings) without re-extracting slides.

## 3. Lecture Page Structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ enrichment.title }} · Lecture Notes</title>
  <link rel="icon" href="assets/favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="assets/lecture.css">
</head>
<body>
  <header class="page-header">
    <nav><a href="../index.html">← Back to batch</a></nav>
    <h1>{{ enrichment.title }}</h1>
    <p class="meta">
      <span>{{ media.duration_minutes }} min</span> ·
      <span>{{ slide_count }} slides</span> ·
      <span>{{ word_count }} words</span> ·
      <span>{{ enrichment.provider }} {{ enrichment.model }}</span>
    </p>
  </header>

  <main>
    <section id="summary" aria-labelledby="summary-heading">
      <h2 id="summary-heading">Summary</h2>
      <p>{{ enrichment.executive_summary }}</p>
    </section>

    <section id="outline" aria-labelledby="outline-heading">
      <h2 id="outline-heading">Outline</h2>
      <ol>
        {% for item in enrichment.outline %}
          <li><a href="#slide-{{ item.slide_ids[0] }}">{{ item.heading }}</a></li>
        {% endfor %}
      </ol>
    </section>

    <section id="slides" aria-labelledby="slides-heading">
      <h2 id="slides-heading">Slides &amp; commentary</h2>
      {% for slide in slides %}
        <article id="slide-{{ slide.id }}" class="slide">
          <a href="../{{ slide.relative_path }}" class="slide-image-link">
            <img src="assets/slide_{{ '%04d' % slide.id }}_thumb.jpg"
                 alt="{{ slide.analysis.summary }}"
                 loading="lazy"
                 width="1280">
          </a>
          <div class="slide-body">
            <h3>{{ slide.analysis.descriptive_filename or slide.filename }}</h3>
            <p class="slide-time">{{ slide.timestamp_human }}</p>
            <p class="slide-summary">{{ slide.analysis.summary }}</p>
            <div class="slide-commentary">
              <h4>Instructor commentary</h4>
              <p>{{ slide.analysis.instructor_commentary }}</p>
            </div>
            <div class="slide-tags">
              {% for tag in slide.analysis.tags %}
                <span class="tag">{{ tag }}</span>
              {% endfor %}
            </div>
          </div>
        </article>
      {% endfor %}
    </section>

    <section id="transcript" aria-labelledby="transcript-heading">
      <h2 id="transcript-heading">Full transcript</h2>
      <details>
        <summary>Show transcript ({{ word_count }} words)</summary>
        <div class="transcript-segments">
          {% for seg in transcript.segments %}
            <p data-start="{{ seg.start }}">
              <span class="t">{{ seg.start_human }}</span>
              {{ seg.text }}
            </p>
          {% endfor %}
        </div>
      </details>
    </section>

    {% if enrichment.resources %}
    <section id="resources" aria-labelledby="resources-heading">
      <h2 id="resources-heading">Further reading</h2>
      <ul>
        {% for r in enrichment.resources %}
          <li>
            <a href="{{ r.url }}" rel="noopener noreferrer">{{ r.title }}</a>
            <span class="badge">{{ r.source_quality }}</span>
            <p>{{ r.summary }}</p>
          </li>
        {% endfor %}
      </ul>
    </section>
    {% endif %}
  </main>

  <footer>
    <p>Generated {{ enrichment.finished_at }} by Lecture Processor.</p>
    <p><a href="data/lecture.json">Raw lecture data (JSON)</a></p>
    {% if warnings %}
      <details>
        <summary>{{ warnings|length }} warning(s)</summary>
        <ul>{% for w in warnings %}<li>{{ w }}</li>{% endfor %}</ul>
      </details>
    {% endif %}
  </footer>

  <script src="assets/lecture.js" defer></script>
</body>
</html>
```

## 4. Index Page Structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ batch_label }} · Lecture Index</title>
  <link rel="stylesheet" href="assets/batch.css">
</head>
<body>
  <header>
    <h1>{{ batch_label }}</h1>
    <p class="meta">
      {{ summary.completed }} of {{ summary.attempted }} lectures processed
      · {{ summary.enriched }} enriched
      · generated {{ finished_at_human }}
    </p>
  </header>

  <main>
    <ul class="lecture-grid">
      {% for lec in lectures %}
        <li class="lecture-card">
          <a href="{{ lec.lecture_id }}/html/index.html">
            {% if lec.thumbnail_relative_path %}
              <img src="{{ lec.thumbnail_relative_path }}"
                   alt="First slide of {{ lec.title }}"
                   loading="lazy">
            {% endif %}
            <h2>{{ lec.title or lec.source_filename }}</h2>
            <p class="duration">{{ lec.duration_minutes }} min · {{ lec.slide_count }} slides</p>
            <p class="summary">{{ lec.short_summary }}</p>
          </a>
        </li>
      {% endfor %}
    </ul>
  </main>

  <footer>
    <p>Generated by Lecture Processor.</p>
  </footer>
</body>
</html>
```

The thumbnail on each card is the first slide's thumbnail. If enrichment hasn't run, fall back to `source_filename` and omit summary/title sections.

## 5. CSS Approach

Single hand-written `lecture.css` (and `batch.css`). No build step, no preprocessor. Critical decisions:

- **Type scale:** system font stack (`-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif`). Reading body 18px / 1.6 line-height. Generous left margin on prose blocks (max 70ch) for readability.
- **Color:** Light mode default. Dark mode via `prefers-color-scheme: dark` media query. Both palettes use AAA contrast for body text.
- **Layout:** CSS Grid for the slide cards, flexbox for headers. No fixed-width breakpoints; everything reflows.
- **Print:** A `@media print` block that hides nav, expands all `<details>`, and prints slides at full width. Useful for PDF export via browser print.

Total CSS budget: under 8 KB minified. Easy to hit by avoiding utility framework bloat.

## 6. JS Approach

Optional. The HTML works without JS. JS only adds:

- `<details>` polyfill on the rare browser that doesn't support it (Safari < 6 — basically nothing today; consider dropping).
- A "back to top" floating button on long pages.
- A timestamp click handler in transcript view that scrolls to the corresponding slide (uses `data-start` already in the markup).

JS budget: under 4 KB. No framework, no jQuery, no bundler.

## 7. Templating

Use **Jinja2** (already a transitive dep of pip-installed packages; cheap to add explicitly).

Templates live at:

```
src/lecture_processor/html/templates/
├── lecture.html.j2
├── batch_index.html.j2
└── partials/
    ├── slide_card.html.j2
    └── footer.html.j2
```

Static assets at:

```
src/lecture_processor/html/static/
├── lecture.css
├── batch.css
├── lecture.js
├── batch.js
└── favicon.svg
```

The renderer copies the static files into each batch's `assets/` directories. Don't symlink — symlinks break on Windows and on USB drives formatted as exFAT.

## 8. Renderer Module

```
src/lecture_processor/html/
├── __init__.py
├── renderer.py        # render_lecture(lecture: dict, dest: Path), render_batch(batch: dict, dest: Path)
├── thumbnails.py      # generate_thumbnail(src: Path, dest: Path, max_edge: int = 1280)
├── helpers.py         # format_timestamp_human, format_duration, etc.
├── templates/...
└── static/...
```

The renderer is invoked by the AI enrichment orchestrator at the end of each lecture's enrichment, and once at the end of the batch for the index. It can also be invoked standalone via `lecture-processor render <output-dir>` for re-rendering after template changes.

The renderer reads `lecture.json` and `batch.json` only. It does not read transcript.txt or scan the slides directory. This is the contract from document 01.

## 9. Accessibility Checklist

- Every `<img>` has a non-empty `alt` attribute. For decorative images (none currently), use `alt=""`.
- Slide images use the AI-generated `summary` as `alt` text. Falls back to "Slide N from {lecture title}" when no summary exists.
- Heading hierarchy is strictly nested (h1 → h2 → h3). No skipping levels.
- Color is never the sole conveyor of information. Tags use both color and text.
- Focus indicators are preserved (no `outline: none` without a replacement).
- All interactive elements are reachable and operable by keyboard.
- The `lang` attribute is set on `<html>`. Phase 3+ will derive this from `transcript.language`; MVP hardcodes `lang="en"` and documents that non-English lectures are out of scope.

## 10. Print / PDF Export

We do not ship a "PDF export" button in MVP. Users can:

1. Open the lecture HTML.
2. File → Print → Save as PDF.

The print stylesheet ensures this produces a usable document. This is the simplest possible answer that meets the "PDF export" wishlist item from the PRD without committing to a chrome-driving headless renderer in the codebase.

A real PDF generator (WeasyPrint, Playwright, etc.) is a Phase 4+ consideration if users ask for it.

## 11. Versioning

The HTML output carries a `<meta name="generator" content="lecture-processor 0.2.0">` tag and the same in the footer. When the templating system changes meaningfully, bump the generator version. This isn't user-facing, but it makes "what version produced this HTML?" a one-grep question for support.

Templates themselves are not versioned in the artifact — they're not part of the schema. If a user re-renders an old batch with a new template, the HTML changes and the JSON doesn't. That's the right behavior.
