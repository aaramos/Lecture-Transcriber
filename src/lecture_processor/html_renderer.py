import html
import re
from pathlib import Path
from typing import Dict, List, Optional

from .artifacts import LECTURE_ARTIFACT_NAME, load_json
from .models import BatchSummary, FileResult, FileStatus
from .writers import write_text_atomic


def render_lecture_page(lecture_json_path: Path) -> Path:
    artifact = load_json(lecture_json_path)
    output_dir = lecture_json_path.parent
    html_dir = output_dir / "html"
    html_path = html_dir / "index.html"
    write_text_atomic(html_path, _lecture_html(artifact))
    return html_path


def render_batch_index(output_dir: Path, summary: BatchSummary) -> Path:
    html_path = output_dir / "index.html"
    write_text_atomic(html_path, _batch_html(output_dir, summary))
    return html_path


def _lecture_html(artifact: Dict) -> str:
    enrichment = artifact.get("enrichment") or {}
    title = enrichment.get("title") or _fallback_title(artifact)
    summary = enrichment.get("executive_summary") or _fallback_summary(artifact)
    slide_analysis = {_analysis_slide_id(item): item for item in enrichment.get("slide_analysis", [])}
    outline = enrichment.get("outline") or []
    resources = enrichment.get("resources") or []
    slides = artifact.get("slides") or []
    transcript = artifact.get("transcript", {})
    transcript_text = enrichment.get("formatted_transcript") or transcript.get("text") or ""
    body = [
        _html_head(title),
        "<body>",
        "<main>",
        _hero_block(title, summary, slides, outline, resources),
        _flow_block(outline, slides, slide_analysis),
        _transcript_block(transcript_text),
        _resources_block(resources),
        "</main>",
        "</body></html>",
    ]
    return "\n".join(body)


def _batch_html(output_dir: Path, summary: BatchSummary) -> str:
    rows = []
    for result in summary.results:
        artifact = _load_artifact_if_present(result.output_dir)
        title = (artifact.get("enrichment") or {}).get("title") if artifact else None
        title = title or result.title or result.source.stem.replace("_", " ")
        description = (artifact.get("enrichment") or {}).get("executive_summary") if artifact else None
        description = description or result.short_summary or result.message or result.status.value
        html_path = result.html_path or (result.output_dir / "html" / "index.html")
        link = _relative(html_path, output_dir) if html_path.exists() else None
        thumb = _first_slide(result.output_dir)
        rows.append(_batch_row(result, title, description, link, thumb, output_dir))

    title = f"{output_dir.name} Study Index"
    return "\n".join(
        [
            _html_head(title),
            "<body>",
            "<main>",
            f"<header><p class=\"eyebrow\">Batch Study Index</p><h1>{_e(output_dir.name)}</h1>"
            f"<p class=\"summary\">{summary.completed} completed, {summary.failed} failed, {summary.skipped} skipped, {summary.stopped} stopped.</p></header>",
            "<section class=\"lecture-list\">",
            "\n".join(rows) or "<p>No lectures were processed.</p>",
            "</section>",
            "</main>",
            "</body></html>",
        ]
    )


def _html_head(title: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_e(title)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #161716;
      --panel: #232522;
      --line: #42483f;
      --text: #f4f2e9;
      --muted: #b8b3a5;
      --accent: #e08658;
      --accent-soft: rgba(224, 134, 88, 0.16);
      --blue: #79a7c7;
      --blue-soft: rgba(121, 167, 199, 0.16);
      --good: #83bd8c;
      --warn: #d6b85a;
      --bad: #d9847a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.55;
    }}
    main {{
      max-width: 1320px;
      min-width: 0;
      margin: 0 auto;
      border-left: 1px solid var(--line);
      border-right: 1px solid var(--line);
    }}
    header, .hero-band, .content-band {{ border-bottom: 1px solid var(--line); padding: 34px; }}
    header {{ margin: 0; }}
    h1 {{ margin: 0; max-width: 980px; font-size: clamp(2.4rem, 7vw, 5.4rem); line-height: 1.02; letter-spacing: 0; }}
    h2 {{ margin: 0; font-size: 1.45rem; }}
    h3, h4, p {{ margin: 0; }}
    .eyebrow {{
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 0.75rem;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .summary {{ max-width: 920px; margin-top: 22px; color: #ded9ca; font-size: 1.08rem; line-height: 1.65; }}
    .hero-band {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(220px, 280px);
      gap: 28px;
      min-height: 380px;
      align-items: end;
    }}
    .metric-strip {{ display: grid; gap: 10px; }}
    .metric-strip div {{ border: 1px solid var(--line); border-radius: 8px; padding: 16px; background: var(--panel); }}
    .metric-strip strong, .metric-strip span {{ display: block; }}
    .metric-strip strong {{ font-size: 1.9rem; }}
    .metric-strip span {{ color: var(--muted); font-size: 0.76rem; font-weight: 800; text-transform: uppercase; }}
    .section-head {{ display: flex; align-items: end; justify-content: space-between; gap: 16px; margin-bottom: 18px; }}
    .flow-list {{ display: grid; gap: 12px; }}
    .flow-section {{ border: 1px solid var(--line); border-radius: 8px; background: var(--panel); overflow: hidden; }}
    .flow-section summary {{
      display: grid;
      grid-template-columns: auto 42px minmax(0, 1fr) auto;
      align-items: center;
      gap: 14px;
      padding: 14px;
      cursor: pointer;
      list-style: none;
    }}
    .flow-section summary::-webkit-details-marker {{ display: none; }}
    .flow-section summary::before {{ content: "\\25B8"; color: var(--muted); font-size: 1.05rem; font-weight: 900; }}
    .flow-section[open] summary {{ border-bottom: 1px solid var(--line); }}
    .flow-section[open] summary::before {{ content: "\\25BE"; color: var(--accent); }}
    .outline-number, .slide-id {{
      display: grid;
      place-items: center;
      border-radius: 999px;
      font-weight: 900;
    }}
    .outline-number {{ width: 34px; height: 34px; background: var(--accent-soft); color: var(--accent); }}
    .flow-title {{ min-width: 0; font-size: 0.95rem; font-weight: 900; overflow-wrap: anywhere; }}
    .flow-count {{ color: var(--muted); font-size: 0.75rem; font-weight: 900; text-transform: uppercase; }}
    .flow-slides {{ display: grid; gap: 14px; padding: 14px; }}
    .slide-row {{
      display: grid;
      grid-template-columns: minmax(220px, 320px) minmax(0, 1fr);
      gap: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: rgba(255, 255, 255, 0.02);
    }}
    .slide-media {{
      display: grid;
      min-height: 180px;
      place-items: center;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #111;
    }}
    .slide-media img {{ width: 100%; height: 100%; object-fit: contain; }}
    .missing-slide {{ color: var(--warn); font-size: 0.75rem; font-weight: 800; text-align: center; }}
    .slide-copy {{ display: grid; gap: 12px; align-content: start; }}
    .slide-title-row {{ display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 10px; align-items: center; }}
    .slide-id {{ width: 38px; height: 38px; background: var(--blue-soft); color: var(--blue); }}
    .slide-title-row h4 {{ overflow-wrap: anywhere; font-size: 1.05rem; }}
    .slide-summary, .commentary, .caption, .resource-card p, .transcript-text {{ color: #ded9ca; font-size: 0.93rem; line-height: 1.6; }}
    .caption {{ border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; background: rgba(255, 255, 255, 0.03); color: var(--muted); }}
    .caption strong {{ color: var(--text); }}
    .commentary {{ border-left: 3px solid var(--accent); padding-left: 12px; }}
    .tag-row {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .tag {{ border: 1px solid var(--line); border-radius: 999px; padding: 4px 8px; color: var(--muted); font-size: 0.72rem; font-weight: 800; }}
    .transcript-text {{ display: grid; max-width: 980px; gap: 1rem; font-size: 0.98rem; line-height: 1.72; }}
    .resource-list {{ display: grid; gap: 12px; }}
    .resource-card {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      align-items: start;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      background: var(--panel);
    }}
    .resource-card a {{ color: var(--text); font-weight: 900; text-decoration-color: var(--accent); text-underline-offset: 3px; }}
    .resource-card p {{ margin-top: 8px; }}
    .quality {{ border: 1px solid var(--line); border-radius: 999px; padding: 4px 9px; color: var(--muted); font-size: 0.72rem; font-weight: 900; text-transform: uppercase; }}
    .quality.high {{ border-color: rgba(131, 189, 140, 0.5); background: rgba(131, 189, 140, 0.12); color: var(--good); }}
    .empty-state {{ color: var(--muted); }}
    .lecture-list {{ padding: 18px 34px 34px; }}
    .lecture-row {{ display: grid; grid-template-columns: 132px minmax(0, 1fr) auto; gap: 16px; align-items: center; border-bottom: 1px solid var(--line); padding: 16px 0; }}
    .lecture-row img {{ width: 132px; aspect-ratio: 16 / 9; object-fit: cover; border: 1px solid var(--line); border-radius: 4px; background: var(--panel); }}
    .lecture-row.failed h2, .lecture-row.failed .status {{ color: var(--bad); }}
    .lecture-row.skipped h2, .lecture-row.skipped .status {{ color: var(--warn); }}
    .lecture-row.stopped h2, .lecture-row.stopped .status {{ color: var(--warn); }}
    .lecture-row h2 {{ margin: 0 0 4px; font-size: 1.1rem; }}
    .lecture-row p {{ margin: 0; color: var(--muted); }}
    .lecture-row a {{ color: var(--accent); font-weight: 800; text-decoration: none; }}
    @media (max-width: 980px) {{
      .hero-band {{ grid-template-columns: 1fr; min-height: auto; }}
      .metric-strip {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
      .flow-section summary {{ grid-template-columns: auto 42px minmax(0, 1fr); }}
      .flow-count {{ grid-column: 3; }}
      .slide-row, .resource-card, .lecture-row {{ grid-template-columns: 1fr; }}
      .lecture-row img {{ width: 100%; }}
    }}
  </style>
</head>"""


def _hero_block(title: str, summary: str, slides: List[Dict], outline: List[Dict], resources: List[Dict]) -> str:
    metrics = [
        (str(len(slides)), "Slides"),
        (str(len(outline)), "Sections"),
        (str(len(resources)), "Resources"),
    ]
    metric_html = "".join(
        f"<div><strong>{_e(value)}</strong><span>{_e(label)}</span></div>" for value, label in metrics
    )
    return f"""
      <section class="hero-band">
        <div>
          <p class="eyebrow">AI Study Notes</p>
          <h1>{_e(title)}</h1>
          <p class="summary">{_e(summary)}</p>
        </div>
        <div class="metric-strip" aria-label="Lecture metrics">{metric_html}</div>
      </section>
    """


def _flow_block(outline: List[Dict], slides: List[Dict], slide_analysis: Dict[int, Dict]) -> str:
    sections = _sections_with_slides(outline, slides, slide_analysis)
    if not sections:
        return """
          <section class="content-band">
            <div class="section-head"><div><p class="eyebrow">Outline</p><h2>Lecture Flow</h2></div></div>
            <p class="empty-state">No slides were extracted for this lecture.</p>
          </section>
        """

    rows = []
    for index, section in enumerate(sections):
        rows.append(
            f"""
            <details id="section-{_e(section["id"])}" class="flow-section" {"open" if index == 0 else ""}>
              <summary>
                <span class="outline-number">{_e(section["id"])}</span>
                <span class="flow-title">{_e(section["heading"])}</span>
                <span class="flow-count">{len(section["slides"])} slides</span>
              </summary>
              <div class="flow-slides">
                {''.join(_slide_card(slide, slide_analysis.get(_slide_id(slide), {})) for slide in section["slides"])}
              </div>
            </details>
            """
        )
    return f"""
      <section class="content-band">
        <div class="section-head"><div><p class="eyebrow">Outline</p><h2>Lecture Flow</h2></div></div>
        <div class="flow-list">{''.join(rows)}</div>
      </section>
    """


def _sections_with_slides(outline: List[Dict], slides: List[Dict], slide_analysis: Dict[int, Dict]) -> List[Dict]:
    if not slides and slide_analysis:
        slides = [{"id": slide_id} for slide_id in sorted(slide_analysis)]
    if not slides:
        return []

    slide_by_id = {_slide_id(slide): slide for slide in slides}
    if not outline:
        return [{"id": 1, "heading": "Lecture slides", "slides": slides}]

    sections = []
    used_ids = set()
    for index, item in enumerate(outline, start=1):
        section_slides = []
        for numeric_id in _outline_slide_ids(item):
            if numeric_id <= 0:
                continue
            used_ids.add(numeric_id)
            section_slides.append(slide_by_id.get(numeric_id, {"id": numeric_id}))
        if not section_slides:
            continue
        sections.append(
            {
                "id": item.get("id") or index,
                "heading": item.get("heading") or item.get("title") or f"Section {index}",
                "slides": section_slides,
            }
        )

    extras = [slide for slide in slides if _slide_id(slide) not in used_ids]
    if extras:
        heading = "Additional slides" if sections else "Lecture slides"
        sections.append({"id": len(sections) + 1, "heading": heading, "slides": extras})
    return sections


def _slide_card(slide: Dict, analysis: Dict) -> str:
    slide_id = _slide_id(slide)
    title = analysis.get("descriptive_filename") or slide.get("filename") or f"Slide {slide_id}"
    summary = analysis.get("summary") or "No slide summary is available yet."
    caption = analysis.get("caption")
    commentary = analysis.get("instructor_commentary") or ""
    tags = analysis.get("tags") or []
    image_path = slide.get("relative_path")
    if image_path:
        media = f'<img src="{_e("../" + str(image_path))}" alt="{_e(title)}">'
    else:
        media = '<span class="missing-slide">Slide image not available</span>'
    caption_html = f'<p class="caption"><strong>Visible on slide:</strong> {_e(caption)}</p>' if caption else ""
    commentary_html = f'<p class="commentary">{_e(commentary)}</p>' if commentary else ""
    tags_html = "".join(f'<span class="tag">{_e(str(tag))}</span>' for tag in tags)
    return f"""
      <article id="slide-{slide_id}" class="slide-row">
        <div class="slide-media">{media}</div>
        <div class="slide-copy">
          <div class="slide-title-row"><span class="slide-id">{slide_id}</span><h4>{_e(title)}</h4></div>
          <p class="slide-summary">{_e(summary)}</p>
          {caption_html}
          {commentary_html}
          <div class="tag-row">{tags_html}</div>
        </div>
      </article>
    """


def _slide_id(slide: Dict) -> int:
    return _numeric_slide_id(slide.get("id") or slide.get("slide_id") or slide.get("filename"))


def _analysis_slide_id(item: Dict) -> int:
    return _numeric_slide_id(item.get("slide_id") or item.get("id") or item.get("descriptive_filename"))


def _outline_slide_ids(item: Dict) -> List[int]:
    if item.get("slide_ids") is not None:
        return _numeric_slide_ids(item.get("slide_ids"))
    if item.get("slide_id") is not None:
        return _numeric_slide_ids(item.get("slide_id"))
    return []


def _numeric_slide_ids(value) -> List[int]:
    if isinstance(value, (list, tuple, set)):
        return [_numeric_slide_id(item) for item in value]
    if isinstance(value, str):
        matches = re.findall(r"\d+", value)
        if matches:
            return [int(match) for match in matches]
    numeric_id = _numeric_slide_id(value)
    return [numeric_id] if numeric_id else []


def _numeric_slide_id(value) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(value))
        return int(match.group(0)) if match else 0


def _resources_block(resources: List[Dict]) -> str:
    if not resources:
        return ""
    items = "".join(
        f"""
        <article class="resource-card">
          <div>
            <a href="{_e(item.get('url', '#'))}">{_e(item.get('title', 'Resource'))}</a>
            <p>{_e(item.get('summary', ''))}</p>
          </div>
          <span class="quality {_e(item.get('source_quality', 'medium'))}">{_e(item.get('source_quality', 'medium'))}</span>
        </article>
        """
        for item in resources
    )
    return f"""
      <section class="content-band">
        <div class="section-head"><div><p class="eyebrow">Resources</p><h2>Further Learning</h2></div></div>
        <div class="resource-list">{items}</div>
      </section>
    """


def _transcript_block(text: str) -> str:
    if not text:
        return ""
    paragraphs = "".join(f"<p>{_e(paragraph)}</p>" for paragraph in _paragraphs(text))
    return f"""
      <section class="content-band">
        <div class="section-head"><div><p class="eyebrow">Transcript</p><h2>Lecture Transcript</h2></div></div>
        <div class="transcript-text">{paragraphs}</div>
      </section>
    """


def _paragraphs(text: str) -> List[str]:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    blocks = [block.strip() for block in normalized.split("\n\n") if block.strip()]
    return [" ".join(block.split()) for block in blocks]


def _batch_row(result: FileResult, title: str, description: str, link: Optional[str], thumb: Optional[Path], output_dir: Path) -> str:
    image_html = ""
    if thumb:
        image_html = f"<img src=\"{_e(_relative(thumb, output_dir))}\" alt=\"{_e(title)} preview\">"
    else:
        image_html = "<div></div>"
    action = f"<a href=\"{_e(link)}\">Open</a>" if link else f"<span class=\"status\">{_e(result.status.value)}</span>"
    return f"""
      <article class="lecture-row {result.status.value}">
        {image_html}
        <div>
          <h2>{_e(title)}</h2>
          <p>{_e(description)}</p>
        </div>
        <div>{action}</div>
      </article>
    """


def _load_artifact_if_present(output_dir: Path) -> Optional[Dict]:
    path = output_dir / LECTURE_ARTIFACT_NAME
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def _fallback_title(artifact: Dict) -> str:
    return str(artifact.get("source", {}).get("filename") or artifact.get("lecture_id") or "Lecture")


def _fallback_summary(artifact: Dict) -> str:
    text = " ".join(str(artifact.get("transcript", {}).get("text") or "").split())
    if not text:
        return "This lecture was processed locally. No AI summary is available yet."
    if len(text) <= 360:
        return text
    return text[:357].rsplit(" ", 1)[0] + "..."


def _first_slide(output_dir: Path) -> Optional[Path]:
    slides = sorted((output_dir / "slides").glob("*.png"))
    return slides[0] if slides else None


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _e(value) -> str:
    return html.escape(str(value), quote=True)
