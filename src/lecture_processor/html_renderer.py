import html
import re
from pathlib import Path
from typing import Dict, List, Optional

from .artifacts import LECTURE_ARTIFACT_NAME, load_json
from .models import BatchSummary, FileResult, FileStatus
from .writers import write_text_atomic

LOCAL_OVERVIEW_STUB_SUMMARY = (
    "Local overview stub. This placeholder keeps the experimental routing path working "
    "until a real local overview model is configured."
)


def render_lecture_page(lecture_json_path: Path) -> Path:
    artifact = load_json(lecture_json_path)
    output_dir = lecture_json_path.parent
    html_dir = output_dir / "html"
    html_path = html_dir / "index.html"
    write_text_atomic(html_path, _lecture_html(artifact, output_dir))
    return html_path


def render_batch_index(output_dir: Path, summary: BatchSummary) -> Path:
    html_path = output_dir / "index.html"
    write_text_atomic(html_path, _batch_html(output_dir, summary))
    return html_path


def _lecture_html(artifact: Dict, output_dir: Path) -> str:
    enrichment = artifact.get("enrichment") or {}
    title = enrichment.get("title") or _fallback_title(artifact)
    slide_analysis = {_analysis_slide_id(item): item for item in enrichment.get("slide_analysis", [])}
    outline = enrichment.get("outline") or []
    resources = enrichment.get("resources") or []
    slides = artifact.get("slides") or []
    transcript = artifact.get("transcript", {})
    transcript_segments = transcript.get("segments") or []
    transcript_text = enrichment.get("formatted_transcript") or transcript.get("text") or ""
    body = [
        _html_head(title),
        "<body>",
        "<main>",
        _hero_block(title, slides, outline, resources),
        _video_block(artifact, output_dir),
        _flow_block(outline, slides, slide_analysis, transcript_segments),
        _transcript_block(transcript_text),
        _resources_block(resources),
        "</main>",
        _image_modal(),
        _modal_script(),
        "</body></html>",
    ]
    return "\n".join(body)


def _batch_html(output_dir: Path, summary: BatchSummary) -> str:
    rows = []
    for result in summary.results:
        artifact = _load_artifact_if_present(result.output_dir)
        title = (artifact.get("enrichment") or {}).get("title") if artifact else None
        title = title or result.title or _display_name(result.source.stem)
        description = (artifact.get("enrichment") or {}).get("executive_summary") if artifact else None
        description = _batch_description(description or result.short_summary or result.message or result.status.value)
        html_path = result.html_path or (result.output_dir / "html" / "index.html")
        link = _relative(html_path, output_dir) if html_path.exists() else None
        thumb = _first_slide(result.output_dir)
        rows.append(_batch_row(result, title, description, link, thumb, output_dir))

    display_title = _display_name(output_dir.name)
    title = f"{display_title} Study Index"
    return "\n".join(
        [
            _html_head(title),
            "<body>",
            "<main>",
            f"<header><h1>{_e(display_title)}</h1></header>",
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
    h1 {{ margin: 0; max-width: none; font-size: clamp(2.4rem, 7vw, 5.4rem); line-height: 1.02; letter-spacing: 0; }}
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
      grid-template-columns: 1fr;
      gap: 24px;
      min-height: 0;
      align-items: start;
    }}
    .metric-strip {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; max-width: 720px; }}
    .metric-strip div {{
      display: flex;
      align-items: baseline;
      gap: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      background: var(--panel);
    }}
    .metric-strip strong {{ font-size: 1.45rem; }}
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
    .outline-number {{
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
    .video-panel {{
      display: grid;
      gap: 12px;
      max-width: 980px;
    }}
    .video-panel video {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #111;
    }}
    .slide-media {{
      display: grid;
      min-height: 180px;
      gap: 10px;
      overflow: hidden;
      margin: 0;
    }}
    .slide-image-button {{
      display: grid;
      width: 100%;
      min-height: 180px;
      place-items: center;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0;
      background: #111;
      color: inherit;
      cursor: zoom-in;
    }}
    .slide-image-button img {{ width: 100%; height: 100%; object-fit: contain; }}
    .slide-image-button:focus-visible {{
      outline: 3px solid var(--blue);
      outline-offset: 3px;
    }}
    .missing-slide {{ color: var(--warn); font-size: 0.75rem; font-weight: 800; text-align: center; }}
    .slide-copy {{ display: grid; gap: 12px; align-content: start; }}
    .commentary, .caption, .resource-card p, .transcript-text {{ color: #ded9ca; font-size: 0.93rem; line-height: 1.6; }}
    .caption {{ border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; background: rgba(255, 255, 255, 0.03); color: var(--muted); font-style: italic; }}
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
    .image-modal {{
      width: min(96vw, 1280px);
      max-height: 94vh;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: var(--panel);
      color: var(--text);
    }}
    .image-modal::backdrop {{ background: rgba(0, 0, 0, 0.78); }}
    .image-modal figure {{ display: grid; gap: 12px; margin: 0; }}
    .image-modal img {{ width: 100%; max-height: 78vh; object-fit: contain; background: #111; border-radius: 6px; }}
    .image-modal figcaption {{ color: #ded9ca; font-size: 0.95rem; line-height: 1.55; }}
    .modal-close {{
      float: right;
      margin-bottom: 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      background: transparent;
      color: var(--text);
      font-weight: 900;
      cursor: pointer;
    }}
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
      .metric-strip {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
      .flow-section summary {{ grid-template-columns: auto 42px minmax(0, 1fr); }}
      .flow-count {{ grid-column: 3; }}
      .slide-row, .resource-card, .lecture-row {{ grid-template-columns: 1fr; }}
      .lecture-row img {{ width: 100%; }}
    }}
    @media (max-width: 640px) {{
      .metric-strip {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>"""


def _hero_block(title: str, slides: List[Dict], outline: List[Dict], resources: List[Dict]) -> str:
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
          <h1>{_e(title)}</h1>
        </div>
        <div class="metric-strip" aria-label="Lecture metrics">{metric_html}</div>
      </section>
    """


def _video_block(artifact: Dict, output_dir: Path) -> str:
    video_src = _processed_video_src(artifact, output_dir)
    if not video_src:
        return ""
    video_label = _video_label(artifact, video_src)
    return f"""
      <section class="content-band" aria-label="{_e(video_label)}">
        <div class="section-head"><div><p class="eyebrow">{_e(video_label)}</p></div></div>
        <div class="video-panel">
          <video controls preload="metadata">
            <source src="{_e(video_src)}" type="{_e(_video_mime_type(video_src))}">
          </video>
        </div>
      </section>
    """


def _flow_block(
    outline: List[Dict],
    slides: List[Dict],
    slide_analysis: Dict[int, Dict],
    transcript_segments: List[Dict],
) -> str:
    sections = _sections_with_slides(outline, slides, slide_analysis)
    if not sections:
        return """
          <section class="content-band">
            <div class="section-head"><div><h2>Lecture Flow</h2></div></div>
            <p class="empty-state">No slides were extracted for this lecture.</p>
          </section>
        """

    rows = []
    for section in sections:
        rows.append(
            f"""
            <details id="section-{_e(section["id"])}" class="flow-section" open>
              <summary>
                <span class="outline-number">{_e(section["id"])}</span>
                <span class="flow-title">{_e(_section_title(section, slide_analysis))}</span>
                <span class="flow-count">{len(section["slides"])} slides</span>
              </summary>
              <div class="flow-slides">
                {''.join(_slide_card(slide, slide_analysis.get(_slide_id(slide), {}), transcript_segments) for slide in section["slides"])}
              </div>
            </details>
            """
        )
    return f"""
      <section class="content-band">
        <div class="section-head"><div><h2>Lecture Flow</h2></div></div>
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


def _section_title(section: Dict, slide_analysis: Dict[int, Dict]) -> str:
    section_slides = section.get("slides") or []
    if len(section_slides) == 1:
        slide = section_slides[0]
        title = _known_slide_title(slide, slide_analysis.get(_slide_id(slide), {}))
        if title:
            return title
    heading = str(section.get("heading") or "").strip()
    if _generic_section_heading(heading):
        return ""
    return heading


def _generic_section_heading(value: str) -> bool:
    heading = value.strip().lower()
    if not heading:
        return True
    return bool(
        re.fullmatch(r"slide\s+\d+\s+discussion", heading)
        or re.fullmatch(r"section\s+\d+", heading)
        or heading in {"lecture slides", "additional slides"}
    )


def _slide_card(slide: Dict, analysis: Dict, transcript_segments: List[Dict]) -> str:
    slide_id = _slide_id(slide)
    title = _known_slide_title(slide, analysis)
    idea = _slide_idea(slide, analysis)
    commentary = _slide_commentary(slide, analysis, transcript_segments)
    tags = [str(tag) for tag in (analysis.get("tags") or []) if _useful_tag(str(tag))]
    image_path = slide.get("relative_path")
    caption_html = f'<figcaption class="caption">{_e(idea)}</figcaption>' if idea else ""
    image_alt = title or slide.get("filename") or f"Slide {slide_id}"
    if image_path:
        image_src = "../" + str(image_path)
        media = (
            f'<figure class="slide-media">'
            f'<button class="slide-image-button" type="button" data-full-image="{_e(image_src)}" '
            f'data-caption="{_e(idea)}" data-alt="{_e(image_alt)}">'
            f'<img src="{_e(image_src)}" alt="{_e(image_alt)}">'
            f'</button>{caption_html}</figure>'
        )
    else:
        media = '<div class="slide-media"><span class="missing-slide">Slide image not available</span></div>'
    commentary_html = f'<p class="commentary">{_e(commentary)}</p>' if commentary else ""
    tags_html = "".join(f'<span class="tag">{_e(tag)}</span>' for tag in tags)
    tag_row = f'<div class="tag-row">{tags_html}</div>' if tags_html else ""
    return f"""
      <article id="slide-{slide_id}" class="slide-row">
        {media}
        <div class="slide-copy">
          {commentary_html}
          {tag_row}
        </div>
      </article>
    """


def _known_slide_title(slide: Dict, analysis: Dict) -> str:
    for value in (slide.get("title"), analysis.get("title")):
        title = _clean_title(value)
        if title:
            return title
    return ""


def _clean_title(value) -> str:
    title = " ".join(str(value or "").split())
    if not title:
        return ""
    if re.search(r"\.(png|jpe?g|webp|gif)$", title, re.IGNORECASE):
        return ""
    if re.fullmatch(r"slide[-_\s]*\d+.*", title, re.IGNORECASE):
        return ""
    return title


def _slide_idea(slide: Dict, analysis: Dict) -> str:
    title = _known_slide_title(slide, analysis)
    for key in ("summary", "caption"):
        candidate = _content_summary(analysis.get(key), title)
        if candidate and not _looks_visual_description(candidate):
            return _trim_text(candidate, 420)
    return title


def _content_summary(value, title: str) -> str:
    text = _clean_commentary(value)
    if not text:
        return ""
    text = _strip_leading_title(text, title)
    text = re.sub(
        r'^the frame shows a presentation slide titled "[^"]+"\s+with\s+',
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r'^the slide shows a presentation slide titled "[^"]+"\s+with\s+',
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r'^a presentation slide titled "[^"]+"\s+with\s+', "", text, flags=re.IGNORECASE)
    text = re.sub(r"^the frame (shows|features|contains)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^the slide (shows|features|contains)\s+", "", text, flags=re.IGNORECASE)
    text = text.strip(" .")
    if not text:
        return ""
    return text[0].upper() + text[1:] + "."


def _strip_leading_title(text: str, title: str) -> str:
    clean_title = _clean_title(title)
    if clean_title and text.lower().startswith(clean_title.lower()):
        return text[len(clean_title) :].strip(" :-")
    return text


def _looks_visual_description(text: str) -> bool:
    lowered = text.lower()
    visual_terms = [
        "background",
        "bottom left",
        "diagonal",
        "frame",
        "logo",
        "person",
        "purple",
        "right side",
        "left side",
        "transition effect",
    ]
    return sum(1 for term in visual_terms if term in lowered) >= 2


def _slide_commentary(slide: Dict, analysis: Dict, transcript_segments: List[Dict]) -> str:
    commentary = _clean_commentary(analysis.get("instructor_commentary"))
    if not commentary:
        commentary = _commentary_from_segments(slide, transcript_segments)
    return _trim_text(commentary, 900)


def _commentary_from_segments(slide: Dict, transcript_segments: List[Dict]) -> str:
    linked_ids = {str(item) for item in (slide.get("linked_segment_ids") or [])}
    if not linked_ids:
        return ""
    parts = [
        str(segment.get("text") or "").strip()
        for segment in transcript_segments
        if str(segment.get("id")) in linked_ids and str(segment.get("text") or "").strip()
    ]
    return " ".join(parts)


def _clean_commentary(value) -> str:
    text = " ".join(str(value or "").split())
    text = re.sub(r"\[\d+\]\s*", "", text)
    return " ".join(text.split())


def _trim_text(value: str, max_length: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rsplit(" ", 1)[0] + "..."


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
      <section class="content-band" aria-label="Further Learning">
        <div class="section-head"><div><p class="eyebrow">Further Learning</p></div></div>
        <div class="resource-list">{items}</div>
      </section>
    """


def _transcript_block(text: str) -> str:
    if not text:
        return ""
    paragraphs = "".join(f"<p>{_e(paragraph)}</p>" for paragraph in _paragraphs(text))
    return f"""
      <section class="content-band" aria-label="Transcript">
        <div class="section-head"><div><p class="eyebrow">Transcript</p></div></div>
        <div class="transcript-text">{paragraphs}</div>
      </section>
    """


def _paragraphs(text: str) -> List[str]:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    blocks = [block.strip() for block in normalized.split("\n\n") if block.strip()]
    if len(blocks) > 1:
        return [" ".join(block.split()) for block in blocks]
    return _sentence_paragraphs(" ".join(normalized.split()))


def _sentence_paragraphs(text: str) -> List[str]:
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text) if item.strip()]
    if len(sentences) <= 1:
        return [text]
    paragraphs = []
    current = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence)
        if current and (current_length + sentence_length > 760 or len(current) >= 5):
            paragraphs.append(" ".join(current))
            current = []
            current_length = 0
        current.append(sentence)
        current_length += sentence_length + 1
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def _processed_video_src(artifact: Dict, output_dir: Path) -> Optional[str]:
    normalized_path = str((artifact.get("media") or {}).get("normalized_path") or "").strip()
    if not normalized_path:
        return None
    candidate = Path(normalized_path)
    if not candidate.is_absolute():
        candidate = output_dir / candidate
    if not candidate.exists():
        return None
    try:
        relative = candidate.relative_to(output_dir)
    except ValueError:
        return candidate.as_uri()
    return "../" + str(relative)


def _video_label(artifact: Dict, src: str) -> str:
    source = artifact.get("source") or {}
    media = artifact.get("media") or {}
    name = source.get("filename") or media.get("normalized_path") or Path(src).name
    return _human_name(Path(str(name)).stem)


def _human_name(value: str) -> str:
    return " ".join(str(value or "").replace("_", " ").split())


def _video_mime_type(src: str) -> str:
    extension = Path(src).suffix.lower()
    if extension == ".webm":
        return "video/webm"
    if extension == ".mov":
        return "video/quicktime"
    return "video/mp4"


def _useful_tag(value: str) -> bool:
    tag = value.strip().lower()
    if not tag:
        return False
    blocked = {
        "smart-slide-extraction",
        "transitioning",
        "partial",
        "full",
        "full-screen",
        "split-left",
        "split-right",
        "video-player",
        "duplicate",
        "not-slide",
    }
    if tag in blocked:
        return False
    if re.fullmatch(r"slide[-_\s]*\d+", tag):
        return False
    return True


def _image_modal() -> str:
    return """
      <dialog id="image-modal" class="image-modal">
        <button class="modal-close" type="button" value="close" aria-label="Close image">Close</button>
        <figure>
          <img alt="">
          <figcaption hidden></figcaption>
        </figure>
      </dialog>
    """


def _modal_script() -> str:
    return """
      <script>
        (() => {
          const modal = document.getElementById("image-modal");
          if (!modal) return;
          const modalImage = modal.querySelector("img");
          const modalCaption = modal.querySelector("figcaption");
          const closeButton = modal.querySelector(".modal-close");

          document.querySelectorAll("[data-full-image]").forEach((button) => {
            button.addEventListener("click", () => {
              modalImage.src = button.dataset.fullImage || "";
              modalImage.alt = button.dataset.alt || "Slide image";
              modalCaption.textContent = button.dataset.caption || "";
              modalCaption.hidden = !modalCaption.textContent.trim();
              modal.showModal();
            });
          });

          closeButton.addEventListener("click", () => modal.close());
          modal.addEventListener("click", (event) => {
            if (event.target === modal) modal.close();
          });
          modal.addEventListener("close", () => {
            modalImage.removeAttribute("src");
            modalCaption.textContent = "";
            modalCaption.hidden = true;
          });
        })();
      </script>
    """


def _batch_row(result: FileResult, title: str, description: str, link: Optional[str], thumb: Optional[Path], output_dir: Path) -> str:
    image_html = ""
    if thumb:
        image_html = f"<img src=\"{_e(_relative(thumb, output_dir))}\" alt=\"{_e(title)} preview\">"
    else:
        image_html = "<div></div>"
    action = f"<a href=\"{_e(link)}\">Open</a>" if link else f"<span class=\"status\">{_e(result.status.value)}</span>"
    description_html = f"<p>{_e(description)}</p>" if description else ""
    return f"""
      <article class="lecture-row {result.status.value}">
        {image_html}
        <div>
          <h2>{_e(title)}</h2>
          {description_html}
        </div>
        <div>{action}</div>
      </article>
    """


def _batch_description(value: str) -> str:
    clean = " ".join(str(value or "").split())
    if not clean:
        return ""
    if clean == LOCAL_OVERVIEW_STUB_SUMMARY:
        return "Overview unavailable"
    return clean


def _display_name(value: str) -> str:
    clean = str(value or "").strip()
    clean = re.sub(r"\.[A-Za-z0-9]+$", "", clean)
    clean = re.sub(r"([_-])processed$", "", clean, flags=re.IGNORECASE)
    clean = clean.replace("_", " ")
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean or "Lecture"


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
