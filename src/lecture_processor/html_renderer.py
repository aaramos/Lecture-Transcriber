import html
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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
    slide_analysis = {item.get("slide_id"): item for item in enrichment.get("slide_analysis", [])}
    outline = enrichment.get("outline") or []
    resources = enrichment.get("resources") or []
    slides = artifact.get("slides") or []
    transcript = artifact.get("transcript", {})
    body = [
        _html_head(title),
        "<body>",
        "<main>",
        f"<header><p class=\"eyebrow\">Lecture Study Page</p><h1>{_e(title)}</h1>"
        f"<p class=\"summary\">{_e(summary)}</p></header>",
        _stats_block(artifact),
        _outline_block(outline),
        _slides_block(slides, slide_analysis),
        _resources_block(resources),
        _transcript_block(transcript.get("text") or ""),
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
      color-scheme: light;
      --ink: #24211d;
      --muted: #6f675d;
      --line: #ddd6cc;
      --paper: #fbfaf7;
      --accent: #9f5137;
      --soft: #f2eee7;
      --good: #2f7d4f;
      --bad: #a6423d;
      --warn: #906d1f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.55;
    }}
    main {{ width: min(1080px, calc(100vw - 32px)); margin: 0 auto; padding: 34px 0 56px; }}
    header {{ margin-bottom: 28px; }}
    h1 {{ margin: 0; font-size: clamp(2rem, 5vw, 4.5rem); line-height: 0.98; letter-spacing: 0; }}
    h2 {{ margin: 28px 0 12px; font-size: 1.45rem; }}
    h3 {{ margin: 0 0 6px; }}
    .eyebrow {{ margin: 0 0 8px; color: var(--accent); font-size: 0.78rem; font-weight: 800; text-transform: uppercase; }}
    .summary {{ max-width: 820px; margin: 16px 0 0; color: var(--muted); font-size: 1.08rem; }}
    .stats, .lecture-list, .outline, .resources, .transcript {{ border-top: 1px solid var(--line); padding-top: 18px; margin-top: 22px; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; }}
    .stat {{ background: var(--soft); border-radius: 6px; padding: 12px; }}
    .stat strong {{ display: block; font-size: 1.4rem; }}
    .stat span {{ color: var(--muted); font-size: 0.82rem; }}
    .lecture-row {{ display: grid; grid-template-columns: 132px minmax(0, 1fr) auto; gap: 16px; align-items: center; border-bottom: 1px solid var(--line); padding: 16px 0; }}
    .lecture-row img {{ width: 132px; aspect-ratio: 16 / 9; object-fit: cover; border: 1px solid var(--line); border-radius: 4px; background: var(--soft); }}
    .lecture-row.failed h2, .lecture-row.failed .status {{ color: var(--bad); }}
    .lecture-row.skipped h2, .lecture-row.skipped .status {{ color: var(--warn); }}
    .lecture-row.stopped h2, .lecture-row.stopped .status {{ color: var(--warn); }}
    .lecture-row h2 {{ margin: 0 0 4px; font-size: 1.1rem; }}
    .lecture-row p {{ margin: 0; color: var(--muted); }}
    .lecture-row a {{ color: var(--accent); font-weight: 800; text-decoration: none; }}
    .slide {{ display: grid; grid-template-columns: minmax(220px, 420px) minmax(0, 1fr); gap: 18px; border-top: 1px solid var(--line); padding: 20px 0; }}
    .slide img {{ width: 100%; border: 1px solid var(--line); border-radius: 4px; background: white; }}
    .tags {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}
    .tag {{ background: var(--soft); border-radius: 999px; padding: 4px 8px; color: var(--muted); font-size: 0.78rem; }}
    .transcript pre {{ overflow: auto; white-space: pre-wrap; background: var(--soft); border-radius: 6px; padding: 14px; }}
    @media (max-width: 760px) {{
      .lecture-row, .slide {{ grid-template-columns: 1fr; }}
      .lecture-row img {{ width: 100%; }}
    }}
  </style>
</head>"""


def _stats_block(artifact: Dict) -> str:
    media = artifact.get("media", {})
    transcript = artifact.get("transcript", {})
    slides = artifact.get("slides") or []
    enrichment = artifact.get("enrichment")
    values = [
        (f"{media.get('duration_seconds', 0) / 60:.1f}", "minutes"),
        (str(transcript.get("word_count", 0)), "transcript words"),
        (str(len(slides)), "slides"),
        ("AI" if enrichment else "Local", "study notes"),
    ]
    return "<section class=\"stats\">" + "".join(
        f"<div class=\"stat\"><strong>{_e(value)}</strong><span>{_e(label)}</span></div>"
        for value, label in values
    ) + "</section>"


def _outline_block(outline: List[Dict]) -> str:
    if not outline:
        return ""
    items = "\n".join(
        f"<li><strong>{_e(item.get('heading', 'Section'))}</strong>"
        f"<span> slides {_e(', '.join(str(slide) for slide in item.get('slide_ids', [])) or 'none')}</span></li>"
        for item in outline
    )
    return f"<section class=\"outline\"><h2>Outline</h2><ol>{items}</ol></section>"


def _slides_block(slides: List[Dict], slide_analysis: Dict[int, Dict]) -> str:
    if not slides:
        return "<section><h2>Slides</h2><p>No slides were extracted for this lecture.</p></section>"
    rows = []
    for slide in slides:
        analysis = slide_analysis.get(slide.get("id")) or {}
        image = _e("../" + slide.get("relative_path", ""))
        title = analysis.get("descriptive_filename") or slide.get("filename") or f"Slide {slide.get('id')}"
        summary = analysis.get("summary") or "No slide summary is available yet."
        commentary = analysis.get("instructor_commentary") or ""
        tags = analysis.get("tags") or []
        rows.append(
            f"""
            <article class="slide">
              <div><img src="{image}" alt="{_e(title)}"></div>
              <div>
                <p class="eyebrow">Slide {slide.get('id')}</p>
                <h3>{_e(title)}</h3>
                <p>{_e(summary)}</p>
                {f'<p><strong>Instructor context:</strong> {_e(commentary)}</p>' if commentary else ''}
                <div class="tags">{''.join(f'<span class="tag">{_e(str(tag))}</span>' for tag in tags)}</div>
              </div>
            </article>
            """
        )
    return "<section><h2>Slides And Commentary</h2>" + "\n".join(rows) + "</section>"


def _resources_block(resources: List[Dict]) -> str:
    if not resources:
        return ""
    items = "\n".join(
        f"<li><a href=\"{_e(item.get('url', '#'))}\">{_e(item.get('title', 'Resource'))}</a>"
        f"<p>{_e(item.get('summary', ''))}</p></li>"
        for item in resources
    )
    return f"<section class=\"resources\"><h2>Further Learning</h2><ul>{items}</ul></section>"


def _transcript_block(text: str) -> str:
    if not text:
        return ""
    return f"<section class=\"transcript\"><h2>Transcript</h2><pre>{_e(text)}</pre></section>"


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
