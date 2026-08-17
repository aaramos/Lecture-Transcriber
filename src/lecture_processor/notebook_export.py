import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .artifacts import load_json
from .timecode import format_timecode, format_timecode_range
from .writers import write_text_atomic

CONTINUATION_WORD_THRESHOLD = 250
TRANSCRIPT_CHUNK_WORD_TARGET = 200

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_LEADING_HASH = re.compile(r"(?m)^(#+)")


def export_notebook_markdown(lecture_json_path: Path, course: str = "") -> Optional[Path]:
    """Render lecture.json to notebook.md for Open Notebook ingestion."""
    artifact = load_json(lecture_json_path)
    processing = artifact.get("processing") or {}
    if processing.get("status") == "failed":
        return None
    output_dir = lecture_json_path.parent
    notebook_path = output_dir / "notebook.md"
    write_text_atomic(notebook_path, _render_notebook(artifact, lecture_json_path, course))
    return notebook_path


def _render_notebook(artifact: Dict, lecture_json_path: Path, course: str) -> str:
    enrichment = artifact.get("enrichment") or {}
    slides = artifact.get("slides") or []
    transcript = artifact.get("transcript") or {}
    segments = transcript.get("segments") or []
    transcript_end = _transcript_end(segments)
    lines: List[str] = []
    lines.append(_h1_title(artifact, lecture_json_path, course))
    lines.append(_source_line(artifact, transcript_end))
    summary = _summary_section(enrichment)
    if summary:
        lines.append(summary)
    topics = _key_topics_section(enrichment)
    if topics:
        lines.append(topics)
    if slides:
        lines.extend(_slide_sections(slides, enrichment, segments, transcript_end))
    elif segments:
        lines.append(_transcript_only_sections(segments))
    reading = _further_reading_section(enrichment)
    if reading:
        lines.append(reading)
    parts = [line for line in lines if line]
    return "\n\n".join(parts) + "\n"


def _transcript_end(segments: List[Dict]) -> float:
    return max((float(seg.get("end") or 0.0) for seg in segments), default=0.0)


def _h1_title(artifact: Dict, lecture_json_path: Path, course: str) -> str:
    enrichment = artifact.get("enrichment") or {}
    title = enrichment.get("title") or artifact.get("lecture_id") or lecture_json_path.parent.name
    context = course or lecture_json_path.parent.name
    return f"# {context} \u2014 {title}"


def _source_line(artifact: Dict, transcript_end: float) -> str:
    source = artifact.get("source") or {}
    media = artifact.get("media") or {}
    slides = artifact.get("slides") or []
    filename = source.get("filename") or ""
    duration_seconds = float(media.get("duration_seconds") or 0.0)
    effective_duration = transcript_end if transcript_end > 0.0 else duration_seconds
    duration = _format_duration(effective_duration)
    slide_count = len(slides)
    return f"Source: {filename} \u00b7 {duration} \u00b7 {slide_count} slides"


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes = total // 60
    secs = total % 60
    if secs == 0:
        return f"{minutes} min"
    return f"{minutes} min {secs} sec"


def _summary_section(enrichment: Dict) -> str:
    text = (enrichment.get("executive_summary") or "").strip()
    if not text:
        return ""
    return f"## Summary\n\n{text}"


def _key_topics_section(enrichment: Dict) -> str:
    outline = enrichment.get("outline") or []
    headings = [str(item.get("heading") or "").strip() for item in outline if item.get("heading")]
    headings = [h for h in headings if h]
    if not headings:
        return ""
    return f"## Key Topics\n\n{', '.join(headings)}"


def _slide_sections(slides: List[Dict], enrichment: Dict, segments: List[Dict], transcript_end: float) -> List[str]:
    analysis_map = _build_analysis_map(enrichment)
    slide_timestamps = [float(slide.get("timestamp_seconds") or 0.0) for slide in slides]
    first_ts = slide_timestamps[0] if slide_timestamps else 0.0
    pre_first_indices = _collect_pre_first_slide_indices(slides, segments)
    out: List[str] = []
    for index, slide in enumerate(slides):
        analysis = analysis_map.get(_slide_id(slide)) or {}
        title = _slide_title(slide, analysis)
        start = slide_timestamps[index]
        if index + 1 < len(slides):
            end = slide_timestamps[index + 1]
        else:
            end = transcript_end if transcript_end > start else start
        if end > start:
            range_str = format_timecode_range(start, end)
        else:
            range_str = f"({format_timecode(start)})"
        heading_base = f"Slide {_slide_id(slide)} \u2014 {title}"
        description = _slide_description(slide, analysis)
        transcript_text = _transcript_for_slide(slide, index, slides, segments)
        if index == 0 and pre_first_indices:
            prelude = _join_segments(segments, pre_first_indices)
            transcript_text = f"{prelude} {transcript_text}".strip() if transcript_text else prelude
        out.extend(_emit_slide_sections(heading_base, range_str, start, end, description, transcript_text))
    return out


def _emit_slide_sections(
    heading_base: str,
    range_str: str,
    start: float,
    end: float,
    description: str,
    transcript_text: str,
) -> List[str]:
    word_count = len(transcript_text.split()) if transcript_text else 0
    if word_count <= CONTINUATION_WORD_THRESHOLD or not transcript_text:
        body = _format_slide_body(description, transcript_text)
        heading = f"### {heading_base} {range_str}"
        return [f"{heading}\n\n{body}".rstrip() if body else heading]
    return _split_into_continuations(heading_base, range_str, start, end, description, transcript_text)


def _split_into_continuations(
    heading_base: str,
    range_str: str,
    start: float,
    end: float,
    description: str,
    transcript_text: str,
) -> List[str]:
    sentences = _split_sentences(transcript_text)
    chunks = _chunk_sentences(sentences, CONTINUATION_WORD_THRESHOLD)
    if len(chunks) <= 1:
        body = _format_slide_body(description, transcript_text)
        heading = f"### {heading_base} {range_str}"
        return [f"{heading}\n\n{body}".rstrip() if body else heading]
    section_times = _continuation_time_ranges(chunks, start, end)
    out: List[str] = []
    first_body = _format_slide_body(description, chunks[0].strip())
    first_heading = f"### {heading_base} {range_str}"
    out.append(f"{first_heading}\n\n{first_body}".rstrip() if first_body else first_heading)
    for i, chunk in enumerate(chunks[1:], start=2):
        label = f"{heading_base} (cont. {i})"
        cstart, cend = section_times[i - 1]
        if cend > cstart:
            crange = format_timecode_range(cstart, cend)
        else:
            crange = f"({format_timecode(cstart)})"
        cbody = _neutralize_markdown(f"**Instructor said:** {chunk.strip()}")
        out.append(f"### {label} {crange}\n\n{cbody}")
    return out


def _format_slide_body(description: str, transcript_text: str) -> str:
    body_lines: List[str] = []
    if description:
        body_lines.append(_neutralize_markdown(f"**On the slide:** {description}"))
    if transcript_text:
        body_lines.append(_neutralize_markdown(f"**Instructor said:** {transcript_text.strip()}"))
    return "\n\n".join(body_lines)


def _continuation_time_ranges(chunks: List[str], start: float, end: float) -> List[Tuple[float, float]]:
    if end <= start or len(chunks) <= 1:
        return [(start, end)] * len(chunks)
    span = end - start
    per = span / len(chunks)
    ranges: List[Tuple[float, float]] = [(start, start + per)]
    for i in range(1, len(chunks)):
        prev_end = ranges[i - 1][1]
        cur_end = prev_end + per if i < len(chunks) - 1 else end
        ranges.append((prev_end, cur_end))
    return ranges


def _split_sentences(text: str) -> List[str]:
    cleaned = " ".join(text.split())
    parts = _SENTENCE_END.split(cleaned)
    return [p.strip() for p in parts if p.strip()]


def _chunk_sentences(sentences: List[str], target_words: int) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_words = 0
    for sentence in sentences:
        words = len(sentence.split())
        if current and current_words + words > target_words:
            chunks.append(" ".join(current))
            current = [sentence]
            current_words = words
        else:
            current.append(sentence)
            current_words += words
    if current:
        chunks.append(" ".join(current))
    return chunks


def _build_analysis_map(enrichment: Dict) -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}
    for item in enrichment.get("slide_analysis") or []:
        out[_analysis_slide_id(item)] = item or {}
    return out


def _transcript_for_slide(slide: Dict, slide_index: int, slides: List[Dict], segments: List[Dict]) -> str:
    linked = slide.get("linked_segment_ids") or []
    if linked:
        return _join_segments(segments, linked)
    return _transcript_for_slide_window(slide_index, slides, segments)


def _transcript_for_slide_window(slide_index: int, slides: List[Dict], segments: List[Dict]) -> str:
    if not slides or not segments:
        return ""
    timestamps = [float(s.get("timestamp_seconds") or 0.0) for s in slides]
    window_start = max(0.0, timestamps[slide_index])
    if slide_index + 1 < len(slides):
        window_end = timestamps[slide_index + 1]
    else:
        window_end = max((float(seg.get("end") or 0.0) for seg in segments), default=window_start)
    if window_end <= window_start:
        window_end = window_start
    indices: List[int] = []
    for i, seg in enumerate(segments):
        seg_start = float(seg.get("start") or 0.0)
        seg_end = float(seg.get("end") or seg_start)
        if seg_end > window_start and seg_start < window_end:
            indices.append(i)
    if not indices:
        for i, seg in enumerate(segments):
            seg_start = float(seg.get("start") or 0.0)
            if seg_start >= window_start:
                indices = [i]
                break
    return _join_segments(segments, indices)


def _join_segments(segments: List[Dict], indices: List[int]) -> str:
    parts: List[str] = []
    for idx in indices:
        if 0 <= idx < len(segments):
            text = str(segments[idx].get("text") or "").strip()
            if text:
                parts.append(text)
    joined = " ".join(parts)
    return " ".join(joined.split())


def _collect_pre_first_slide_indices(slides: List[Dict], segments: List[Dict]) -> List[int]:
    if not slides or not segments:
        return []
    first_ts = float(slides[0].get("timestamp_seconds") or 0.0)
    if first_ts <= 0.0:
        return []
    linked_union: set = set()
    for slide in slides:
        for idx in slide.get("linked_segment_ids") or []:
            linked_union.add(idx)
    out: List[int] = []
    for i, seg in enumerate(segments):
        if i in linked_union:
            continue
        seg_end = float(seg.get("end") or 0.0)
        if seg_end <= first_ts:
            out.append(i)
    return out


def _transcript_only_sections(segments: List[Dict]) -> str:
    full_text = " ".join(str(seg.get("text") or "").strip() for seg in segments if seg.get("text"))
    full_text = " ".join(full_text.split())
    if not full_text:
        return ""
    sentences = _split_sentences(full_text)
    chunks = _chunk_sentences(sentences, TRANSCRIPT_CHUNK_WORD_TARGET)
    out: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        body = _neutralize_markdown(chunk.strip())
        out.append(f"### Transcript (part {i})\n\n{body}")
    return "\n\n".join(out)


def _neutralize_markdown(text: str) -> str:
    return _LEADING_HASH.sub(r"\\\1", text)


def _slide_description(slide: Dict, analysis: Dict) -> str:
    for source in (analysis, slide):
        caption = (source.get("caption") or "").strip() if source else ""
        if caption:
            return caption
    desc = (slide.get("description") or "").strip() if slide else ""
    if desc:
        return desc
    summary = (analysis.get("summary") or "").strip() if analysis else ""
    if summary:
        return summary
    return ""


def _slide_title(slide: Dict, analysis: Dict) -> str:
    title = (slide.get("title") or "").strip() if slide else ""
    if title:
        return title
    if analysis:
        caption = (analysis.get("caption") or "").strip()
        if caption:
            return caption
    slide_id = _slide_id(slide)
    return f"Slide {slide_id}"


def _slide_id(slide: Dict) -> int:
    return _numeric_slide_id(slide.get("id") or slide.get("slide_id") or slide.get("filename"))


def _analysis_slide_id(item: Dict) -> int:
    return _numeric_slide_id(item.get("slide_id") or item.get("id") or item.get("descriptive_filename"))


def _numeric_slide_id(value) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(value))
        return int(match.group(0)) if match else 0


def _further_reading_section(enrichment: Dict) -> str:
    resources = enrichment.get("resources") or []
    items: List[str] = []
    for item in resources:
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        summary = (item.get("summary") or "").strip()
        if not title:
            continue
        link = f"[{title}]({url})" if url else title
        line = f"- {link}"
        if summary:
            line = f"{line} \u2014 {summary}"
        items.append(line)
    if not items:
        return ""
    return "## Further Reading\n\n" + "\n".join(items)