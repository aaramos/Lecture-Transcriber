import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .ai.providers.mlx_openai import (
    VISION_HQ_PROFILE,
    _OpenAICompatibleClient,
)

DEFAULT_SLIDE_CLASSIFIER_MODEL = "gemma-4-26b-a4b-it-ram-14gb-mlx"

CLASSIFY_PROMPT = """/think
You are a vision model classifying a single video frame. Follow these
steps in order.

STEP 1 — IGNORE VIDEO PLAYER ARTIFACTS
Before examining the image, mentally block out the bottom edge of the
frame. The following are always present and must be completely ignored:
- Play/pause buttons, progress bars, timeline scrubbers
- Volume controls, fullscreen buttons, playback speed indicators
- Timestamps or duration displays (e.g. "00:22 / 04:22")
- Any control bar or overlay along the bottom edge
These are not part of the image content. Do not mention them. Do not let
them influence your classification.

STEP 2 — DESCRIBE WHAT YOU SEE
Ignoring video player artifacts, describe the frame in 2-3 sentences.
Be specific:
- Is there a person? Where in the frame?
- Is there text? Quote it exactly
- Is there a diagram, chart, or graphic?
- What is the background — solid color, slide design, real environment?

STEP 3 — CLASSIFY
Using only your Step 2 description, work through this decision tree in
order. Stop at the first rule that matches.

  A. Is there a diagonal wipe streak, transition effect, or motion blur
     cutting across the frame?
     → Ask: is a person the primary subject, filling most of the frame,
       with no presentation slide content visible in either half?
       If YES (person-dominant, no readable slide): NOT SLIDE
       If NO (slide content is still readable and fills a significant
       portion): SLIDE

  B. Is the frame blank, solid color, or near-solid with no readable
     text or graphics?
     → NOT SLIDE

  C. Is the only visible text a name/title lower-third, a logo watermark,
     a small branded bug (e.g. "Northwestern | Kellogg"), or a branded
     title card over a solid color background?
     → NOT SLIDE
     (A watermark, lower-third, or branding-only title card does NOT
     make a frame a SLIDE)

  D. Does a presentation slide fill roughly 40% or more of the frame as
     the primary visual — with readable content, bullet points, diagrams,
     or structured graphics?
     → SLIDE

  E. Does a person appear on one side only, with slide content dominating
     the other half?
     → SLIDE

  F. None of the above match?
     → NOT SLIDE (default)

IMPORTANT RULES:
- You MUST output exactly SLIDE or NOT SLIDE. No other verdict is valid.
- UNCERTAIN is not an acceptable response under any circumstances.
- When two rules seem to conflict, the earlier rule wins.
- A wrong confident answer is better than a hedged non-answer.

STEP 4 — ATTRIBUTES (SLIDE verdicts only)
Only complete this step if VERDICT is SLIDE. If VERDICT is NOT SLIDE,
skip this step entirely — output nothing for these fields.

TITLE:
  Read the slide heading or title directly from the frame.
  If no explicit title is visible, infer a 4-6 word label from the most
  prominent text or content.
  If the slide is mid-build with no title yet visible, use the most
  prominent text you can read.
  Keep it consistent — if you can read "Questions for Business Leaders"
  write exactly that, not a paraphrase. This field is used for grouping.

BUILD_STAGE:
  partial      — slide is visible but content is still being revealed
  full         — complete slide content visible, nothing greyed out
  transitioning — wipe/blur/animation present but slide still dominant

LAYOUT:
  full-screen  — slide fills entire frame, no person visible
  split-left   — slide on left half, person on right
  split-right  — slide on right half, person on left

---

OUTPUT FORMAT

If VERDICT is NOT SLIDE, output exactly two lines:
DESCRIPTION: [your 2-3 sentence description from Step 2]
VERDICT: NOT SLIDE

If VERDICT is SLIDE, output exactly five lines:
DESCRIPTION: [your 2-3 sentence description from Step 2]
VERDICT: SLIDE
TITLE: [slide title or inferred label]
BUILD_STAGE: partial | full | transitioning
LAYOUT: full-screen | split-left | split-right

No other text. No blank lines between output lines."""


@dataclass(frozen=True)
class SlideClassifierResult:
    description: str = ""
    verdict: str = "NOT SLIDE"
    title: Optional[str] = None
    build_stage: Optional[str] = None
    layout: Optional[str] = None
    raw_response: str = ""


class LmStudioSlideClassifier:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: int,
        prompt: str = CLASSIFY_PROMPT,
    ) -> None:
        self.model = str(model or DEFAULT_SLIDE_CLASSIFIER_MODEL).strip() or DEFAULT_SLIDE_CLASSIFIER_MODEL
        self.prompt = prompt
        self._client = _OpenAICompatibleClient(
            base_url,
            self.model,
            timeout_seconds,
            profile_name=VISION_HQ_PROFILE,
            disable_thinking=False,
        )

    def classify(self, image_path: Path) -> SlideClassifierResult:
        raw, _usage = self._client.chat_text_with_openai_images(
            _vision_messages(self.prompt, image_path),
            max_tokens=250,
            temperature=0.0,
            context=f"Smart slide classifier {image_path.name}",
        )
        parsed = parse_classifier_response(raw)
        return SlideClassifierResult(raw_response=raw, **parsed)


def parse_classifier_response(text: str) -> Dict[str, Optional[str]]:
    result: Dict[str, Optional[str]] = {
        "description": "",
        "verdict": "NOT SLIDE",
        "title": None,
        "build_stage": None,
        "layout": None,
    }
    for line in str(text or "").strip().splitlines():
        if line.startswith("DESCRIPTION:"):
            result["description"] = line[len("DESCRIPTION:") :].strip()
        elif line.startswith("VERDICT:"):
            result["verdict"] = line[len("VERDICT:") :].strip()
        elif line.startswith("TITLE:"):
            result["title"] = line[len("TITLE:") :].strip()
        elif line.startswith("BUILD_STAGE:"):
            result["build_stage"] = line[len("BUILD_STAGE:") :].strip()
        elif line.startswith("LAYOUT:"):
            result["layout"] = line[len("LAYOUT:") :].strip()
    if result["verdict"] != "SLIDE":
        result["verdict"] = "NOT SLIDE"
        result["title"] = None
        result["build_stage"] = None
        result["layout"] = None
    return result


def _vision_messages(prompt: str, image_path: Path) -> List[Dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(image_path)},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _image_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{_image_mime_type(image_path)};base64,{encoded}"


def _image_mime_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "application/octet-stream"
