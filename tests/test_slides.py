import tempfile
import unittest
from pathlib import Path

from lecture_processor.slides import ClassifiedFrame, select_best_unique_slides


class SlideSelectionTests(unittest.TestCase):
    def test_select_best_unique_slides_keeps_full_frame_per_title_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = [
                ClassifiedFrame(
                    path=root / "a.png",
                    timestamp=4.0,
                    verdict="SLIDE",
                    title="Questions for Business Leaders",
                    build_stage="partial",
                    layout="split-right",
                ),
                ClassifiedFrame(
                    path=root / "b.png",
                    timestamp=8.0,
                    verdict="SLIDE",
                    title="Questions for Business Leader",
                    build_stage="full",
                    layout="full-screen",
                ),
                ClassifiedFrame(
                    path=root / "c.png",
                    timestamp=20.0,
                    verdict="NOT SLIDE",
                ),
            ]

            selected = select_best_unique_slides(frames)

        self.assertEqual([8.0], [frame.timestamp for frame in selected])

    def test_select_best_unique_slides_keeps_best_partial_when_no_full_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = [
                ClassifiedFrame(
                    path=root / "a.png",
                    timestamp=4.0,
                    verdict="SLIDE",
                    title="Market Map",
                    build_stage="transitioning",
                    layout="full-screen",
                ),
                ClassifiedFrame(
                    path=root / "b.png",
                    timestamp=8.0,
                    verdict="SLIDE",
                    title="Market Map",
                    build_stage="partial",
                    layout="split-left",
                ),
            ]

            selected = select_best_unique_slides(frames)

        self.assertEqual([8.0], [frame.timestamp for frame in selected])

    def test_nearby_title_drift_merges_without_not_slide_between(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = [
                ClassifiedFrame(
                    path=root / "a.png",
                    timestamp=10.0,
                    verdict="SLIDE",
                    title="Four Pillars of AI Strategy",
                    build_stage="partial",
                    layout="full-screen",
                ),
                ClassifiedFrame(
                    path=root / "b.png",
                    timestamp=16.0,
                    verdict="SLIDE",
                    title="Vision goals and outcomes",
                    build_stage="full",
                    layout="full-screen",
                ),
            ]

            selected = select_best_unique_slides(frames)

        self.assertEqual([16.0], [frame.timestamp for frame in selected])


if __name__ == "__main__":
    unittest.main()
