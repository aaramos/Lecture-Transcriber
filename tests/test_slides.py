import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from lecture_processor.errors import ProcessingError
from lecture_processor.slides import ClassifiedFrame, SlideBackend, SlideExtractor, select_best_unique_slides


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


class SlideExtractorTests(unittest.TestCase):
    def test_opencv_fps_range_validation_clamps(self):
        mock_cv2 = MagicMock()
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = mock_capture

        # Test with FPS = 0
        mock_capture.get.return_value = 0.0
        extractor = SlideExtractor(backend=SlideBackend.OPENCV)

        with patch("importlib.import_module", return_value=mock_cv2), tempfile.TemporaryDirectory() as tmp:
            extractor.extract(Path("video.mp4"), Path(tmp))
            # Verify CAP_PROP_FPS was read
            mock_capture.get.assert_any_call(mock_cv2.CAP_PROP_FPS)

        # Test with FPS = 300 (exceeds 240)
        mock_capture.get.return_value = 300.0
        extractor = SlideExtractor(backend=SlideBackend.OPENCV)

        with patch("importlib.import_module", return_value=mock_cv2), tempfile.TemporaryDirectory() as tmp:
            extractor.extract(Path("video.mp4"), Path(tmp))
            mock_capture.get.assert_any_call(mock_cv2.CAP_PROP_FPS)

    def test_opencv_imwrite_failure_raises_processing_error(self):
        mock_cv2 = MagicMock()
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        # First read succeeds, second fails to terminate loop
        mock_capture.read.side_effect = [(True, "fake_frame"), (False, None)]
        mock_capture.get.return_value = 30.0
        mock_cv2.VideoCapture.return_value = mock_capture
        # imwrite fails
        mock_cv2.imwrite.return_value = False

        extractor = SlideExtractor(backend=SlideBackend.OPENCV)

        with patch("importlib.import_module", return_value=mock_cv2), tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ProcessingError) as context:
                extractor.extract(Path("video.mp4"), Path(tmp))
            self.assertIn("Failed to save slide image", str(context.exception))


if __name__ == "__main__":
    unittest.main()
