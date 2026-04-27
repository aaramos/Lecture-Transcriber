import json
import os
import tempfile
import unittest
from pathlib import Path

from lecture_processor.config import AudioQuality, BatchConfig, RecordingSpeed
from lecture_processor.errors import LectureProcessorError
from lecture_processor.models import FileStatus, MediaInfo, TranscriptResult, TranscriptSegment
from lecture_processor.pipeline import BatchProcessor, allocate_output_dirs, normalized_video_output_path


class FakeInspector:
    def __init__(self, durations):
        self.durations = durations

    def probe(self, path):
        return MediaInfo(
            path=path,
            duration_seconds=self.durations[path.name],
            avg_frame_rate=30.0,
            real_frame_rate=30.0,
            is_vfr=False,
        )


class FakeNormalizer:
    def __init__(self):
        self.calls = []

    def normalize(self, source, destination, media_info, recording_speed, audio_quality):
        self.calls.append((source, destination, media_info, recording_speed, audio_quality))
        destination.write_text("normalized", encoding="utf-8")
        return destination


class FakeTranscriber:
    def __init__(self, fail_for=None):
        self.fail_for = set(fail_for or [])

    def transcribe(self, media_path):
        if (
            media_path.name in self.fail_for
            or media_path.stem in self.fail_for
            or media_path.parent.name in self.fail_for
        ):
            raise RuntimeError("Whisper returned empty transcript")
        return TranscriptResult(
            text="hello lecture",
            segments=[
                TranscriptSegment(start=10.0, end=20.0, text="hello lecture"),
            ],
        )


class FakeSlideExtractor:
    def __init__(self):
        self.scales = []

    def extract(self, media_path, output_dir, timestamp_scale=1.0):
        self.scales.append(timestamp_scale)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "slide_0001_00-00-05.png").write_text("png", encoding="utf-8")
        return 1


class BatchProcessorTests(unittest.TestCase):
    def test_allocates_unique_output_dirs_after_sanitizing_names(self):
        root = Path("/tmp/out")
        files = [Path("Lecture 1.mov"), Path("Lecture:1.mov")]

        allocated = allocate_output_dirs(files, root)

        self.assertEqual(allocated[Path("Lecture 1.mov")], root / "Lecture_1")
        self.assertEqual(allocated[Path("Lecture:1.mov")], root / "Lecture_1_2")

    def test_skips_short_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "tiny.mov"
            source.write_text("video", encoding="utf-8")
            output = root / "out"
            config = BatchConfig(input_dir=root, output_dir=output)

            summary = BatchProcessor(
                config=config,
                inspector=FakeInspector({"tiny.mov": 18.0}),
                normalizer=FakeNormalizer(),
                transcriber=FakeTranscriber(),
                slide_extractor=FakeSlideExtractor(),
            ).run()

            self.assertEqual(summary.skipped, 1)
            self.assertEqual(summary.results[0].status, FileStatus.SKIPPED)
            self.assertTrue((output / "tiny" / "processing_log.txt").exists())

    def test_2x_without_saved_normalized_video_scales_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "lecture.mov"
            source.write_text("video", encoding="utf-8")
            output = root / "out"
            slides = FakeSlideExtractor()
            config = BatchConfig(
                input_dir=root,
                output_dir=output,
                recording_speed=RecordingSpeed.DOUBLE,
                confirm_normalization=True,
                save_normalized_video=False,
                audio_quality=AudioQuality.FAST,
            )

            summary = BatchProcessor(
                config=config,
                inspector=FakeInspector({"lecture.mov": 120.0}),
                normalizer=FakeNormalizer(),
                transcriber=FakeTranscriber(),
                slide_extractor=slides,
            ).run()

            self.assertEqual(summary.completed, 1)
            self.assertEqual(slides.scales, [0.5])
            srt = (output / "lecture" / "transcript.srt").read_text(encoding="utf-8")
            self.assertIn("00:00:05,000 --> 00:00:10,000", srt)
            self.assertFalse((output / "lecture" / ".normalized_work.mp4").exists())
            self.assertFalse((output / "lecture" / "normalized_video.mp4").exists())

    def test_temp_normalized_video_is_removed_after_downstream_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "lecture.mov"
            source.write_text("video", encoding="utf-8")
            output = root / "out"
            config = BatchConfig(
                input_dir=root,
                output_dir=output,
                recording_speed=RecordingSpeed.DOUBLE,
                confirm_normalization=True,
                save_normalized_video=False,
            )

            summary = BatchProcessor(
                config=config,
                inspector=FakeInspector({"lecture.mov": 120.0}),
                normalizer=FakeNormalizer(),
                transcriber=FakeTranscriber(fail_for={"lecture"}),
                slide_extractor=FakeSlideExtractor(),
            ).run()

            self.assertEqual(summary.failed, 1)
            self.assertFalse((output / "lecture" / ".normalized_work.mp4").exists())
            self.assertTrue((output / "lecture" / "processing_log.txt").exists())

    def test_saved_normalized_video_keeps_source_lecture_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "Lecture 1: Intro.mov"
            source.write_text("video", encoding="utf-8")
            output = root / "out"
            config = BatchConfig(
                input_dir=root,
                output_dir=output,
                recording_speed=RecordingSpeed.DOUBLE,
                confirm_normalization=True,
            )

            BatchProcessor(
                config=config,
                inspector=FakeInspector({"Lecture 1: Intro.mov": 120.0}),
                normalizer=FakeNormalizer(),
                transcriber=FakeTranscriber(),
                slide_extractor=FakeSlideExtractor(),
            ).run()

            self.assertTrue((output / "Lecture_1_Intro" / "Lecture_1_Intro.mp4").exists())
            self.assertFalse((output / "Lecture_1_Intro" / "normalized_video.mp4").exists())

    def test_failure_preserves_partial_output_and_batch_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["bad.mov", "good.mov"]:
                (root / name).write_text("video", encoding="utf-8")
            output = root / "out"
            config = BatchConfig(
                input_dir=root,
                output_dir=output,
                recording_speed=RecordingSpeed.DOUBLE,
                confirm_normalization=True,
            )

            summary = BatchProcessor(
                config=config,
                inspector=FakeInspector({"bad.mov": 120.0, "good.mov": 120.0}),
                normalizer=FakeNormalizer(),
                transcriber=FakeTranscriber(fail_for={"bad"}),
                slide_extractor=FakeSlideExtractor(),
            ).run()

            self.assertEqual(summary.attempted, 2)
            self.assertEqual(summary.failed, 1)
            self.assertEqual(summary.completed, 1)
            bad_dir = output / "bad"
            self.assertTrue((bad_dir / "processing_log.txt").exists())
            self.assertTrue((bad_dir / "bad.mp4").exists())
            self.assertFalse((bad_dir / "transcript.txt").exists())
            log = (bad_dir / "processing_log.txt").read_text(encoding="utf-8")
            self.assertIn("Partial output preserved for review", log)

    def test_startup_cleanup_removes_stale_temp_files_in_output_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "lecture.mov"
            source.write_text("video", encoding="utf-8")
            output = root / "out"
            orphan = output / "old-run"
            orphan.mkdir(parents=True)
            (orphan / ".normalized_work.mp4").write_text("temporary", encoding="utf-8")
            (orphan / ".transcript.txt.tmp").write_text("temporary", encoding="utf-8")
            (orphan / ".lecture.mp4.ffmpeg.tmp").write_text("temporary", encoding="utf-8")
            config = BatchConfig(input_dir=root, output_dir=output)

            BatchProcessor(
                config=config,
                inspector=FakeInspector({"lecture.mov": 120.0}),
                normalizer=FakeNormalizer(),
                transcriber=FakeTranscriber(),
                slide_extractor=FakeSlideExtractor(),
            ).run()

            self.assertFalse((orphan / ".normalized_work.mp4").exists())
            self.assertFalse((orphan / ".transcript.txt.tmp").exists())
            self.assertFalse((orphan / ".lecture.mp4.ffmpeg.tmp").exists())

    def test_output_lock_blocks_second_batch_using_same_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "lecture.mov"
            source.write_text("video", encoding="utf-8")
            output = root / "out"
            output.mkdir()
            (output / ".lecture_processor.lock").write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
            config = BatchConfig(input_dir=root, output_dir=output)

            with self.assertRaises(LectureProcessorError) as error:
                BatchProcessor(
                    config=config,
                    inspector=FakeInspector({"lecture.mov": 120.0}),
                    normalizer=FakeNormalizer(),
                    transcriber=FakeTranscriber(),
                    slide_extractor=FakeSlideExtractor(),
                ).run()

            self.assertIn("already being processed", str(error.exception))

    def test_output_lock_replaces_stale_lock_and_clears_after_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "lecture.mov"
            source.write_text("video", encoding="utf-8")
            output = root / "out"
            output.mkdir()
            (output / ".lecture_processor.lock").write_text(json.dumps({"pid": -1}), encoding="utf-8")
            config = BatchConfig(input_dir=root, output_dir=output)

            summary = BatchProcessor(
                config=config,
                inspector=FakeInspector({"lecture.mov": 120.0}),
                normalizer=FakeNormalizer(),
                transcriber=FakeTranscriber(),
                slide_extractor=FakeSlideExtractor(),
            ).run()

            self.assertEqual(summary.completed, 1)
            self.assertFalse((output / ".lecture_processor.lock").exists())

    def test_normalized_video_output_path_uses_portable_source_name(self):
        path = normalized_video_output_path(Path("Lecture: 2.mov"), Path("/tmp/out"))

        self.assertEqual(path, Path("/tmp/out") / "Lecture_2.mp4")

    def test_emits_batch_and_file_progress_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "lecture.mov"
            source.write_text("video", encoding="utf-8")
            output = root / "out"
            events = []
            config = BatchConfig(input_dir=root, output_dir=output)

            BatchProcessor(
                config=config,
                inspector=FakeInspector({"lecture.mov": 120.0}),
                normalizer=FakeNormalizer(),
                transcriber=FakeTranscriber(),
                slide_extractor=FakeSlideExtractor(),
                progress_callback=events.append,
            ).run()

            kinds = [event["kind"] for event in events]
            self.assertIn("batch_started", kinds)
            self.assertIn("file_started", kinds)
            self.assertIn("step_started", kinds)
            self.assertIn("file_finished", kinds)
            self.assertEqual(events[-1]["kind"], "batch_finished")
            finished = [event for event in events if event["kind"] == "file_finished"][0]
            self.assertEqual(finished["status"], "completed")


if __name__ == "__main__":
    unittest.main()
