import json
import os
import tempfile
import unittest
from pathlib import Path

from lecture_processor.config import AIProviderName, AudioQuality, BatchConfig, RecordingSpeed, TranscriptionEngine
from lecture_processor.control import update_control_file
from lecture_processor.errors import LectureProcessorError, ProcessingStopped
from lecture_processor.models import FileStatus, MediaInfo, TranscriptResult, TranscriptSegment
from lecture_processor.pipeline import (
    BatchProcessor,
    allocate_output_dirs,
    enrich_processed_batch,
    normalized_video_output_path,
)


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

    def normalize(self, source, destination, media_info, recording_speed, audio_quality, stop_requested=None):
        self.calls.append((source, destination, media_info, recording_speed, audio_quality))
        destination.write_text("normalized", encoding="utf-8")
        return destination


class FakeAudioExtractor:
    def __init__(self):
        self.calls = []

    def extract(self, source, destination, media_info, stop_requested=None):
        self.calls.append((source, destination, media_info))
        destination.write_text("audio", encoding="utf-8")
        return destination


class FakeTranscriber:
    def __init__(self, fail_for=None):
        self.fail_for = set(fail_for or [])
        self.inputs = []

    def transcribe(self, media_path):
        self.inputs.append(media_path)
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

    def extract(self, media_path, output_dir, timestamp_scale=1.0, stop_requested=None):
        self.scales.append(timestamp_scale)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "slide_0001_00-00-05.png").write_text("png", encoding="utf-8")
        return 1


class StopDuringNormalizer:
    def __init__(self, stop_for=None):
        self.stop_for = set(stop_for or [])

    def normalize(self, source, destination, media_info, recording_speed, audio_quality, stop_requested=None):
        if source.name not in self.stop_for:
            destination.write_text("normalized", encoding="utf-8")
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        (destination.parent / "partial.txt").write_text("partial", encoding="utf-8")
        destination.write_text("partial video", encoding="utf-8")
        raise ProcessingStopped("Stopped by user")


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
                audio_extractor=FakeAudioExtractor(),
                transcriber=FakeTranscriber(),
                slide_extractor=FakeSlideExtractor(),
            ).run()

            self.assertEqual(summary.skipped, 1)
            self.assertEqual(summary.results[0].status, FileStatus.SKIPPED)
            self.assertTrue((output / "tiny" / "processing_log.txt").exists())
            self.assertTrue((output / "tiny" / "lecture.json").exists())

    def test_user_skipped_files_do_not_create_output_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["skip.mov", "keep.mov"]:
                (root / name).write_text("video", encoding="utf-8")
            output = root / "out"
            control_file = output / ".lecture_processor_control.json"
            update_control_file(control_file, skip=["skip.mov"])
            events = []
            normalizer = FakeNormalizer()
            config = BatchConfig(
                input_dir=root,
                output_dir=output,
                control_file=control_file,
                recording_speed=RecordingSpeed.DOUBLE,
                confirm_normalization=True,
            )

            summary = BatchProcessor(
                config=config,
                inspector=FakeInspector({"skip.mov": 120.0, "keep.mov": 120.0}),
                normalizer=normalizer,
                audio_extractor=FakeAudioExtractor(),
                transcriber=FakeTranscriber(),
                slide_extractor=FakeSlideExtractor(),
                progress_callback=events.append,
            ).run()

            self.assertEqual(summary.completed, 1)
            self.assertEqual(summary.skipped, 1)
            self.assertEqual([call[0].name for call in normalizer.calls], ["keep.mov"])
            self.assertFalse((output / "skip").exists())
            skipped_events = [event for event in events if event.get("source") == "skip.mov"]
            self.assertEqual([event["kind"] for event in skipped_events], ["file_finished"])
            self.assertEqual(skipped_events[0]["status"], "skipped")

    def test_completed_outputs_are_skipped_without_deleting_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "lecture.mov"
            source.write_text("video", encoding="utf-8")
            output = root / "out"
            lecture_dir = output / "lecture"
            html_dir = lecture_dir / "html"
            html_dir.mkdir(parents=True)
            (html_dir / "index.html").write_text("html", encoding="utf-8")
            (lecture_dir / "lecture.json").write_text(
                json.dumps(
                    {
                        "source": {"filename": "lecture.mov"},
                        "media": {"duration_seconds": 120},
                        "transcript": {"text": "hello lecture", "word_count": 2},
                        "slides": [{"id": 1}],
                        "processing": {"status": "completed"},
                        "enrichment": {"title": "Existing", "executive_summary": "Already done."},
                    }
                ),
                encoding="utf-8",
            )
            config = BatchConfig(input_dir=root, output_dir=output)

            summary = BatchProcessor(
                config=config,
                inspector=FakeInspector({}),
                normalizer=FakeNormalizer(),
                audio_extractor=FakeAudioExtractor(),
                transcriber=FakeTranscriber(),
                slide_extractor=FakeSlideExtractor(),
            ).run()

            self.assertEqual(summary.completed, 0)
            self.assertEqual(summary.skipped, 1)
            self.assertEqual(summary.results[0].message, "Already processed")
            self.assertTrue((lecture_dir / "lecture.json").exists())
            self.assertEqual(summary.results[0].word_count, 2)
            self.assertEqual(summary.results[0].slide_count, 1)

    def test_user_stopped_file_removes_partial_output_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["stop.mov", "keep.mov"]:
                (root / name).write_text("video", encoding="utf-8")
            output = root / "out"
            config = BatchConfig(
                input_dir=root,
                output_dir=output,
                recording_speed=RecordingSpeed.DOUBLE,
                confirm_normalization=True,
                concurrent_files=1,
            )

            summary = BatchProcessor(
                config=config,
                inspector=FakeInspector({"stop.mov": 120.0, "keep.mov": 120.0}),
                normalizer=StopDuringNormalizer(stop_for={"stop.mov"}),
                audio_extractor=FakeAudioExtractor(),
                transcriber=FakeTranscriber(),
                slide_extractor=FakeSlideExtractor(),
            ).run()

            self.assertEqual(summary.stopped, 1)
            self.assertEqual(summary.completed, 1)
            self.assertEqual(summary.failed, 0)
            self.assertFalse((output / "stop").exists())
            self.assertTrue((output / "keep" / "keep.mp4").exists())

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
                audio_extractor=FakeAudioExtractor(),
                transcriber=FakeTranscriber(),
                slide_extractor=slides,
            ).run()

            self.assertEqual(summary.completed, 1)
            self.assertEqual(slides.scales, [0.5])
            srt = (output / "lecture" / "transcript.srt").read_text(encoding="utf-8")
            self.assertIn("00:00:05,000 --> 00:00:10,000", srt)
            self.assertFalse((output / "lecture" / ".normalized_work.mp4").exists())
            self.assertFalse((output / "lecture" / "normalized_video.mp4").exists())
            self.assertTrue((output / "lecture" / "lecture.json").exists())
            self.assertTrue((output / "lecture" / "html" / "index.html").exists())
            self.assertTrue((output / "batch.json").exists())
            self.assertTrue((output / "index.html").exists())

    def test_transcription_uses_clean_temporary_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "lecture.mov"
            source.write_text("video", encoding="utf-8")
            output = root / "out"
            audio_extractor = FakeAudioExtractor()
            transcriber = FakeTranscriber()
            config = BatchConfig(input_dir=root, output_dir=output)

            summary = BatchProcessor(
                config=config,
                inspector=FakeInspector({"lecture.mov": 120.0}),
                normalizer=FakeNormalizer(),
                audio_extractor=audio_extractor,
                transcriber=transcriber,
                slide_extractor=FakeSlideExtractor(),
            ).run()

            self.assertEqual(summary.completed, 1)
            self.assertEqual(audio_extractor.calls[0][1].name, ".transcription_audio.wav")
            self.assertEqual(transcriber.inputs[0].name, ".transcription_audio.wav")
            self.assertFalse((output / "lecture" / ".transcription_audio.wav").exists())
            log = (output / "lecture" / "processing_log.txt").read_text(encoding="utf-8")
            self.assertIn("Audio", log)
            self.assertIn("16 kHz mono WAV", log)

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
                audio_extractor=FakeAudioExtractor(),
                transcriber=FakeTranscriber(fail_for={"lecture"}),
                slide_extractor=FakeSlideExtractor(),
            ).run()

            self.assertEqual(summary.failed, 1)
            self.assertFalse((output / "lecture" / ".normalized_work.mp4").exists())
            self.assertFalse((output / "lecture" / ".transcription_audio.wav").exists())
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
                audio_extractor=FakeAudioExtractor(),
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
                audio_extractor=FakeAudioExtractor(),
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
            (orphan / ".transcription_audio.wav").write_text("temporary", encoding="utf-8")
            (orphan / ".transcript.txt.tmp").write_text("temporary", encoding="utf-8")
            (orphan / ".lecture.mp4.ffmpeg.tmp").write_text("temporary", encoding="utf-8")
            config = BatchConfig(input_dir=root, output_dir=output)

            BatchProcessor(
                config=config,
                inspector=FakeInspector({"lecture.mov": 120.0}),
                normalizer=FakeNormalizer(),
                audio_extractor=FakeAudioExtractor(),
                transcriber=FakeTranscriber(),
                slide_extractor=FakeSlideExtractor(),
            ).run()

            self.assertFalse((orphan / ".normalized_work.mp4").exists())
            self.assertFalse((orphan / ".transcription_audio.wav").exists())
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
                    audio_extractor=FakeAudioExtractor(),
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
                audio_extractor=FakeAudioExtractor(),
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
                audio_extractor=FakeAudioExtractor(),
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

    def test_mock_enrichment_adds_ai_title_and_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "lecture.mov"
            source.write_text("video", encoding="utf-8")
            output = root / "out"
            config = BatchConfig(input_dir=root, output_dir=output, ai_provider=AIProviderName.MOCK)

            summary = BatchProcessor(
                config=config,
                inspector=FakeInspector({"lecture.mov": 120.0}),
                normalizer=FakeNormalizer(),
                audio_extractor=FakeAudioExtractor(),
                transcriber=FakeTranscriber(),
                slide_extractor=FakeSlideExtractor(),
            ).run()

            self.assertEqual(summary.completed, 1)
            self.assertTrue(summary.results[0].enriched)
            lecture_json = json.loads((output / "lecture" / "lecture.json").read_text(encoding="utf-8"))
            self.assertEqual(lecture_json["enrichment"]["provider"], "mock")
            self.assertIn("Hello Lecture", lecture_json["enrichment"]["title"])
            self.assertIn("formatted_transcript", lecture_json["enrichment"])
            html = (output / "lecture" / "html" / "index.html").read_text(encoding="utf-8")
            self.assertIn("AI Study Notes", html)
            self.assertIn("Lecture Transcript", html)
            batch = json.loads((output / "batch.json").read_text(encoding="utf-8"))
            self.assertEqual(batch["summary"]["enriched"], 1)

    def test_processed_batch_enrichment_skips_already_enhanced_lectures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pending_dir = root / "pending"
            enhanced_dir = root / "enhanced"
            pending_dir.mkdir()
            enhanced_dir.mkdir()
            base_artifact = {
                "lecture_id": "lecture",
                "source": {"filename": "lecture.mov"},
                "media": {"duration_seconds": 120},
                "transcript": {"text": "hello lecture", "word_count": 2, "segments": []},
                "slides": [],
                "processing": {"status": "completed"},
                "enrichment": None,
            }
            (pending_dir / "lecture.json").write_text(json.dumps(base_artifact), encoding="utf-8")
            enhanced_artifact = {
                **base_artifact,
                "lecture_id": "enhanced",
                "source": {"filename": "enhanced.mov"},
                "enrichment": {"title": "Enhanced", "executive_summary": "Already enhanced."},
            }
            (enhanced_dir / "lecture.json").write_text(json.dumps(enhanced_artifact), encoding="utf-8")
            events = []
            config = BatchConfig(
                input_dir=root,
                output_dir=root,
                transcription_engine=TranscriptionEngine.NONE,
                ai_provider=AIProviderName.MOCK,
            )

            summary = enrich_processed_batch(config, progress_callback=events.append)

            self.assertEqual(summary.completed, 1)
            self.assertEqual(summary.skipped, 1)
            self.assertIn("Already enhanced", [result.message for result in summary.results])
            pending = json.loads((pending_dir / "lecture.json").read_text(encoding="utf-8"))
            self.assertEqual(pending["enrichment"]["provider"], "mock")
            self.assertTrue((pending_dir / "html" / "index.html").exists())
            self.assertTrue((root / "index.html").exists())
            self.assertIn("batch_finished", [event["kind"] for event in events])


if __name__ == "__main__":
    unittest.main()
