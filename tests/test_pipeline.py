import json
import os
import threading
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
    discover_mov_files,
    enrich_processed_batch,
    normalized_video_output_path,
)


class FakeInspector:
    def __init__(self, durations):
        self.durations = durations

    def probe(self, path):
        value = self.durations[path.name]
        if isinstance(value, dict):
            duration = value.get("duration", 120.0)
            has_audio = value.get("has_audio", True)
            has_video = value.get("has_video", True)
        else:
            duration = value
            has_audio = True
            has_video = True
        return MediaInfo(
            path=path,
            duration_seconds=duration,
            avg_frame_rate=30.0,
            real_frame_rate=30.0,
            is_vfr=False,
            has_audio=has_audio,
            has_video=has_video,
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
        self.transcription_calls = []
        self.last_command_text = "ffmpeg -i fake -vn -ac 1 -ar 16000 fake.wav"

    def extract(self, source, destination, media_info, stop_requested=None):
        self.calls.append((source, destination, media_info))
        destination.write_text("audio", encoding="utf-8")
        return destination

    def extract_for_transcription(
        self,
        source,
        destination,
        media_info,
        recording_speed,
        audio_quality,
        stop_requested=None,
    ):
        self.transcription_calls.append((source, destination, media_info, recording_speed, audio_quality))
        if recording_speed is RecordingSpeed.NORMAL:
            return self.extract(source, destination, media_info, stop_requested=stop_requested)
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


class ParallelBarrierNormalizer:
    def __init__(self, normalize_started, audio_started):
        self.normalize_started = normalize_started
        self.audio_started = audio_started
        self.calls = []

    def normalize(self, source, destination, media_info, recording_speed, audio_quality, stop_requested=None):
        self.calls.append((source, destination, media_info, recording_speed, audio_quality))
        self.normalize_started.set()
        if not self.audio_started.wait(timeout=1):
            raise RuntimeError("Audio did not start while Normalize was still running")
        destination.write_text("normalized", encoding="utf-8")
        return destination


class ParallelBarrierAudioExtractor(FakeAudioExtractor):
    def __init__(self, normalize_started, audio_started):
        super().__init__()
        self.normalize_started = normalize_started
        self.audio_started = audio_started

    def extract_for_transcription(
        self,
        source,
        destination,
        media_info,
        recording_speed,
        audio_quality,
        stop_requested=None,
    ):
        self.transcription_calls.append((source, destination, media_info, recording_speed, audio_quality))
        self.audio_started.set()
        if not self.normalize_started.wait(timeout=1):
            raise RuntimeError("Normalize did not start while Audio was still running")
        destination.write_text("audio", encoding="utf-8")
        return destination


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
    def test_discovers_supported_sources_in_folder_and_single_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["lecture.mov", "lecture.mp4", "lecture.mkv", "audio.mp3", "captions.srt", "notes.txt"]:
                (root / name).write_text("source", encoding="utf-8")
            (root / "ignore.pdf").write_text("ignore", encoding="utf-8")

            discovered = [path.name for path in discover_mov_files(root)]

            self.assertEqual(
                discovered,
                ["audio.mp3", "captions.srt", "lecture.mkv", "lecture.mov", "lecture.mp4", "notes.txt"],
            )
            self.assertEqual(discover_mov_files(root / "lecture.mp4"), [root / "lecture.mp4"])
            self.assertEqual(discover_mov_files(root / "ignore.pdf"), [])

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

    def test_audio_file_uses_transcription_without_slide_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "lecture.mp3"
            source.write_text("audio", encoding="utf-8")
            output = root / "out"
            slide_extractor = FakeSlideExtractor()

            summary = BatchProcessor(
                config=BatchConfig(input_dir=root, output_dir=output),
                inspector=FakeInspector({"lecture.mp3": {"duration": 120.0, "has_audio": True, "has_video": False}}),
                normalizer=FakeNormalizer(),
                audio_extractor=FakeAudioExtractor(),
                transcriber=FakeTranscriber(),
                slide_extractor=slide_extractor,
            ).run()

            self.assertEqual(summary.completed, 1)
            self.assertEqual(summary.results[0].slide_count, 0)
            self.assertFalse((output / "lecture" / "slides").exists())
            self.assertEqual(slide_extractor.scales, [])

    def test_transcript_file_imports_without_transcriber(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "lecture.srt"
            source.write_text(
                "1\n00:00:01,000 --> 00:00:03,500\nWelcome to class.\n",
                encoding="utf-8",
            )
            output = root / "out"

            summary = BatchProcessor(
                config=BatchConfig(input_dir=root, output_dir=output),
                inspector=FakeInspector({}),
                normalizer=FakeNormalizer(),
                audio_extractor=FakeAudioExtractor(),
                transcriber=None,
                slide_extractor=FakeSlideExtractor(),
            ).run()

            artifact = json.loads((output / "lecture" / "lecture.json").read_text(encoding="utf-8"))
            self.assertEqual(summary.completed, 1)
            self.assertEqual(summary.results[0].word_count, 3)
            self.assertEqual(summary.results[0].slide_count, 0)
            self.assertEqual(artifact["transcript"]["engine"], "import")
            self.assertEqual(artifact["media"]["duration_seconds"], 3.5)
            self.assertFalse(artifact["media"]["has_video"])

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
            audio_extractor = FakeAudioExtractor()
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
                audio_extractor=audio_extractor,
                transcriber=FakeTranscriber(),
                slide_extractor=slides,
            ).run()

            self.assertEqual(summary.completed, 1)
            self.assertEqual(audio_extractor.transcription_calls[0][0], source)
            self.assertEqual(audio_extractor.transcription_calls[0][3], RecordingSpeed.DOUBLE)
            self.assertEqual(audio_extractor.calls, [])
            self.assertEqual(slides.scales, [0.5])
            srt = (output / "lecture" / "transcript.srt").read_text(encoding="utf-8")
            self.assertIn("00:00:05,000 --> 00:00:10,000", srt)
            self.assertFalse((output / "lecture" / ".normalized_work.mp4").exists())
            self.assertFalse((output / "lecture" / "normalized_video.mp4").exists())
            self.assertTrue((output / "lecture" / "lecture.json").exists())
            self.assertTrue((output / "lecture" / "html" / "index.html").exists())
            self.assertTrue((output / "batch.json").exists())
            self.assertTrue((output / "index.html").exists())

    def test_2x_saved_normalized_video_keeps_transcript_and_slides_on_1x_timeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "lecture.mov"
            source.write_text("video", encoding="utf-8")
            output = root / "out"
            slides = FakeSlideExtractor()
            audio_extractor = FakeAudioExtractor()
            config = BatchConfig(
                input_dir=root,
                output_dir=output,
                recording_speed=RecordingSpeed.DOUBLE,
                confirm_normalization=True,
                save_normalized_video=True,
                audio_quality=AudioQuality.FAST,
            )

            summary = BatchProcessor(
                config=config,
                inspector=FakeInspector({"lecture.mov": 120.0}),
                normalizer=FakeNormalizer(),
                audio_extractor=audio_extractor,
                transcriber=FakeTranscriber(),
                slide_extractor=slides,
            ).run()

            self.assertEqual(summary.completed, 1)
            self.assertEqual(audio_extractor.transcription_calls[0][0], source)
            self.assertEqual(audio_extractor.transcription_calls[0][3], RecordingSpeed.DOUBLE)
            self.assertEqual(slides.scales, [1.0])
            srt = (output / "lecture" / "transcript.srt").read_text(encoding="utf-8")
            self.assertIn("00:00:10,000 --> 00:00:20,000", srt)
            self.assertTrue((output / "lecture" / "lecture.mp4").exists())

    def test_2x_runs_normalize_and_audio_in_parallel_before_transcription(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "lecture.mov"
            source.write_text("video", encoding="utf-8")
            output = root / "out"
            events = []
            normalize_started = threading.Event()
            audio_started = threading.Event()
            normalizer = ParallelBarrierNormalizer(normalize_started, audio_started)
            audio_extractor = ParallelBarrierAudioExtractor(normalize_started, audio_started)
            transcriber = FakeTranscriber()
            config = BatchConfig(
                input_dir=root,
                output_dir=output,
                recording_speed=RecordingSpeed.DOUBLE,
                confirm_normalization=True,
                concurrent_files=1,
            )

            summary = BatchProcessor(
                config=config,
                inspector=FakeInspector({"lecture.mov": 120.0}),
                normalizer=normalizer,
                audio_extractor=audio_extractor,
                transcriber=transcriber,
                slide_extractor=FakeSlideExtractor(),
                progress_callback=events.append,
            ).run()

            self.assertEqual(summary.completed, 1)
            self.assertTrue(normalize_started.is_set())
            self.assertTrue(audio_started.is_set())
            self.assertEqual(transcriber.inputs[0].name, ".transcription_audio.wav")
            timeline = [
                (event["kind"], event["step"])
                for event in events
                if event["kind"] in {"step_started", "step_finished"}
            ]
            self.assertLess(timeline.index(("step_started", "Normalize")), timeline.index(("step_finished", "Normalize")))
            self.assertLess(timeline.index(("step_started", "Audio")), timeline.index(("step_finished", "Normalize")))
            log = (output / "lecture" / "processing_log.txt").read_text(encoding="utf-8")
            self.assertIn("Normalize  OK", log)
            self.assertIn("Audio      OK", log)

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
            self.assertIn("Profile:  quality (faster-whisper, medium.en)", log)
            self.assertIn("Audio", log)
            self.assertIn("16 kHz mono WAV", log)
            self.assertIn("Audio cmd: ffmpeg -i fake -vn -ac 1 -ar 16000 fake.wav", log)

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
            self.assertIn("file_media", kinds)
            self.assertIn("step_started", kinds)
            self.assertIn("file_finished", kinds)
            self.assertEqual(events[-1]["kind"], "batch_finished")
            started = [event for event in events if event["kind"] == "batch_started"][0]
            self.assertEqual(started["transcription_profile"], "quality")
            self.assertEqual(started["transcription_engine"], "faster-whisper")
            media = [event for event in events if event["kind"] == "file_media"][0]
            self.assertEqual(media["duration_seconds"], 120.0)
            finished = [event for event in events if event["kind"] == "file_finished"][0]
            self.assertEqual(finished["status"], "completed")
            self.assertEqual(finished["duration_seconds"], 120.0)
            self.assertGreater(finished["elapsed_seconds"], 0)

    def test_2x_progress_uses_finished_lecture_duration_for_length_and_speed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "lecture.mov"
            source.write_text("video", encoding="utf-8")
            output = root / "out"
            events = []
            config = BatchConfig(
                input_dir=root,
                output_dir=output,
                recording_speed=RecordingSpeed.DOUBLE,
                confirm_normalization=True,
                concurrent_files=1,
            )

            BatchProcessor(
                config=config,
                inspector=FakeInspector({"lecture.mov": 120.0}),
                normalizer=FakeNormalizer(),
                audio_extractor=FakeAudioExtractor(),
                transcriber=FakeTranscriber(),
                slide_extractor=FakeSlideExtractor(),
                progress_callback=events.append,
            ).run()

            media = [event for event in events if event["kind"] == "file_media"][0]
            self.assertEqual(media["source_duration_seconds"], 120.0)
            self.assertEqual(media["duration_seconds"], 240.0)
            finished = [event for event in events if event["kind"] == "file_finished"][0]
            self.assertEqual(finished["source_duration_seconds"], 120.0)
            self.assertEqual(finished["duration_seconds"], 240.0)
            self.assertEqual(finished["lecture_duration_seconds"], 240.0)
            self.assertEqual(finished["normalized_duration_seconds"], 240.0)
            batch = json.loads((output / "batch.json").read_text(encoding="utf-8"))
            self.assertEqual(batch["lectures"][0]["duration_minutes"], 4.0)

    def test_mock_enrichment_adds_ai_title_and_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "lecture.mov"
            source.write_text("video", encoding="utf-8")
            output = root / "out"
            events = []
            config = BatchConfig(input_dir=root, output_dir=output, ai_provider=AIProviderName.MOCK)

            summary = BatchProcessor(
                config=config,
                inspector=FakeInspector({"lecture.mov": 120.0}),
                normalizer=FakeNormalizer(),
                audio_extractor=FakeAudioExtractor(),
                transcriber=FakeTranscriber(),
                slide_extractor=FakeSlideExtractor(),
                progress_callback=events.append,
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
            enrichment_finished = [event for event in events if event["kind"] == "enrichment_finished"][0]
            self.assertGreater(enrichment_finished["input_tokens"], 0)
            self.assertGreater(enrichment_finished["output_tokens"], 0)
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

    def test_processed_batch_enrichment_retries_enrich_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lecture_dir = root / "failed_enrich"
            lecture_dir.mkdir()
            (lecture_dir / "lecture.json").write_text(
                json.dumps(
                    {
                        "lecture_id": "failed_enrich",
                        "source": {"filename": "failed_enrich.mov"},
                        "media": {"duration_seconds": 120},
                        "transcript": {"text": "hello lecture", "word_count": 2, "segments": []},
                        "slides": [],
                        "processing": {
                            "status": "failed",
                            "failure_step": "Enrich",
                            "failure_message": "Gemini returned an empty response.",
                        },
                        "enrichment": None,
                    }
                ),
                encoding="utf-8",
            )
            config = BatchConfig(
                input_dir=root,
                output_dir=root,
                transcription_engine=TranscriptionEngine.NONE,
                ai_provider=AIProviderName.MOCK,
            )

            summary = enrich_processed_batch(config)

            self.assertEqual(summary.completed, 1)
            artifact = json.loads((lecture_dir / "lecture.json").read_text(encoding="utf-8"))
            self.assertEqual(artifact["enrichment"]["provider"], "mock")
            self.assertTrue((lecture_dir / "html" / "index.html").exists())


if __name__ == "__main__":
    unittest.main()
