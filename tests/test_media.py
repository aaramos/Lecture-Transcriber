import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from shlex import quote as shlex_quote
from subprocess import CompletedProcess

from lecture_processor.config import AudioQuality, FfmpegHwAccel, RecordingSpeed
from lecture_processor.errors import DependencyMissingError, ProcessingError, ProcessingStopped
from lecture_processor.media import (
    CleanAudioExtractor,
    MediaNormalizer,
    _audio_filter_for,
    _clean_audio_filter_for,
    _hwaccel_args,
    _has_audio_stream,
    _require_command,
    _run_interruptible,
    _transcription_audio_filter_for,
    ensure_media_tools,
)
from lecture_processor.models import MediaInfo


class MediaDependencyTests(unittest.TestCase):
    def test_preflight_reports_missing_ffprobe(self):
        with self.assertRaises(DependencyMissingError) as context:
            ensure_media_tools(
                ffprobe_path="definitely-not-ffprobe",
                ffmpeg_path="definitely-not-ffmpeg",
                needs_ffmpeg=False,
            )

        self.assertIn("ffprobe", str(context.exception))

    def test_finds_workspace_local_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tool = root / ".tools" / "darwin_arm64" / "ffmpeg"
            tool.parent.mkdir(parents=True)
            tool.write_text("binary", encoding="utf-8")
            previous = Path.cwd()
            try:
                import os

                os.chdir(root)
                self.assertEqual(Path(_require_command("ffmpeg")).resolve(), tool.resolve())
            finally:
                os.chdir(previous)

    def test_high_quality_falls_back_when_rubberband_filter_is_missing(self):
        audio_filter = _audio_filter_for(
            "ffmpeg",
            AudioQuality.HIGH,
            filter_available=lambda _ffmpeg, _filter_name: False,
        )

        self.assertEqual(audio_filter, "atempo=0.5")

    def test_uses_rubberband_when_high_quality_filter_is_available(self):
        audio_filter = _audio_filter_for(
            "ffmpeg",
            AudioQuality.HIGH,
            filter_available=lambda _ffmpeg, filter_name: filter_name == "rubberband",
        )

        self.assertEqual(audio_filter, "rubberband=tempo=0.5")

    def test_clean_audio_filter_uses_available_speech_filters(self):
        audio_filter = _clean_audio_filter_for(
            "ffmpeg",
            filter_available=lambda _ffmpeg, filter_name: filter_name in {"highpass", "loudnorm"},
        )

        self.assertEqual(audio_filter, "highpass=f=80,loudnorm=I=-18:TP=-2:LRA=11")

    def test_clean_audio_extractor_writes_mono_wav_and_removes_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            ffmpeg = Path(tmp) / "ffmpeg"
            ffmpeg.write_text("binary", encoding="utf-8")
            source = Path(tmp) / "lecture.mp4"
            source.write_text("video", encoding="utf-8")
            destination = Path(tmp) / "out" / ".transcription_audio.wav"
            captured = {}

            def runner(command, capture_output, text):
                captured["command"] = command
                Path(command[-1]).write_text("audio", encoding="utf-8")
                return CompletedProcess(command, 0, "", "")

            CleanAudioExtractor(
                ffmpeg_path=str(ffmpeg),
                runner=runner,
                filter_available=lambda _ffmpeg, filter_name: filter_name == "loudnorm",
            ).extract(
                source=source,
                destination=destination,
                media_info=MediaInfo(path=source, duration_seconds=120.0, has_audio=True),
            )

            command = captured["command"]
            self.assertIn("-vn", command)
            self.assertEqual(command[command.index("-map") + 1], "0:a:0")
            self.assertEqual(command[command.index("-ac") + 1], "1")
            self.assertEqual(command[command.index("-ar") + 1], "16000")
            self.assertEqual(command[command.index("-af") + 1], "loudnorm=I=-18:TP=-2:LRA=11")
            self.assertEqual(command[-3:-1], ["-f", "wav"])
            self.assertTrue(destination.exists())
            self.assertFalse((destination.parent / "..transcription_audio.wav.ffmpeg.tmp").exists())

    def test_extract_for_transcription_keeps_1x_command_equivalent(self):
        with tempfile.TemporaryDirectory() as tmp:
            ffmpeg = Path(tmp) / "ffmpeg"
            ffmpeg.write_text("binary", encoding="utf-8")
            source = Path(tmp) / "lecture.mp4"
            source.write_text("video", encoding="utf-8")
            destination = Path(tmp) / "out" / ".transcription_audio.wav"
            captured = {}

            def runner(command, capture_output, text):
                captured["command"] = command
                Path(command[-1]).write_text("audio", encoding="utf-8")
                return CompletedProcess(command, 0, "", "")

            CleanAudioExtractor(
                ffmpeg_path=str(ffmpeg),
                runner=runner,
                filter_available=lambda _ffmpeg, filter_name: filter_name == "loudnorm",
            ).extract_for_transcription(
                source=source,
                destination=destination,
                media_info=MediaInfo(path=source, duration_seconds=120.0, has_audio=True),
                recording_speed=RecordingSpeed.NORMAL,
                audio_quality=AudioQuality.HIGH,
            )

            command = captured["command"]
            self.assertIn("-vn", command)
            self.assertEqual(command[command.index("-af") + 1], "loudnorm=I=-18:TP=-2:LRA=11")
            self.assertNotIn("atempo=0.5", " ".join(command))
            self.assertNotIn("rubberband=tempo=0.5", " ".join(command))

    def test_extract_for_transcription_2x_puts_speed_filter_before_clean_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            ffmpeg = Path(tmp) / "ffmpeg"
            ffmpeg.write_text("binary", encoding="utf-8")
            source = Path(tmp) / "lecture.mp4"
            source.write_text("video", encoding="utf-8")
            destination = Path(tmp) / "out" / ".transcription_audio.wav"
            captured = {}

            def runner(command, capture_output, text):
                captured["command"] = command
                Path(command[-1]).write_text("audio", encoding="utf-8")
                return CompletedProcess(command, 0, "", "")

            extractor = CleanAudioExtractor(
                ffmpeg_path=str(ffmpeg),
                runner=runner,
                filter_available=lambda _ffmpeg, filter_name: filter_name in {"rubberband", "highpass", "loudnorm"},
            )
            extractor.extract_for_transcription(
                source=source,
                destination=destination,
                media_info=MediaInfo(path=source, duration_seconds=120.0, has_audio=True),
                recording_speed=RecordingSpeed.DOUBLE,
                audio_quality=AudioQuality.HIGH,
            )

            command = captured["command"]
            self.assertIn("-vn", command)
            self.assertEqual(command[command.index("-map") + 1], "0:a:0")
            self.assertEqual(command[command.index("-ac") + 1], "1")
            self.assertEqual(command[command.index("-ar") + 1], "16000")
            self.assertEqual(command[command.index("-c:a") + 1], "pcm_s16le")
            self.assertEqual(command[-3:-1], ["-f", "wav"])
            self.assertEqual(
                command[command.index("-af") + 1],
                "rubberband=tempo=0.5,highpass=f=80,loudnorm=I=-18:TP=-2:LRA=11",
            )
            self.assertEqual(extractor.last_command_text, " ".join(shlex_quote(part) for part in command))

    def test_transcription_audio_filter_uses_atempo_when_rubberband_is_missing(self):
        audio_filter = _transcription_audio_filter_for(
            "ffmpeg",
            RecordingSpeed.DOUBLE,
            AudioQuality.HIGH,
            filter_available=lambda _ffmpeg, filter_name: filter_name in {"highpass", "loudnorm"},
        )

        self.assertEqual(audio_filter, "atempo=0.5,highpass=f=80,loudnorm=I=-18:TP=-2:LRA=11")

    def test_clean_audio_extractor_rejects_video_without_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            ffmpeg = Path(tmp) / "ffmpeg"
            ffmpeg.write_text("binary", encoding="utf-8")

            with self.assertRaises(ProcessingError):
                CleanAudioExtractor(ffmpeg_path=str(ffmpeg)).extract(
                    source=Path("lecture.mp4"),
                    destination=Path(tmp) / ".transcription_audio.wav",
                    media_info=MediaInfo(path=Path("lecture.mp4"), duration_seconds=120.0, has_audio=False),
                )

    def test_real_ffmpeg_transcription_audio_is_mono_16khz_pcm_wav(self):
        ffmpeg = shutil.which("ffmpeg") or str(_require_command("ffmpeg"))
        ffprobe = shutil.which("ffprobe") or str(_require_command("ffprobe"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mov"
            destination = root / "out" / ".transcription_audio.wav"
            create = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=160x90:rate=10:duration=0.5",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=1000:duration=0.5",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    str(source),
                ],
                capture_output=True,
                text=True,
            )
            if create.returncode != 0:
                self.skipTest(f"Could not create ffmpeg fixture: {create.stderr}")

            CleanAudioExtractor(ffmpeg_path=ffmpeg).extract_for_transcription(
                source=source,
                destination=destination,
                media_info=MediaInfo(path=source, duration_seconds=0.5, has_audio=True),
                recording_speed=RecordingSpeed.DOUBLE,
                audio_quality=AudioQuality.FAST,
            )

            probe = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=codec_name,sample_rate,channels",
                    "-of",
                    "json",
                    str(destination),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(probe.returncode, 0, probe.stderr)
            stream = json.loads(probe.stdout)["streams"][0]
            self.assertEqual(stream["codec_name"], "pcm_s16le")
            self.assertEqual(stream["sample_rate"], "16000")
            self.assertEqual(stream["channels"], 1)

    def test_interruptible_runner_stops_child_process(self):
        calls = 0

        def stop_requested():
            nonlocal calls
            calls += 1
            return calls > 1

        with self.assertRaises(ProcessingStopped):
            _run_interruptible(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                subprocess.run,
                stop_requested=stop_requested,
            )

    def test_interruptible_runner_drains_child_output_while_waiting(self):
        completed = _run_interruptible(
            [
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('x' * 200000); sys.stderr.flush(); print('done')",
            ],
            subprocess.run,
            stop_requested=lambda: False,
        )

        self.assertEqual(completed.returncode, 0)
        self.assertIn("done", completed.stdout)
        self.assertLess(len(completed.stderr), 200000)
        self.assertGreater(len(completed.stderr), 0)

    def test_interruptible_runner_fails_when_output_file_stalls(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "partial.mp4"

            completed = _run_interruptible(
                [
                    sys.executable,
                    "-c",
                    (
                        "import pathlib, sys, time; "
                        "pathlib.Path(sys.argv[1]).write_text('partial', encoding='utf-8'); "
                        "time.sleep(10)"
                    ),
                    str(output),
                ],
                subprocess.run,
                stop_requested=lambda: False,
                progress_path=output,
                stall_timeout_seconds=0.2,
                poll_interval_seconds=0.05,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("stalled", completed.stderr)

    def test_normalization_command_handles_video_without_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            ffmpeg = Path(tmp) / "ffmpeg"
            ffmpeg.write_text("binary", encoding="utf-8")
            captured = {}

            def runner(command, capture_output, text):
                captured["command"] = command
                Path(command[-1]).write_text("video", encoding="utf-8")
                return CompletedProcess(command, 0, "", "")

            media_info = MediaInfo(
                path=Path("lecture.mov"),
                duration_seconds=120.0,
                has_audio=False,
            )

            MediaNormalizer(ffmpeg_path=str(ffmpeg), runner=runner).normalize(
                source=Path("lecture.mov"),
                destination=Path(tmp) / "out" / "normalized_video.mp4",
                media_info=media_info,
                recording_speed=RecordingSpeed.DOUBLE,
                audio_quality=AudioQuality.FAST,
            )

            command = captured["command"]
            self.assertIn("-an", command)
            self.assertNotIn("[a]", command)
            self.assertEqual(command[-3:-1], ["-f", "mp4"])
            self.assertTrue((Path(tmp) / "out" / "normalized_video.mp4").exists())
            self.assertFalse((Path(tmp) / "out" / ".normalized_video.mp4.ffmpeg.tmp").exists())

    def test_detects_audio_streams_from_probe_payload(self):
        self.assertTrue(_has_audio_stream({"streams": [{"codec_type": "audio"}]}))
        self.assertFalse(_has_audio_stream({"streams": [{"codec_type": "video"}]}))

    def test_videotoolbox_hwaccel_is_enabled_on_apple_silicon_auto(self):
        self.assertEqual(_hwaccel_args(FfmpegHwAccel.AUTO, apple_silicon=True), ["-hwaccel", "videotoolbox"])
        self.assertEqual(_hwaccel_args(FfmpegHwAccel.AUTO, apple_silicon=False), [])
        self.assertEqual(_hwaccel_args(FfmpegHwAccel.NONE, apple_silicon=True), [])

    def test_normalization_command_includes_videotoolbox_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            ffmpeg = Path(tmp) / "ffmpeg"
            ffmpeg.write_text("binary", encoding="utf-8")
            captured = {}

            def runner(command, capture_output, text):
                captured["command"] = command
                Path(command[-1]).write_text("video", encoding="utf-8")
                return CompletedProcess(command, 0, "", "")

            media_info = MediaInfo(
                path=Path("lecture.mov"),
                duration_seconds=120.0,
                has_audio=False,
            )

            MediaNormalizer(
                ffmpeg_path=str(ffmpeg),
                runner=runner,
                ffmpeg_hwaccel=FfmpegHwAccel.VIDEOTOOLBOX,
            ).normalize(
                source=Path("lecture.mov"),
                destination=Path(tmp) / "out" / "normalized_video.mp4",
                media_info=media_info,
                recording_speed=RecordingSpeed.DOUBLE,
                audio_quality=AudioQuality.FAST,
            )

            command = captured["command"]
            self.assertLess(command.index("-hwaccel"), command.index("-i"))
            self.assertIn("videotoolbox", command)

    def test_auto_videotoolbox_falls_back_to_software_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            ffmpeg = Path(tmp) / "ffmpeg"
            ffmpeg.write_text("binary", encoding="utf-8")
            commands = []

            def runner(command, capture_output, text):
                commands.append(command)
                if len(commands) == 1:
                    Path(command[-1]).write_text("partial", encoding="utf-8")
                    return CompletedProcess(command, 1, "", "hw failed")
                Path(command[-1]).write_text("video", encoding="utf-8")
                return CompletedProcess(command, 0, "", "")

            media_info = MediaInfo(
                path=Path("lecture.mov"),
                duration_seconds=120.0,
                has_audio=False,
            )

            MediaNormalizer(
                ffmpeg_path=str(ffmpeg),
                runner=runner,
                ffmpeg_hwaccel=FfmpegHwAccel.AUTO,
                apple_silicon=True,
            ).normalize(
                source=Path("lecture.mov"),
                destination=Path(tmp) / "out" / "lecture.mp4",
                media_info=media_info,
                recording_speed=RecordingSpeed.DOUBLE,
                audio_quality=AudioQuality.FAST,
            )

            self.assertEqual(len(commands), 2)
            self.assertIn("-hwaccel", commands[0])
            self.assertNotIn("-hwaccel", commands[1])
            self.assertEqual((Path(tmp) / "out" / "lecture.mp4").read_text(encoding="utf-8"), "video")
            self.assertFalse((Path(tmp) / "out" / ".lecture.mp4.ffmpeg.tmp").exists())

    def test_normalization_removes_temp_output_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            ffmpeg = Path(tmp) / "ffmpeg"
            ffmpeg.write_text("binary", encoding="utf-8")

            def runner(command, capture_output, text):
                Path(command[-1]).write_text("partial", encoding="utf-8")
                return CompletedProcess(command, 1, "", "failed")

            media_info = MediaInfo(
                path=Path("lecture.mov"),
                duration_seconds=120.0,
                has_audio=False,
            )
            destination = Path(tmp) / "out" / "lecture.mp4"

            with self.assertRaises(ProcessingError):
                MediaNormalizer(
                    ffmpeg_path=str(ffmpeg),
                    runner=runner,
                    ffmpeg_hwaccel=FfmpegHwAccel.NONE,
                ).normalize(
                    source=Path("lecture.mov"),
                    destination=destination,
                    media_info=media_info,
                    recording_speed=RecordingSpeed.DOUBLE,
                    audio_quality=AudioQuality.FAST,
                )

            self.assertFalse(destination.exists())
            self.assertFalse((destination.parent / ".lecture.mp4.ffmpeg.tmp").exists())

    def test_real_ffmpeg_normalization_accepts_hidden_temp_output(self):
        ffmpeg = shutil.which("ffmpeg") or str(_require_command("ffmpeg"))
        ffprobe = shutil.which("ffprobe") or str(_require_command("ffprobe"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mov"
            destination = root / "out" / "lecture.mp4"
            create = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=160x90:rate=10:duration=0.5",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=1000:duration=0.5",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    str(source),
                ],
                capture_output=True,
                text=True,
            )
            if create.returncode != 0:
                self.skipTest(f"Could not create ffmpeg fixture: {create.stderr}")

            from lecture_processor.media import MediaInspector

            media_info = MediaInspector(ffprobe_path=ffprobe).probe(source)
            MediaNormalizer(
                ffmpeg_path=ffmpeg,
                ffmpeg_hwaccel=FfmpegHwAccel.NONE,
            ).normalize(
                source=source,
                destination=destination,
                media_info=media_info,
                recording_speed=RecordingSpeed.DOUBLE,
                audio_quality=AudioQuality.FAST,
            )

            self.assertTrue(destination.exists())
            self.assertFalse((destination.parent / ".lecture.mp4.ffmpeg.tmp").exists())


if __name__ == "__main__":
    unittest.main()
