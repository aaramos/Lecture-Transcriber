import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess

from lecture_processor.config import AudioQuality, FfmpegHwAccel, RecordingSpeed
from lecture_processor.errors import DependencyMissingError, ProcessingError
from lecture_processor.media import (
    MediaNormalizer,
    _audio_filter_for,
    _hwaccel_args,
    _has_audio_stream,
    _require_command,
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
