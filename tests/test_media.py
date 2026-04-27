import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess

from lecture_processor.config import AudioQuality, RecordingSpeed
from lecture_processor.errors import DependencyMissingError
from lecture_processor.media import (
    MediaNormalizer,
    _audio_filter_for,
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

    def test_detects_audio_streams_from_probe_payload(self):
        self.assertTrue(_has_audio_stream({"streams": [{"codec_type": "audio"}]}))
        self.assertFalse(_has_audio_stream({"streams": [{"codec_type": "video"}]}))


if __name__ == "__main__":
    unittest.main()
