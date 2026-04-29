import os
import unittest
from pathlib import Path
from unittest import mock

from lecture_processor.config import TranscriptionEngine, TranscriptionQuality
from lecture_processor.errors import DependencyMissingError, ProcessingError
from lecture_processor.profiles import FAST_PROFILE_ID, QUALITY_PROFILE_ID, TURBO_PROFILE_ID, get_profile
from lecture_processor.transcription import (
    FasterWhisperTranscriber,
    MLXWhisperTranscriber,
    NullTranscriber,
    _whisper_cpp_segment_times,
    build_transcriber,
    configure_whisper_cpp_runtime_env,
)


class FakeWhisperCppSegment:
    t0 = 125
    t1 = 260


class FakeFasterWhisperSegment:
    start = 1.0
    end = 3.0
    text = " hello lecture "


class FakeMLXWhisperModule:
    def __init__(self):
        self.calls = []

    def transcribe(self, media_path, **options):
        self.calls.append({"media_path": media_path, "options": options})
        return {
            "text": "hello lecture",
            "segments": [{"start": 1.0, "end": 3.0, "text": " hello lecture "}],
        }


class TranscriptionTests(unittest.TestCase):
    def test_whisper_cpp_segment_times_convert_centiseconds(self):
        self.assertEqual(_whisper_cpp_segment_times(FakeWhisperCppSegment()), (1.25, 2.6))

    def test_explicit_whisper_cpp_reports_missing_dependency(self):
        with mock.patch(
            "lecture_processor.transcription.importlib.import_module",
            side_effect=ImportError("missing"),
        ):
            with self.assertRaises(DependencyMissingError) as context:
                build_transcriber(TranscriptionEngine.WHISPER_CPP, "tiny")

        self.assertIn("pywhispercpp", str(context.exception))

    def test_explicit_whisper_cpp_reports_initialization_failure(self):
        with mock.patch(
            "lecture_processor.transcription.WhisperCppTranscriber",
            side_effect=RuntimeError("model failed"),
        ):
            with self.assertRaises(ProcessingError) as context:
                build_transcriber(TranscriptionEngine.WHISPER_CPP, "large-v3")

        self.assertIn("pywhispercpp", str(context.exception))

    def test_auto_falls_back_when_preferred_whisper_cpp_cannot_initialize(self):
        with mock.patch(
            "lecture_processor.transcription.WhisperCppTranscriber",
            side_effect=RuntimeError("model failed"),
        ), mock.patch(
            "lecture_processor.transcription.FasterWhisperTranscriber",
            return_value=NullTranscriber(),
        ):
            transcriber = build_transcriber(
                TranscriptionEngine.AUTO,
                "large-v3",
                prefer_whisper_cpp=True,
            )

        self.assertIsInstance(transcriber._transcriber, NullTranscriber)

    def test_faster_whisper_uses_quality_english_transcription_options_by_default(self):
        captured = {}

        class FakeWhisperModel:
            def __init__(self, model_name, **model_options):
                captured["model_name"] = model_name
                captured["model_options"] = model_options

            def transcribe(self, media_path, **options):
                captured["media_path"] = media_path
                captured["options"] = options
                return iter([FakeFasterWhisperSegment()]), object()

        fake_module = type("FakeFasterWhisperModule", (), {"WhisperModel": FakeWhisperModel})
        with mock.patch("lecture_processor.transcription.importlib.import_module", return_value=fake_module):
            result = FasterWhisperTranscriber("large-v3").transcribe(Path("clean.wav"))

        self.assertEqual(result.text, "hello lecture")
        self.assertEqual(captured["model_name"], "large-v3")
        self.assertEqual(captured["model_options"]["compute_type"], "default")
        self.assertEqual(captured["media_path"], "clean.wav")
        self.assertEqual(captured["options"]["language"], "en")
        self.assertFalse(captured["options"]["condition_on_previous_text"])
        self.assertTrue(captured["options"]["vad_filter"])
        self.assertEqual(captured["options"]["beam_size"], 5)
        self.assertEqual(captured["options"]["best_of"], 5)
        self.assertEqual(captured["options"]["temperature"], [0.0, 0.2, 0.4])
        self.assertEqual(captured["options"]["no_repeat_ngram_size"], 5)

    def test_faster_whisper_quality_modes_adjust_model_and_decode_options(self):
        captured = []

        class FakeWhisperModel:
            def __init__(self, model_name, **model_options):
                captured.append({"model_name": model_name, "model_options": model_options})

            def transcribe(self, media_path, **options):
                captured[-1]["options"] = options
                return iter([FakeFasterWhisperSegment()]), object()

        fake_module = type("FakeFasterWhisperModule", (), {"WhisperModel": FakeWhisperModel})
        with mock.patch("lecture_processor.transcription.importlib.import_module", return_value=fake_module):
            FasterWhisperTranscriber("large-v3", TranscriptionQuality.ACCURATE).transcribe(Path("clean.wav"))
            FasterWhisperTranscriber("large-v3", TranscriptionQuality.FAST).transcribe(Path("clean.wav"))

        self.assertEqual(captured[0]["model_options"]["compute_type"], "default")
        self.assertEqual(captured[0]["options"]["beam_size"], 5)
        self.assertEqual(captured[0]["options"]["best_of"], 5)
        self.assertEqual(captured[0]["options"]["temperature"], [0.0, 0.2, 0.4])
        self.assertEqual(captured[1]["model_options"]["compute_type"], "int8")
        self.assertEqual(captured[1]["options"]["beam_size"], 1)
        self.assertEqual(captured[1]["options"]["best_of"], 1)
        self.assertEqual(captured[1]["options"]["temperature"], [0.0, 0.2])
        self.assertEqual(captured[1]["options"]["no_repeat_ngram_size"], 5)

    def test_build_transcriber_passes_profile_to_faster_whisper(self):
        with mock.patch(
            "lecture_processor.transcription.FasterWhisperTranscriber",
            return_value=NullTranscriber(),
        ) as constructor:
            build_transcriber(
                TranscriptionEngine.FASTER_WHISPER,
                "medium.en",
                quality=TranscriptionQuality.ACCURATE,
                profile_id=QUALITY_PROFILE_ID,
            )

        args, kwargs = constructor.call_args
        self.assertEqual(args, ("medium.en",))
        self.assertEqual(kwargs["profile"].id, QUALITY_PROFILE_ID)

    def test_fast_profile_passes_speed_kwargs_to_faster_whisper(self):
        captured = {}

        class FakeWhisperModel:
            def __init__(self, model_name, **model_options):
                captured["model_name"] = model_name
                captured["model_options"] = model_options

            def transcribe(self, media_path, **options):
                captured["options"] = options
                return iter([FakeFasterWhisperSegment()]), object()

        fake_module = type("FakeFasterWhisperModule", (), {"WhisperModel": FakeWhisperModel})
        with mock.patch("lecture_processor.profiles.detect_performance_core_count", return_value=6), mock.patch(
            "lecture_processor.transcription.importlib.import_module",
            return_value=fake_module,
        ):
            build_transcriber(
                TranscriptionEngine.FASTER_WHISPER,
                "ignored",
                profile_id=FAST_PROFILE_ID,
            ).transcribe(Path("clean.wav"))

        self.assertEqual(captured["model_name"], "medium.en")
        self.assertEqual(captured["model_options"]["compute_type"], "int8")
        self.assertEqual(captured["model_options"]["cpu_threads"], 6)
        self.assertEqual(captured["options"]["beam_size"], 1)
        self.assertEqual(captured["options"]["best_of"], 1)
        self.assertEqual(captured["options"]["temperature"], [0.0, 0.2])

    def test_mlx_engine_accepts_turbo_profile_options(self):
        fake_module = FakeMLXWhisperModule()
        profile = get_profile(TURBO_PROFILE_ID)

        with mock.patch("lecture_processor.transcription._mlx_whisper_import_available", return_value=True), mock.patch(
            "lecture_processor.transcription.importlib.import_module",
            return_value=fake_module,
        ):
            result = MLXWhisperTranscriber(profile.model, profile.engine_kwargs, profile=profile).transcribe(
                Path("clean.wav")
            )

        self.assertEqual(result.text, "hello lecture")
        self.assertEqual(result.segments[0].start, 1.0)
        call = fake_module.calls[0]
        self.assertEqual(call["options"]["path_or_hf_repo"], "mlx-community/whisper-large-v3-mlx")
        self.assertTrue(call["options"]["word_timestamps"])
        self.assertFalse(call["options"]["condition_on_previous_text"])

    def test_build_transcriber_uses_mlx_engine_for_turbo_profile(self):
        with mock.patch("lecture_processor.transcription._mlx_whisper_import_available", return_value=True), mock.patch(
            "lecture_processor.transcription.MLXWhisperTranscriber",
            return_value=NullTranscriber(),
        ) as constructor:
            transcriber = build_transcriber(
                TranscriptionEngine.FASTER_WHISPER,
                "ignored",
                profile_id=TURBO_PROFILE_ID,
            )

        self.assertIsInstance(transcriber._transcriber, NullTranscriber)
        args, kwargs = constructor.call_args
        self.assertEqual(args[0], "mlx-community/whisper-large-v3-mlx")
        self.assertEqual(args[1]["transcribe_options"]["temperature"], 0.0)
        self.assertEqual(kwargs["profile"].id, TURBO_PROFILE_ID)

    def test_turbo_profile_reports_unavailable_mlx_without_crashing(self):
        with mock.patch("lecture_processor.transcription._mlx_whisper_import_available", return_value=False):
            with self.assertRaises(DependencyMissingError) as context:
                build_transcriber(
                    TranscriptionEngine.FASTER_WHISPER,
                    "ignored",
                    profile_id=TURBO_PROFILE_ID,
                )

        self.assertIn("Turbo transcription requires Apple Silicon", str(context.exception))

    def test_whisper_cpp_runtime_defaults_disable_unstable_metal_decoder(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            configure_whisper_cpp_runtime_env()

            self.assertEqual(os.environ["PYWHISPERCPP_USE_GPU"], "0")
            self.assertEqual(os.environ["PYWHISPERCPP_FLASH_ATTN"], "0")

    def test_whisper_cpp_runtime_preserves_explicit_overrides(self):
        with mock.patch.dict(
            os.environ,
            {"PYWHISPERCPP_USE_GPU": "1", "PYWHISPERCPP_FLASH_ATTN": "1"},
            clear=True,
        ):
            configure_whisper_cpp_runtime_env()

            self.assertEqual(os.environ["PYWHISPERCPP_USE_GPU"], "1")
            self.assertEqual(os.environ["PYWHISPERCPP_FLASH_ATTN"], "1")


if __name__ == "__main__":
    unittest.main()
