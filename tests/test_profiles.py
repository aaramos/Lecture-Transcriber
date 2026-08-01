import unittest
from unittest import mock

from lecture_processor.profiles import (
    FAST_PROFILE_ID,
    QUALITY_PROFILE_ID,
    PARAKEET_PROFILE_ID,
    TURBO_PROFILE_ID,
    all_profiles,
    detect_performance_core_count,
    get_profile,
    profile_from_legacy_quality,
)


class TranscriptionProfileTests(unittest.TestCase):
    def test_registry_loads_profiles_in_ui_order(self):
        profiles = list(all_profiles())

        self.assertEqual(
            [profile.id for profile in profiles],
            [QUALITY_PROFILE_ID, FAST_PROFILE_ID, TURBO_PROFILE_ID, PARAKEET_PROFILE_ID],
        )
        self.assertEqual(profiles[0].engine, "faster-whisper")
        self.assertEqual(profiles[0].model, "medium.en")
        self.assertEqual(profiles[1].engine_kwargs["model_options"]["compute_type"], "int8")
        self.assertEqual(profiles[2].engine, "mlx-whisper")
        self.assertTrue(profiles[2].engine_kwargs["transcribe_options"]["word_timestamps"])
        self.assertEqual(profiles[3].engine, "parakeet-mlx")
        self.assertEqual(profiles[3].model, "animaslabs/parakeet-tdt-0.6b-v3-mlx")
        self.assertEqual(profiles[3].engine_kwargs["transcribe_options"]["chunk_duration"], 600.0)

    def test_legacy_quality_values_migrate_to_profiles(self):
        self.assertEqual(profile_from_legacy_quality("accurate"), QUALITY_PROFILE_ID)
        self.assertEqual(profile_from_legacy_quality("balanced"), FAST_PROFILE_ID)
        self.assertEqual(profile_from_legacy_quality("fast"), FAST_PROFILE_ID)

    def test_fast_profile_uses_detected_performance_core_count(self):
        with mock.patch("lecture_processor.profiles.detect_performance_core_count", return_value=6):
            profile = get_profile(FAST_PROFILE_ID)

        self.assertEqual(profile.engine_kwargs["model_options"]["cpu_threads"], 6)

    def test_performance_core_count_falls_back_to_half_cpu_count(self):
        with mock.patch("lecture_processor.profiles.subprocess.check_output", side_effect=OSError), mock.patch(
            "lecture_processor.profiles.os.cpu_count",
            return_value=10,
        ):
            self.assertEqual(detect_performance_core_count(), 5)


if __name__ == "__main__":
    unittest.main()
