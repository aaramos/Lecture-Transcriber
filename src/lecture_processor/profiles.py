import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


QUALITY_PROFILE_ID = "quality"
FAST_PROFILE_ID = "fast"
TURBO_PROFILE_ID = "turbo"
PARAKEET_PROFILE_ID = "parakeet"
DEFAULT_PROFILE_ID = QUALITY_PROFILE_ID


@dataclass(frozen=True)
class TranscriptionProfile:
    id: str
    display_name: str
    engine: str
    model: str
    engine_kwargs: Dict[str, Any]
    description: str


_PROFILE_ORDER = (
    QUALITY_PROFILE_ID,
    FAST_PROFILE_ID,
    TURBO_PROFILE_ID,
    PARAKEET_PROFILE_ID,
)

_LEGACY_PROFILE_ALIASES = {
    "": DEFAULT_PROFILE_ID,
    "accurate": QUALITY_PROFILE_ID,
    "balanced": FAST_PROFILE_ID,
    "default": DEFAULT_PROFILE_ID,
    "faster-whisper": QUALITY_PROFILE_ID,
    "fast": FAST_PROFILE_ID,
    "mlx-whisper": TURBO_PROFILE_ID,
    "parakeet": PARAKEET_PROFILE_ID,
    "parakeet-mlx": PARAKEET_PROFILE_ID,
    "senstella/parakeet-mlx": PARAKEET_PROFILE_ID,
    "quality": QUALITY_PROFILE_ID,
    "turbo": TURBO_PROFILE_ID,
}


def detect_performance_core_count() -> int:
    try:
        output = subprocess.check_output(
            ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1,
        )
        detected = int(output.strip())
        if detected > 0:
            return detected
    except Exception:
        pass

    cpu_count = os.cpu_count()
    if cpu_count and cpu_count > 1:
        return max(1, cpu_count // 2)
    return 4


def normalize_profile_id(value: Optional[str]) -> str:
    key = str(value or "").strip().lower()
    try:
        return _LEGACY_PROFILE_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(_PROFILE_ORDER)
        raise ValueError(f"Unknown transcription profile '{value}'. Choose one of: {valid}.") from exc


def profile_from_legacy_quality(value: Optional[str]) -> str:
    return normalize_profile_id(value)


def all_profiles() -> Iterable[TranscriptionProfile]:
    return [get_profile(profile_id) for profile_id in _PROFILE_ORDER]


def get_profile(profile_id: Optional[str]) -> TranscriptionProfile:
    profile_id = normalize_profile_id(profile_id)
    if profile_id == QUALITY_PROFILE_ID:
        return TranscriptionProfile(
            id=QUALITY_PROFILE_ID,
            display_name="Quality",
            engine="faster-whisper",
            model="medium.en",
            engine_kwargs={
                "model_options": {
                    "compute_type": "default",
                },
                "transcribe_options": {
                    "language": "en",
                    "task": "transcribe",
                    "beam_size": 5,
                    "best_of": 5,
                    "temperature": [0.0, 0.2, 0.4],
                    "repetition_penalty": 1.05,
                    "no_repeat_ngram_size": 5,
                    "condition_on_previous_text": False,
                    "vad_filter": True,
                    "vad_parameters": {"min_silence_duration_ms": 500},
                },
                "quality": "accurate",
            },
            description="Highest accuracy. Best for difficult audio. ~1.5-2x real-time.",
        )
    if profile_id == FAST_PROFILE_ID:
        return TranscriptionProfile(
            id=FAST_PROFILE_ID,
            display_name="Fast",
            engine="faster-whisper",
            model="medium.en",
            engine_kwargs={
                "model_options": {
                    "compute_type": "int8",
                    "cpu_threads": detect_performance_core_count(),
                },
                "transcribe_options": {
                    "language": "en",
                    "task": "transcribe",
                    "beam_size": 1,
                    "best_of": 1,
                    "temperature": [0.0, 0.2],
                    "repetition_penalty": 1.05,
                    "no_repeat_ngram_size": 5,
                    "condition_on_previous_text": False,
                    "vad_filter": True,
                    "vad_parameters": {"min_silence_duration_ms": 500},
                },
                "quality": "fast",
            },
            description="Recommended for most lectures. Minimal accuracy loss vs Quality, ~2-3x faster.",
        )
    if profile_id == PARAKEET_PROFILE_ID:
        return TranscriptionProfile(
            id=PARAKEET_PROFILE_ID,
            display_name="Parakeet",
            engine="parakeet-mlx",
            model="animaslabs/parakeet-tdt-0.6b-v3-mlx",
            engine_kwargs={
                "transcribe_options": {
                    "dtype": "bfloat16",
                    # Bound the model's mel/inference working set for long lectures.
                    "chunk_duration": 600.0,
                    "overlap_duration": 15.0,
                    "decoding_config": None,
                },
                "quality": "accurate",
                "vad_filter": True,
            },
            description=(
                "Parakeet is a lightweight MLX speech model for Apple Silicon. It can be a good option "
                "for fast transcriptions on longer audio when dependencies are available."
            ),
        )
    return TranscriptionProfile(
        id=TURBO_PROFILE_ID,
        display_name="Turbo",
        engine="mlx-whisper",
        model="mlx-community/whisper-large-v3-mlx",
        engine_kwargs={
            "transcribe_options": {
                "language": "en",
                "task": "transcribe",
                "temperature": 0.0,
                "condition_on_previous_text": False,
                "word_timestamps": True,
                "no_speech_threshold": 0.6,
            },
            "quality": "accurate",
            "vad_filter": True,
        },
        description="Maximum speed using Apple Silicon acceleration. Excellent accuracy. Requires M-series Mac.",
    )
