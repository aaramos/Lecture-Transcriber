import importlib
import os
import subprocess
import sys
import threading
import tempfile
from pathlib import Path
from typing import Dict, Optional, Protocol

from .config import TranscriptionEngine, TranscriptionQuality
from .errors import DependencyMissingError, ProcessingError
from .models import TranscriptResult, TranscriptSegment
from .profiles import PARAKEET_PROFILE_ID, TranscriptionProfile, get_profile


class Transcriber(Protocol):
    def transcribe(self, media_path: Path) -> TranscriptResult:
        ...


class NullTranscriber:
    metadata = {
        "resolved_engine": "none",
        "model": "",
        "coreml_used": False,
        "language": "en",
    }

    def transcribe(self, media_path: Path) -> TranscriptResult:
        return TranscriptResult(
            text="",
            segments=[],
        )


class LockedTranscriber:
    """Serializes transcription so large Whisper models are loaded once."""

    def __init__(self, transcriber: Transcriber) -> None:
        self._transcriber = transcriber
        self._lock = threading.Lock()

    def transcribe(self, media_path: Path) -> TranscriptResult:
        with self._lock:
            return self._transcriber.transcribe(media_path)

    @property
    def metadata(self) -> dict:
        return getattr(self._transcriber, "metadata", {})


class FasterWhisperTranscriber:
    COMMON_TRANSCRIBE_OPTIONS = {
        "language": "en",
        "task": "transcribe",
        "compression_ratio_threshold": 2.2,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 500},
    }
    QUALITY_PRESETS = {
        TranscriptionQuality.ACCURATE: {
            "model_options": {"compute_type": "default"},
            "transcribe_options": {
                "beam_size": 5,
                "best_of": 5,
                "temperature": [0.0, 0.2, 0.4],
                "repetition_penalty": 1.05,
                "no_repeat_ngram_size": 5,
            },
        },
        TranscriptionQuality.BALANCED: {
            "model_options": {"compute_type": "default"},
            "transcribe_options": {
                "beam_size": 2,
                "best_of": 2,
                "temperature": 0.0,
                "repetition_penalty": 1.03,
                "no_repeat_ngram_size": 5,
            },
        },
        TranscriptionQuality.FAST: {
            "model_options": {"compute_type": "int8"},
            "transcribe_options": {
                "beam_size": 1,
                "best_of": 1,
                "temperature": [0.0, 0.2],
                "repetition_penalty": 1.05,
                "no_repeat_ngram_size": 5,
            },
        },
    }

    def __init__(
        self,
        model_name: str,
        quality: TranscriptionQuality = TranscriptionQuality.ACCURATE,
        profile: Optional[TranscriptionProfile] = None,
    ) -> None:
        module = importlib.import_module("faster_whisper")
        self.profile = profile
        self.quality = TranscriptionQuality(
            profile.engine_kwargs.get("quality", quality) if profile else quality
        )
        preset = profile.engine_kwargs if profile else self.QUALITY_PRESETS[self.quality]
        self.model_options = dict(preset["model_options"])
        self.transcribe_options = {
            **self.COMMON_TRANSCRIBE_OPTIONS,
            **preset["transcribe_options"],
        }
        self._model = module.WhisperModel(model_name, **self.model_options)
        self.metadata = {
            "resolved_engine": "faster-whisper",
            "model": model_name,
            "profile": profile.id if profile else "",
            "profile_name": profile.display_name if profile else "",
            "quality": self.quality.value,
            "compute_type": self.model_options.get("compute_type", ""),
            "cpu_threads": self.model_options.get("cpu_threads"),
            "coreml_used": False,
            "language": "en",
            "audio_input": "clean_16khz_mono_wav",
            "condition_on_previous_text": False,
            "vad_filter": True,
        }

    def transcribe(self, media_path: Path) -> TranscriptResult:
        segments_iter, _info = self._model.transcribe(str(media_path), **self.transcribe_options)
        segments = []
        text_parts = []
        for segment in segments_iter:
            text = segment.text.strip()
            if text:
                text_parts.append(text)
                segments.append(
                    TranscriptSegment(
                        start=float(segment.start),
                        end=float(segment.end),
                        text=text,
                    )
                )
        text = " ".join(text_parts).strip()
        if not text:
            raise ProcessingError("Whisper returned empty transcript")
        return TranscriptResult(text=text, segments=segments)


class MLXWhisperTranscriber:
    def __init__(
        self,
        model_name: str,
        engine_kwargs: Dict,
        profile: Optional[TranscriptionProfile] = None,
    ) -> None:
        if not _mlx_whisper_import_available():
            raise DependencyMissingError(
                "Turbo transcription requires Apple Silicon with MLX available. "
                "Choose Quality/Fast or refresh dependencies."
            )
        self._module = importlib.import_module("mlx_whisper")
        self.model_name = model_name
        self.profile = profile
        self.transcribe_options = dict(engine_kwargs.get("transcribe_options") or {})
        self.metadata = {
            "resolved_engine": "mlx-whisper",
            "model": model_name,
            "profile": profile.id if profile else "",
            "profile_name": profile.display_name if profile else "",
            "quality": str(engine_kwargs.get("quality") or "accurate"),
            "compute_type": "mlx",
            "coreml_used": False,
            "language": self.transcribe_options.get("language", "en"),
            "audio_input": "clean_16khz_mono_wav",
            "condition_on_previous_text": bool(
                self.transcribe_options.get("condition_on_previous_text", False)
            ),
            "vad_filter": bool(engine_kwargs.get("vad_filter", True)),
            "word_timestamps": bool(self.transcribe_options.get("word_timestamps", False)),
        }

    def transcribe(self, media_path: Path) -> TranscriptResult:
        result = self._module.transcribe(
            str(media_path),
            path_or_hf_repo=self.model_name,
            **self.transcribe_options,
        )
        text = str(result.get("text") or "").strip()
        segments = []
        text_parts = []
        for segment in result.get("segments", []) or []:
            segment_text = str(segment.get("text") or "").strip()
            if segment_text:
                text_parts.append(segment_text)
                start = float(segment.get("start", 0.0) or 0.0)
                end = float(segment.get("end", start) or start)
                segments.append(TranscriptSegment(start=start, end=end, text=segment_text))
        if not text:
            text = " ".join(text_parts).strip()
        if not text:
            raise ProcessingError("Whisper returned empty transcript")
        return TranscriptResult(text=text, segments=segments)


class ParakeetMLXTranscriber:
    def __init__(
        self,
        model_name: str,
        engine_kwargs: Dict,
        profile: Optional[TranscriptionProfile] = None,
    ) -> None:
        if not _parakeet_mlx_import_available():
            raise DependencyMissingError(
                "Parakeet transcription requires Apple Silicon and parakeet-mlx. "
                "Install dependencies, or choose Quality/Fast/Turbo."
            )
        self._module = importlib.import_module("parakeet_mlx")
        self.model_name = model_name
        self.profile = profile
        self.transcribe_options = dict(engine_kwargs.get("transcribe_options") or {})
        self.cache_dir = self.transcribe_options.get("cache_dir") or _default_parakeet_cache_dir()
        self.metadata = {
            "resolved_engine": "parakeet-mlx",
            "model": model_name,
            "profile": profile.id if profile else "",
            "profile_name": profile.display_name if profile else "",
            "quality": str(engine_kwargs.get("quality") or "accurate"),
            "compute_type": "mlx",
            "coreml_used": False,
            "language": self.transcribe_options.get("language", "en"),
            "audio_input": "clean_16khz_mono_wav",
            "condition_on_previous_text": False,
            "vad_filter": bool(engine_kwargs.get("vad_filter", True)),
        }

        dtype = self.transcribe_options.get("dtype")
        dtype_value = _resolve_parakeet_dtype(self._module, dtype)
        from_pretrained_kwargs = {"cache_dir": self.cache_dir}
        if dtype_value is not None:
            from_pretrained_kwargs["dtype"] = dtype_value

        resolved_model = _resolve_parakeet_model_name(model_name, cache_dir=self.cache_dir)
        self._model = self._module.from_pretrained(
            resolved_model,
            **from_pretrained_kwargs,
        )

    def transcribe(self, media_path: Path) -> TranscriptResult:
        options = dict(self.transcribe_options)
        options.pop("cache_dir", None)
        options.pop("language", None)
        if options.get("decoding_config") is None:
            options.pop("decoding_config")
        options.pop("dtype", None)
        result = self._model.transcribe(str(media_path), **options)
        text = str(getattr(result, "text", "") or "").strip()
        if not text and isinstance(result, dict):
            text = str(result.get("text") or "").strip()
        segments = []
        text_parts = []
        sentences = getattr(result, "sentences", None)
        if sentences is None and isinstance(result, dict):
            sentences = result.get("sentences")
        for sentence in sentences or []:
            segment_text = str(
                getattr(sentence, "text", None) if not isinstance(sentence, dict) else sentence.get("text") or ""
            ).strip()
            if segment_text:
                text_parts.append(segment_text)
                if isinstance(sentence, dict):
                    start = float(sentence.get("start", 0.0) or 0.0)
                    end = float(sentence.get("end", start) or start)
                else:
                    start = float(getattr(sentence, "start", 0.0) or 0.0)
                    end = float(getattr(sentence, "end", start) or start)
                segments.append(TranscriptSegment(start=start, end=end, text=segment_text))
        if not text:
            text = " ".join(text_parts).strip()
        if not text:
            raise ProcessingError("Parakeet returned empty transcript")
        return TranscriptResult(text=text, segments=segments)


def _resolve_parakeet_dtype(module: object, dtype: Optional[object]) -> object:
    mlx_module = getattr(module, "mx", None)
    if mlx_module is None:
        try:
            import mlx.core as mlx_module  # type: ignore[attr-defined]
        except Exception:
            return None

    if dtype in (None, "", "bfloat16"):
        return getattr(mlx_module, "bfloat16", getattr(mlx_module, "float16", None))

    if not isinstance(dtype, str):
        return dtype

    normalized = dtype.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "f16": "float16",
        "float16": "float16",
        "half": "float16",
        "f32": "float32",
        "float32": "float32",
        "single": "float32",
    }
    dtype_key = aliases.get(normalized, normalized)
    if hasattr(mlx_module, dtype_key):
        return getattr(mlx_module, dtype_key)

    return None


class OpenAIWhisperTranscriber:
    def __init__(self, model_name: str) -> None:
        module = importlib.import_module("whisper")
        self._model = module.load_model(model_name)
        self.metadata = {
            "resolved_engine": "openai-whisper",
            "model": model_name,
            "coreml_used": False,
            "language": "en",
        }

    def transcribe(self, media_path: Path) -> TranscriptResult:
        result = self._model.transcribe(str(media_path))
        text = str(result.get("text") or "").strip()
        if not text:
            raise ProcessingError("Whisper returned empty transcript")
        segments = [
            TranscriptSegment(
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=str(segment.get("text") or "").strip(),
            )
            for segment in result.get("segments", [])
        ]
        return TranscriptResult(text=text, segments=segments)


class WhisperCppTranscriber:
    def __init__(
        self,
        model_name: str,
        models_dir: Optional[str] = None,
        require_coreml: bool = False,
        n_threads: Optional[int] = None,
    ) -> None:
        configure_whisper_cpp_runtime_env()
        module = importlib.import_module("pywhispercpp.model")
        system_info = str(module.Model.system_info())
        if require_coreml:
            if "COREML = 1" not in system_info:
                raise DependencyMissingError(
                    "pywhispercpp is installed, but it was not built with CoreML support. "
                    "Run scripts/setup-whisper-cpp-coreml.sh, then keep the ggml .bin model "
                    "next to its matching *-encoder.mlmodelc directory."
                )

        model_kwargs = {
            "print_realtime": False,
            "print_progress": False,
            "redirect_whispercpp_logs_to": False,
        }
        if models_dir:
            model_kwargs["models_dir"] = models_dir
        if n_threads:
            model_kwargs["n_threads"] = n_threads

        self._model = module.Model(model_name, **model_kwargs)
        self.metadata = {
            "resolved_engine": "whisper-cpp",
            "model": model_name,
            "coreml_used": "COREML = 1" in system_info,
            "language": "en",
        }

    def transcribe(self, media_path: Path) -> TranscriptResult:
        segments_raw = self._model.transcribe(str(media_path))
        segments = []
        text_parts = []
        for segment in segments_raw:
            text = str(getattr(segment, "text", "") or "").strip()
            if text:
                text_parts.append(text)
            start, end = _whisper_cpp_segment_times(segment)
            segments.append(TranscriptSegment(start=start, end=end, text=text))
        text = " ".join(text_parts).strip()
        if not text:
            raise ProcessingError("Whisper returned empty transcript")
        return TranscriptResult(text=text, segments=segments)


def build_transcriber(
    engine: TranscriptionEngine,
    model_name: str,
    *,
    quality: TranscriptionQuality = TranscriptionQuality.ACCURATE,
    profile_id: Optional[str] = None,
    prefer_whisper_cpp: bool = False,
    whisper_cpp_model_dir: str = "",
    require_whisper_cpp_coreml: bool = False,
    n_threads: Optional[int] = None,
) -> LockedTranscriber:
    if engine is TranscriptionEngine.NONE:
        return LockedTranscriber(NullTranscriber())

    if profile_id:
        profile = get_profile(profile_id)
        try:
            return LockedTranscriber(_build_profile_transcriber(profile))
        except ImportError as exc:
            if profile.engine == "faster-whisper":
                raise DependencyMissingError(
                    "faster-whisper is not installed. Install with: python3 -m pip install -e '.[transcription]'"
                ) from exc
            if profile.engine == "mlx-whisper":
                raise DependencyMissingError(
                    "Turbo transcription requires Apple Silicon and mlx-whisper. "
                    "Refresh dependencies, or choose Quality/Fast."
                ) from exc
            if profile.engine == "parakeet-mlx":
                raise DependencyMissingError(
                    "Parakeet transcription requires Apple Silicon and parakeet-mlx. "
                    "Install dependencies, or choose Quality/Fast/Turbo."
                ) from exc
        except DependencyMissingError:
            raise
        except Exception as exc:
            raise ProcessingError(
                f"Could not initialize {profile.display_name} transcription with model "
                f"'{profile.model}': {exc}"
            ) from exc

    if engine is TranscriptionEngine.PARAKEET_MLX:
        try:
            return LockedTranscriber(_build_profile_transcriber(get_profile(PARAKEET_PROFILE_ID)))
        except DependencyMissingError:
            raise
        except Exception as exc:
            raise ProcessingError("Could not initialize Parakeet transcription engine.") from exc

    wants_whisper_cpp = engine is TranscriptionEngine.WHISPER_CPP or (
        engine is TranscriptionEngine.AUTO and (prefer_whisper_cpp or require_whisper_cpp_coreml)
    )
    if wants_whisper_cpp:
        try:
            return LockedTranscriber(
                WhisperCppTranscriber(
                    model_name,
                    models_dir=whisper_cpp_model_dir or None,
                    require_coreml=require_whisper_cpp_coreml,
                    n_threads=n_threads,
                )
            )
        except ImportError:
            if engine is TranscriptionEngine.WHISPER_CPP or require_whisper_cpp_coreml:
                raise DependencyMissingError(
                    "pywhispercpp is not installed. For Apple Silicon acceleration, run "
                    "scripts/setup-whisper-cpp-coreml.sh or install pywhispercpp from source "
                    "with CoreML enabled."
                )
        except DependencyMissingError:
            if engine is TranscriptionEngine.WHISPER_CPP or require_whisper_cpp_coreml:
                raise
        except Exception as exc:
            if engine is TranscriptionEngine.WHISPER_CPP or require_whisper_cpp_coreml:
                raise ProcessingError(
                    f"Could not initialize pywhispercpp model '{model_name}': {exc}"
                ) from exc

    if engine in (TranscriptionEngine.AUTO, TranscriptionEngine.FASTER_WHISPER):
        try:
            return LockedTranscriber(FasterWhisperTranscriber(model_name, quality))
        except ImportError:
            if engine is TranscriptionEngine.FASTER_WHISPER:
                raise DependencyMissingError(
                    "faster-whisper is not installed. Install with: "
                    "python3 -m pip install -e '.[transcription]'"
                )

    if engine in (TranscriptionEngine.AUTO, TranscriptionEngine.OPENAI_WHISPER):
        try:
            return LockedTranscriber(OpenAIWhisperTranscriber(model_name))
        except ImportError:
            if engine is TranscriptionEngine.OPENAI_WHISPER:
                raise DependencyMissingError(
                    "openai-whisper is not installed. Install the whisper package or use "
                    "--transcription-engine faster-whisper."
                )

    raise DependencyMissingError(
        "No Whisper transcription engine is installed. Install optional dependencies or "
        "use --transcription-engine none for media-only smoke tests."
    )


def _build_profile_transcriber(profile: TranscriptionProfile) -> Transcriber:
    if profile.engine == "faster-whisper":
        return FasterWhisperTranscriber(profile.model, profile=profile)
    if profile.engine == "mlx-whisper":
        return MLXWhisperTranscriber(profile.model, profile.engine_kwargs, profile=profile)
    if profile.engine == "parakeet-mlx":
        return ParakeetMLXTranscriber(profile.model, profile.engine_kwargs, profile=profile)
    raise DependencyMissingError(f"Unsupported transcription profile engine: {profile.engine}")


def _mlx_metal_is_available() -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import mlx.core.metal as metal; exit(0 if metal.is_available() else 1)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _mlx_whisper_import_available() -> bool:
    if not _mlx_metal_is_available():
        return False
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import mlx_whisper"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _parakeet_mlx_import_available() -> bool:
    if not _mlx_metal_is_available():
        return False
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import parakeet_mlx"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _resolve_parakeet_model_name(model_name: str, cache_dir: Optional[str] = None) -> str:
    explicit_model_path = Path(model_name)
    if explicit_model_path.exists():
        config_path = explicit_model_path / "config.json"
        if config_path.is_file():
            return model_name
        raise FileNotFoundError(
            f"Could not initialize Parakeet model at '{model_name}': "
            "Expected 'config.json' to exist in the model directory."
        )

    if "/" not in model_name:
        raise FileNotFoundError(
            f"Could not initialize Parakeet model '{model_name}': no local model path found."
        )

    model_candidates = _get_parakeet_model_candidates(model_name)
    last_error: Optional[BaseException] = None
    for candidate in model_candidates:
        try:
            return _snapshot_download(candidate, cache_dir=cache_dir)
        except Exception as exc:  # pragma: no cover - exercised through tests
            last_error = exc

    raise DependencyMissingError(
        f"Could not download Parakeet model '{model_name}'. "
        f"Tried {', '.join(model_candidates)}. "
        "Verify network access and model id permissions, or switch to a different Parakeet profile."
    ) from last_error


def _get_parakeet_model_candidates(model_name: str) -> list[str]:
    fallback_map = {
        "senstella/parakeet-mlx": [
            "animaslabs/parakeet-tdt-0.6b-v3-mlx",
            "mlx-community/parakeet-tdt-0.6b-v2",
            "senstella/parakeet-tdt-0.6b-v2-mlx",
        ]
    }
    candidates = [model_name]
    if model_name in fallback_map:
        for fallback in fallback_map[model_name]:
            if fallback not in candidates:
                candidates.append(fallback)
    return candidates


def _default_parakeet_cache_dir() -> str:
    candidates = [
        os.environ.get("LECTURE_PROCESSOR_PARAKEET_CACHE_DIR"),
        str(Path.home() / ".cache" / "lecture-processor" / "models" / "parakeet"),
        os.path.join(tempfile.gettempdir(), "lecture-processor", "parakeet"),
    ]
    for base in candidates:
        if not base:
            continue
        path = Path(base)
        try:
            path.mkdir(parents=True, exist_ok=True)
            return str(path)
        except Exception:
            continue
    return tempfile.gettempdir()


def _snapshot_download(model_name: str, cache_dir: Optional[str] = None) -> str:
    from huggingface_hub import snapshot_download

    kwargs = {"repo_id": model_name}
    if cache_dir:
        kwargs["local_dir"] = cache_dir
    return snapshot_download(**kwargs)


def configure_whisper_cpp_runtime_env() -> None:
    os.environ.setdefault("PYWHISPERCPP_USE_GPU", "0")
    os.environ.setdefault("PYWHISPERCPP_FLASH_ATTN", "0")


def _whisper_cpp_segment_times(segment) -> tuple:
    if hasattr(segment, "start") and hasattr(segment, "end"):
        return float(segment.start), float(segment.end)
    if hasattr(segment, "t0") and hasattr(segment, "t1"):
        return float(segment.t0) / 100.0, float(segment.t1) / 100.0
    return 0.0, 0.0
