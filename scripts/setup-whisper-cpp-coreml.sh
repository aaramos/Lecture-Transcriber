#!/usr/bin/env sh
set -eu

# Builds pywhispercpp with CoreML support and generates the matching
# whisper.cpp CoreML encoder bundle for the requested model.
#
# Usage:
#   . ./scripts/dev-env.sh
#   scripts/setup-whisper-cpp-coreml.sh large-v3
#
# Optional second arg chooses the model output directory.

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
MODEL="${1:-large-v3}"
MODEL_DIR="${2:-$PROJECT_ROOT/.models/whisper-cpp}"
BUILD_ROOT="${TMPDIR:-/tmp}/lecture-processor-whisper-cpp"
PYTHON="${PYTHON:-python3}"
PYTHON="$("$PYTHON" -c 'import sys; print(sys.executable)')"
PYTHON_DIR="$(dirname "$PYTHON")"
PYTHON_CMAKE="$BUILD_ROOT/python-for-cmake"

mkdir -p "$MODEL_DIR" "$BUILD_ROOT"
ln -sfn "$PYTHON" "$PYTHON_CMAKE"

"$PYTHON" -m pip install --upgrade \
  pip \
  setuptools \
  wheel \
  build \
  cmake \
  ninja \
  repairwheel \
  coremltools \
  ane_transformers \
  openai-whisper \
  "numpy<2" \
  requests \
  tqdm \
  platformdirs

"$PYTHON" -m pip install --upgrade "numpy<2" "torch==2.5.0"

if [ ! -d "$BUILD_ROOT/pywhispercpp/.git" ]; then
  git clone --recursive https://github.com/abdeladim-s/pywhispercpp "$BUILD_ROOT/pywhispercpp"
else
  git -C "$BUILD_ROOT/pywhispercpp" pull --ff-only
  git -C "$BUILD_ROOT/pywhispercpp" submodule update --init --recursive
fi

cd "$BUILD_ROOT/pywhispercpp"
"$PYTHON" - <<'PY'
from pathlib import Path

utils = Path("pywhispercpp/utils.py")
utils_text = utils.read_text()
utils_text = utils_text.replace("from typing import TextIO\n", "from typing import TextIO, Union\n")
utils_text = utils_text.replace(
    "def redirect_stderr(to: bool | TextIO | str | None = False) -> None:",
    "def redirect_stderr(to: Union[bool, TextIO, str, None] = False) -> None:",
)
utils.write_text(utils_text)

main = Path("src/main.cpp")
main_text = main.read_text()
if "#include <cstdlib>" not in main_text:
    main_text = main_text.replace(
        "#include <pybind11/numpy.h>\n",
        "#include <pybind11/numpy.h>\n#include <cstdlib>\n#include <string>\n",
    )
if "bool env_flag(const char * name, bool default_value)" not in main_text:
    main_text = main_text.replace(
        "struct whisper_model_loader_wrapper {\n    whisper_model_loader* ptr;\n\n};\n",
        """struct whisper_model_loader_wrapper {
    whisper_model_loader* ptr;

};

bool env_flag(const char * name, bool default_value) {
    const char * raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }
    std::string value(raw);
    return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "YES";
}
""",
    )
main_text = main_text.replace(
    "struct whisper_context * ctx = whisper_init_from_file(path_model);",
    """struct whisper_context_params params = whisper_context_default_params();
    params.use_gpu = env_flag("PYWHISPERCPP_USE_GPU", true);
    params.flash_attn = env_flag("PYWHISPERCPP_FLASH_ATTN", false);
    struct whisper_context * ctx = whisper_init_from_file_with_params(path_model, params);""",
)
main.write_text(main_text)
PY

PATH="$PYTHON_DIR:$PATH" \
WHISPER_COREML=1 \
WHISPER_COREML_ALLOW_FALLBACK=1 \
CMAKE_ARGS="-DPython_EXECUTABLE=$PYTHON_CMAKE -DPython_FIND_STRATEGY=LOCATION" \
"$PYTHON" -m pip install --force-reinstall --no-build-isolation --no-cache-dir --no-deps .

if [ ! -d "$BUILD_ROOT/whisper.cpp/.git" ]; then
  git clone https://github.com/ggml-org/whisper.cpp "$BUILD_ROOT/whisper.cpp"
else
  git -C "$BUILD_ROOT/whisper.cpp" pull --ff-only
fi

cd "$BUILD_ROOT/whisper.cpp"
./models/download-ggml-model.sh "$MODEL"
PATH="$PYTHON_DIR:$PATH" ./models/generate-coreml-model.sh "$MODEL"

GGML_MODEL="ggml-$MODEL.bin"
COREML_MODEL="ggml-$MODEL-encoder.mlmodelc"

cp "models/$GGML_MODEL" "$MODEL_DIR/$GGML_MODEL"
rm -rf "$MODEL_DIR/$COREML_MODEL"
cp -R "models/$COREML_MODEL" "$MODEL_DIR/$COREML_MODEL"

"$PYTHON" - <<'PY'
from pywhispercpp.model import Model

print(Model.system_info())
PY

cat <<EOF

CoreML whisper.cpp assets are ready:
  $MODEL_DIR/$GGML_MODEL
  $MODEL_DIR/$COREML_MODEL

To force the app to use this path:
  export LECTURE_PROCESSOR_WHISPER_CPP_MODEL_DIR="$MODEL_DIR"
  export LECTURE_PROCESSOR_REQUIRE_WHISPER_CPP_COREML=1

The desktop app will automatically use .models/whisper-cpp when the requested
model is present there. It also disables whisper.cpp flash attention by default
because that path can crash inside Metal on Apple Silicon.
EOF
