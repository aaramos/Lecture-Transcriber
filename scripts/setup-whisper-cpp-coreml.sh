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

mkdir -p "$MODEL_DIR" "$BUILD_ROOT"

"$PYTHON" -m pip install --upgrade pip setuptools wheel build coremltools ane_transformers openai-whisper

if [ ! -d "$BUILD_ROOT/pywhispercpp/.git" ]; then
  git clone --recursive https://github.com/abdeladim-s/pywhispercpp "$BUILD_ROOT/pywhispercpp"
else
  git -C "$BUILD_ROOT/pywhispercpp" pull --ff-only
  git -C "$BUILD_ROOT/pywhispercpp" submodule update --init --recursive
fi

cd "$BUILD_ROOT/pywhispercpp"
WHISPER_COREML=1 "$PYTHON" -m pip install --force-reinstall .

if [ ! -d "$BUILD_ROOT/whisper.cpp/.git" ]; then
  git clone https://github.com/ggml-org/whisper.cpp "$BUILD_ROOT/whisper.cpp"
else
  git -C "$BUILD_ROOT/whisper.cpp" pull --ff-only
fi

cd "$BUILD_ROOT/whisper.cpp"
./models/download-ggml-model.sh "$MODEL"
./models/generate-coreml-model.sh "$MODEL"

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
EOF
