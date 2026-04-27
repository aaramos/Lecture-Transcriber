#!/usr/bin/env sh
set -eu

# Rebuilds Pillow from source inside the active Python environment.
# Run this after loading scripts/dev-env.sh when testing local image
# processing performance on Apple Silicon.

PYTHON="${PYTHON:-python3}"

"$PYTHON" -m pip uninstall -y Pillow
"$PYTHON" -m pip install --no-binary=:all: --no-cache-dir --force-reinstall "Pillow>=10.0.0"
"$PYTHON" - <<'PY'
from PIL import Image, features

print(f"Pillow {Image.__version__}")
print(f"jpeg: {features.check('jpg')}")
print(f"libjpeg_turbo: {features.check_feature('libjpeg_turbo')}")
PY
