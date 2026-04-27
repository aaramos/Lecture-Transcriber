#!/usr/bin/env sh

# Source this file from the repo root:
# . ./scripts/dev-env.sh

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

export VIRTUAL_ENV="$PROJECT_ROOT/.venv"
export RUSTUP_HOME="$PROJECT_ROOT/.tools/rustup"
export CARGO_HOME="$PROJECT_ROOT/.tools/cargo"
export PATH="$PROJECT_ROOT/.venv/bin:$PROJECT_ROOT/.tools/bin:$PROJECT_ROOT/.tools/darwin_arm64:$PROJECT_ROOT/.tools/cargo/bin:$PATH"

if [ -x "/Applications/Codex.app/Contents/Resources/node" ]; then
  node() {
    /Applications/Codex.app/Contents/Resources/node "$@"
  }

  npm() {
    /Applications/Codex.app/Contents/Resources/node "$PROJECT_ROOT/.tools/npm/bin/npm-cli.js" "$@"
  }

  npx() {
    /Applications/Codex.app/Contents/Resources/node "$PROJECT_ROOT/.tools/npm/bin/npx-cli.js" "$@"
  }
fi

echo "Lecture Processor dev environment loaded."
