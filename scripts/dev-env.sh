#!/usr/bin/env sh

# Source this file from the repo root:
# . ./scripts/dev-env.sh

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
SYSTEM_NODE="$(command -v node 2>/dev/null || true)"

if [ -n "${LECTURE_PROCESSOR_NODE:-}" ] && [ -x "$LECTURE_PROCESSOR_NODE" ]; then
  NODE_BINARY="$LECTURE_PROCESSOR_NODE"
elif [ -x "/Applications/Codex.app/Contents/Resources/node" ]; then
  NODE_BINARY="/Applications/Codex.app/Contents/Resources/node"
elif [ -n "$SYSTEM_NODE" ] && [ -x "$SYSTEM_NODE" ]; then
  NODE_BINARY="$SYSTEM_NODE"
else
  NODE_BINARY=""
fi

export VIRTUAL_ENV="$PROJECT_ROOT/.venv"
export RUSTUP_HOME="$PROJECT_ROOT/.tools/rustup"
export CARGO_HOME="$PROJECT_ROOT/.tools/cargo"
export PATH="$PROJECT_ROOT/.venv/bin:$PROJECT_ROOT/.tools/bin:$PROJECT_ROOT/.tools/darwin_arm64:$PROJECT_ROOT/.tools/cargo/bin:$PATH"

if [ -n "$NODE_BINARY" ]; then
  mkdir -p "$PROJECT_ROOT/.tools/bin"

  # Recreate these wrappers on every load so a moved or upgraded Node runtime
  # cannot leave the documented build commands pointing at a stale binary.
  cat > "$PROJECT_ROOT/.tools/bin/node" <<EOF
#!/usr/bin/env sh
exec "$NODE_BINARY" "\$@"
EOF
  chmod +x "$PROJECT_ROOT/.tools/bin/node"

  if [ -f "$PROJECT_ROOT/.tools/npm/bin/npm-cli.js" ]; then
    cat > "$PROJECT_ROOT/.tools/bin/npm" <<EOF
#!/usr/bin/env sh
exec "$NODE_BINARY" "$PROJECT_ROOT/.tools/npm/bin/npm-cli.js" "\$@"
EOF
    chmod +x "$PROJECT_ROOT/.tools/bin/npm"
  fi

  if [ -f "$PROJECT_ROOT/.tools/npm/bin/npx-cli.js" ]; then
    cat > "$PROJECT_ROOT/.tools/bin/npx" <<EOF
#!/usr/bin/env sh
exec "$NODE_BINARY" "$PROJECT_ROOT/.tools/npm/bin/npx-cli.js" "\$@"
EOF
    chmod +x "$PROJECT_ROOT/.tools/bin/npx"
  fi

  node() {
    "$NODE_BINARY" "$@"
  }

  npm() {
    "$NODE_BINARY" "$PROJECT_ROOT/.tools/npm/bin/npm-cli.js" "$@"
  }

  npx() {
    "$NODE_BINARY" "$PROJECT_ROOT/.tools/npm/bin/npx-cli.js" "$@"
  }
fi

echo "Lecture Processor dev environment loaded."
