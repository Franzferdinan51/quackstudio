#!/bin/sh
# Stage a built quackd binary as the Tauri sidecar the desktop app
# spawns (see src-tauri/tauri.conf.json: bundle.externalBin).
#
# Usage: scripts/stage-sidecar.sh [path-to-quackd]
#
# Default source: target/release/quackd, falling back to
# target/debug/quackd. The destination name carries the target
# triple Tauri expects sidecars to be named with on Apple Silicon
# Macs; Tauri strips the suffix again when bundling.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DEST="$ROOT/src-tauri/binaries/quackd-aarch64-apple-darwin"

if [ $# -ge 1 ]; then
  SRC=$1
elif [ -f "$ROOT/target/release/quackd" ]; then
  SRC="$ROOT/target/release/quackd"
else
  SRC="$ROOT/target/debug/quackd"
fi

if [ ! -f "$SRC" ]; then
  echo "stage-sidecar: no quackd binary at $SRC (build it first: cargo build -p quackd)" >&2
  exit 1
fi

mkdir -p "$ROOT/src-tauri/binaries"
cp "$SRC" "$DEST"
chmod +x "$DEST"
echo "staged quackd sidecar: $SRC -> $DEST"
