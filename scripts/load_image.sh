#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="${1:-$ROOT_DIR/wpb-owner-search-noetic.tar.gz}"

if [ ! -f "$ARCHIVE" ]; then
  printf 'Image archive not found: %s\n' "$ARCHIVE" >&2
  exit 1
fi

if [ -f "$ARCHIVE.sha256" ]; then
  sha256sum -c "$ARCHIVE.sha256"
fi

gunzip -c "$ARCHIVE" | docker load
