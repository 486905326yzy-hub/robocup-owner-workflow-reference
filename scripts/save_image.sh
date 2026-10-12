#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-wpb-owner-search:noetic}"
OUT="${OUT:-$ROOT_DIR/wpb-owner-search-noetic.tar}"

docker save "$IMAGE_NAME" -o "$OUT"
gzip -f "$OUT"
sha256sum "$OUT.gz" >"$OUT.gz.sha256"

printf 'Saved image archive: %s.gz\n' "$OUT"
printf 'Checksum file: %s.gz.sha256\n' "$OUT"
