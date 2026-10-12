#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-wpb-owner-search:noetic}"

cd "$ROOT_DIR"
docker build -t "$IMAGE_NAME" .

printf 'Built image: %s\n' "$IMAGE_NAME"
