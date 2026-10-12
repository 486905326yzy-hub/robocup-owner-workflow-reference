#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
docker run --rm -it --gpus all \
  -v "$ROOT_DIR:/workspace/person_reid_test" \
  -w /workspace/person_reid_test \
  wpb-person-reid:test "$@"
