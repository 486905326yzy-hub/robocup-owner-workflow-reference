#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
docker build -q -t wpb-person-reid:test -f "$ROOT_DIR/Dockerfile" "$ROOT_DIR"
