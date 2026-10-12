#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPER_MODEL_DIR="$ROOT_DIR/models/piper"
WHISPER_MODEL_DIR="$ROOT_DIR/models/whisper"
PIPER_BIN_DIR="$ROOT_DIR/tools/piper"

mkdir -p "$PIPER_MODEL_DIR" "$WHISPER_MODEL_DIR" "$PIPER_BIN_DIR"

download_optional() {
  local url="$1"
  local output="$2"
  local label="$3"
  if curl -L --connect-timeout 30 --retry 3 --retry-delay 3 "$url" -o "$output"; then
    return 0
  fi
  echo "WARN: failed to download $label from $url" >&2
  rm -f "$output"
  return 1
}

python3 -m pip install --user --upgrade \
  tokenizers==0.15.2 \
  huggingface-hub==0.23.5 \
  faster-whisper==1.0.3

if [ ! -x "$PIPER_BIN_DIR/piper" ]; then
  tmp_tar="$(mktemp /tmp/piper_linux_x86_64.XXXXXX.tar.gz)"
  if download_optional "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz" "$tmp_tar" "Piper Linux binary"; then
    tar -xzf "$tmp_tar" -C "$PIPER_BIN_DIR" --strip-components=1
    chmod +x "$PIPER_BIN_DIR/piper"
  fi
  rm -f "$tmp_tar"
fi

MODEL_BASE="${PIPER_VOICE_BASE:-https://hf-mirror.com/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium}"
if [ ! -f "$PIPER_MODEL_DIR/en_US-lessac-medium.onnx" ]; then
  download_optional "$MODEL_BASE/en_US-lessac-medium.onnx" "$PIPER_MODEL_DIR/en_US-lessac-medium.onnx" "Piper lessac model" || true
fi
if [ ! -f "$PIPER_MODEL_DIR/en_US-lessac-medium.onnx.json" ]; then
  download_optional "$MODEL_BASE/en_US-lessac-medium.onnx.json" "$PIPER_MODEL_DIR/en_US-lessac-medium.onnx.json" "Piper lessac config" || true
fi

if [ -x "$PIPER_BIN_DIR/piper" ] && [ -f "$PIPER_MODEL_DIR/en_US-lessac-medium.onnx" ] && [ -f "$PIPER_MODEL_DIR/en_US-lessac-medium.onnx.json" ]; then
  echo "Offline voice dependencies and Piper model are ready."
else
  echo "Offline ASR dependencies are ready. Piper is incomplete; TTS node will use flite fallback until Piper files are present."
fi
echo "Next: cd ~/catkin_ws && catkin_make && source devel/setup.bash"
