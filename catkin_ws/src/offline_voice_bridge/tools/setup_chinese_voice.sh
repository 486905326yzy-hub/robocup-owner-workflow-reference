#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPER_MODEL_DIR="$ROOT_DIR/models/piper"
PIPER_BIN_DIR="$ROOT_DIR/tools/piper"
WHISPER_ZH_DIR="$ROOT_DIR/models/whisper/faster-whisper-small"

mkdir -p "$PIPER_MODEL_DIR" "$PIPER_BIN_DIR" "$WHISPER_ZH_DIR"

download_required() {
  local url="$1"
  local output="$2"
  local label="$3"
  local tmp_output="$output.tmp"
  rm -f "$tmp_output"
  curl --fail -L --connect-timeout 30 --retry 3 --retry-delay 3 "$url" -o "$tmp_output"
  test -s "$tmp_output"
  mv "$tmp_output" "$output"
  echo "Downloaded $label -> $output"
}

python3 -m pip install --user --upgrade \
  tokenizers==0.15.2 \
  huggingface-hub==0.23.5 \
  faster-whisper==1.0.3

if [ ! -x "$PIPER_BIN_DIR/piper" ]; then
  tmp_tar="$(mktemp /tmp/piper_linux_x86_64.XXXXXX.tar.gz)"
  curl --fail -L --connect-timeout 30 --retry 3 --retry-delay 3 \
    "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz" \
    -o "$tmp_tar"
  tar -xzf "$tmp_tar" -C "$PIPER_BIN_DIR" --strip-components=1
  rm -f "$tmp_tar"
  chmod +x "$PIPER_BIN_DIR/piper"
fi

MODEL_BASE="${PIPER_ZH_VOICE_BASE:-https://hf-mirror.com/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium}"
if [ ! -f "$PIPER_MODEL_DIR/zh_CN-huayan-medium.onnx" ]; then
  download_required "$MODEL_BASE/zh_CN-huayan-medium.onnx" "$PIPER_MODEL_DIR/zh_CN-huayan-medium.onnx" "Piper Chinese huayan model"
fi
if [ ! -f "$PIPER_MODEL_DIR/zh_CN-huayan-medium.onnx.json" ]; then
  download_required "$MODEL_BASE/zh_CN-huayan-medium.onnx.json" "$PIPER_MODEL_DIR/zh_CN-huayan-medium.onnx.json" "Piper Chinese huayan config"
fi

WHISPER_BASE="${WHISPER_ZH_BASE:-https://huggingface.co/Systran/faster-whisper-small/resolve/main}"
if [ ! -f "$WHISPER_ZH_DIR/config.json" ]; then
  download_required "$WHISPER_BASE/config.json" "$WHISPER_ZH_DIR/config.json" "faster-whisper-small config"
fi
if [ ! -f "$WHISPER_ZH_DIR/tokenizer.json" ]; then
  download_required "$WHISPER_BASE/tokenizer.json" "$WHISPER_ZH_DIR/tokenizer.json" "faster-whisper-small tokenizer"
fi
if [ ! -f "$WHISPER_ZH_DIR/vocabulary.txt" ]; then
  download_required "$WHISPER_BASE/vocabulary.txt" "$WHISPER_ZH_DIR/vocabulary.txt" "faster-whisper-small vocabulary"
fi
if [ ! -f "$WHISPER_ZH_DIR/model.bin" ]; then
  download_required "$WHISPER_BASE/model.bin" "$WHISPER_ZH_DIR/model.bin" "faster-whisper-small model"
fi

echo "Chinese offline voice dependencies are ready."
