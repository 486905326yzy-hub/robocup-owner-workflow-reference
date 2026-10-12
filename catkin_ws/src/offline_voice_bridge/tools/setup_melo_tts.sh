#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MELO_ZH_DIR="$ROOT_DIR/models/melo/zh"
mkdir -p "$MELO_ZH_DIR"

python3 -m pip install --user --upgrade \
  importlib-metadata 'transformers==4.27.4' 'tokenizers==0.13.3' \
  'pypinyin==0.50.0' 'cn2an==0.5.22' 'jieba==0.42.1' \
  'soundfile==0.13.1' 'numba==0.58.1' 'llvmlite==0.41.1' \
  'g2p_en==2.1.0' 'eng_to_ipa==0.0.2' 'inflect==7.0.0' \
  'unidecode==1.3.7' 'anyascii==0.3.2' 'num2words==0.5.12'

download_required() {
  local url="$1"
  local output="$2"
  local label="$3"
  if [ -s "$output" ]; then
    echo "$label already exists: $output"
    return
  fi
  curl --fail -L -C - --retry 20 --retry-delay 3 --connect-timeout 30 -o "$output" "$url"
  test -s "$output"
  echo "Downloaded $label -> $output"
}

download_required \
  "https://huggingface.co/myshell-ai/MeloTTS-Chinese/resolve/main/config.json" \
  "$MELO_ZH_DIR/config.json" \
  "MeloTTS Chinese config"

download_required \
  "https://huggingface.co/myshell-ai/MeloTTS-Chinese/resolve/main/checkpoint.pth" \
  "$MELO_ZH_DIR/checkpoint.pth" \
  "MeloTTS Chinese checkpoint"

if [ ! -f "$MELO_ZH_DIR/config_no_bert.json" ]; then
  cp "$MELO_ZH_DIR/config.json" "$MELO_ZH_DIR/config_no_bert.json"
fi

python3 - <<PY
import json
path = "$MELO_ZH_DIR/config_no_bert.json"
with open(path, "r", encoding="utf-8") as f:
    cfg = json.load(f)
cfg.setdefault("data", {})["disable_bert"] = True
with open(path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY

echo "MeloTTS Chinese natural voice dependencies are ready."
