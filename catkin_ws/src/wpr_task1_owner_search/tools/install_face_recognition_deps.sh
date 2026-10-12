#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --user --no-cache-dir \
  easydict \
  prettytable \
  scikit-image==0.21.0 \
  scikit-learn==1.3.2 \
  onnx==1.16.2 \
  qudida==0.0.4

python3 -m pip install --user --no-cache-dir --no-deps albumentations==1.3.1

python3 -m pip install --user --no-cache-dir --no-deps insightface==0.7.3

MODEL_ROOT=/home/ubuntu20/.insightface/models
MODEL_DIR="$MODEL_ROOT/buffalo_sc"
MODEL_ZIP="$MODEL_ROOT/buffalo_sc.zip"
MODEL_TMP="$MODEL_ROOT/buffalo_sc.zip.tmp"
mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_DIR/det_500m.onnx" ] || [ ! -f "$MODEL_DIR/w600k_mbf.onnx" ]; then
  wget -c --timeout=30 --tries=10 -O "$MODEL_TMP" \
    https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip
  unzip -tq "$MODEL_TMP"
  unzip -qo "$MODEL_TMP" -d "$MODEL_DIR"
  mv "$MODEL_TMP" "$MODEL_ZIP"
fi

python3 - <<'PY'
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_sc', root='/home/ubuntu20/.insightface', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(320, 320))
print('InsightFace buffalo_sc is ready under /home/ubuntu20/.insightface')
PY
