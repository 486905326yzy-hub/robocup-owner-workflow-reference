#!/usr/bin/env python3
"""Download the official OSNet-AIN MSMT17 ReID checkpoint.

The source URL is intentionally explicit and can be replaced with a locally
verified checkpoint through config.yaml when competition network is unavailable.
"""
from pathlib import Path
import sys
import urllib.request

TARGET = Path(__file__).resolve().parents[1] / 'models' / 'osnet_ain_x1_0_msmt17.pth'
# Official Torchreid model-zoo checkpoint (MSMT17, OSNet-AIN x1.0).
URL = 'https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0/osnet_ain_x1_0_msmt17.pth'

TARGET.parent.mkdir(parents=True, exist_ok=True)
if TARGET.exists() and TARGET.stat().st_size > 10_000_000:
    print(f'Already present: {TARGET}')
    sys.exit(0)
print(f'Downloading official checkpoint to {TARGET}')
urllib.request.urlretrieve(URL, TARGET)
print(f'Downloaded {TARGET.stat().st_size / 1024 / 1024:.1f} MiB')
