from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


class ReIDModel:
    """OSNet-AIN feature extractor for already cropped full-body images.

    The upper part of every crop is masked by default. This deliberately prevents
    the embedding from using face pixels as an identity shortcut.
    """

    def __init__(self, model_name: str, weights: str | Path, device: str = 'auto',
                 image_height: int = 256, image_width: int = 128,
                 mask_head: bool = True, head_mask_ratio: float = 0.22):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable')
        self.device = device
        self.mask_head = mask_head
        self.head_mask_ratio = head_mask_ratio
        self.image_size = (image_height, image_width)
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(
                f'OSNet-AIN ReID weight file not found: {weights}. '
                'Run scripts/download_weights.py first.'
            )

        try:
            from torchreid.utils import FeatureExtractor
        except ImportError as error:
            raise RuntimeError('torchreid is not installed; build the isolated image first') from error

        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=str(weights),
            image_size=self.image_size,
            device=self.device,
            verbose=False,
        )

    def _without_face(self, image: Image.Image) -> Image.Image:
        image = image.convert('RGB').copy()
        if self.mask_head:
            height = image.height
            cutoff = max(1, int(height * self.head_mask_ratio))
            # ImageNet normalization mean in RGB, so masked pixels stay neutral.
            image.paste((123, 116, 103), (0, 0, image.width, cutoff))
        return image

    def extract_embedding(self, image: Image.Image | np.ndarray | str | Path) -> np.ndarray:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8)).convert('RGB')
        if not isinstance(image, Image.Image):
            raise TypeError('image must be a PIL image, RGB numpy image, or image path')
        feature = self.extractor([self._without_face(image)])[0].detach().cpu().numpy()
        norm = float(np.linalg.norm(feature))
        if norm <= 0:
            raise RuntimeError('model returned a zero embedding')
        return (feature / norm).astype(np.float32)
