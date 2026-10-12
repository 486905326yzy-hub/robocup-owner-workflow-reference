from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else project_root() / 'config.yaml'
    with path.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root() / path


def list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob('*')
                  if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
