"""Shared validation and filesystem utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def image_paths(directory: str | Path, recursive: bool = True) -> list[Path]:
    """Return sorted image paths from ``directory``."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    iterator = root.rglob("*") if recursive else root.glob("*")
    paths = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"No supported images found in {root}")
    return paths


def as_2d_float(array: np.ndarray | Sequence[Sequence[float]], name: str) -> np.ndarray:
    """Validate and convert an embedding matrix to finite ``float64`` values."""
    result = np.asarray(array, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array; got shape {result.shape}")
    if result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return result


def normalize_rows(array: np.ndarray | Sequence[Sequence[float]], name: str = "embeddings") -> np.ndarray:
    """L2-normalize every row of an embedding matrix."""
    result = as_2d_float(array, name)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{name} contains one or more zero-norm rows")
    return result / norms


def validate_matching_dimensions(real: np.ndarray, synthetic: np.ndarray) -> None:
    if real.shape[1] != synthetic.shape[1]:
        raise ValueError(
            "Real and synthetic embeddings must have the same dimension; "
            f"got {real.shape[1]} and {synthetic.shape[1]}"
        )


def parse_key_value_specs(values: Iterable[str]) -> dict[str, Path]:
    """Parse repeated ``KEY=PATH`` command-line values."""
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected KEY=PATH, received {value!r}")
        key, raw_path = value.split("=", 1)
        key = key.strip()
        if not key or key in parsed:
            raise ValueError(f"Invalid or duplicate key in {value!r}")
        parsed[key] = Path(raw_path).expanduser()
    return parsed
