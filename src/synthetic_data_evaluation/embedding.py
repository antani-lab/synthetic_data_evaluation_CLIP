"""BiomedCLIP image embedding utilities.

The heavy model dependencies are imported lazily so the numerical metric
functions remain usable with precomputed ``.npy`` embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .core import image_paths, normalize_rows


DEFAULT_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


@dataclass
class BiomedCLIPEncoder:
    """Load BiomedCLIP and encode images as unit-normalized vectors."""

    model_name: str = DEFAULT_MODEL
    weights: str | Path | None = None
    device: str | None = None
    batch_size: int = 32

    def __post_init__(self) -> None:
        try:
            import torch
            import torch.nn.functional as functional
            from open_clip import create_model_from_pretrained
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "BiomedCLIP embedding requires torch and open-clip-torch. "
                "Install the project with the 'biomedclip' extra."
            ) from exc

        self._torch = torch
        self._functional = functional
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = create_model_from_pretrained(
            self.model_name,
            return_transform=True,
        )
        if self.weights is not None:
            state = torch.load(Path(self.weights), map_location="cpu")
            self.model.visual.load_state_dict(state)
        self.model = self.model.to(self.device).eval()

    def encode(self, paths: Sequence[str | Path]) -> np.ndarray:
        """Encode image paths in batches and return normalized embeddings."""
        from PIL import Image

        if not paths:
            raise ValueError("At least one image path is required")
        batches: list[np.ndarray] = []
        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start : start + self.batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            pixels = self._torch.stack([self.preprocess(image) for image in images]).to(self.device)
            with self._torch.inference_mode():
                features = self.model.encode_image(pixels)
                features = self._functional.normalize(features, dim=-1)
            batches.append(features.cpu().numpy())
        return normalize_rows(np.concatenate(batches, axis=0))

    def encode_directory(self, directory: str | Path) -> tuple[np.ndarray, list[Path]]:
        paths = image_paths(directory)
        return self.encode(paths), paths


def load_embeddings(path: str | Path, encoder: BiomedCLIPEncoder | None = None) -> np.ndarray:
    """Load ``.npy``/``.npz`` embeddings or encode an image directory."""
    source = Path(path).expanduser()
    if source.is_file() and source.suffix.lower() == ".npy":
        return normalize_rows(np.load(source), source.name)
    if source.is_file() and source.suffix.lower() == ".npz":
        archive = np.load(source)
        if "embeddings" not in archive:
            raise KeyError(f"{source} must contain an 'embeddings' array")
        return normalize_rows(archive["embeddings"], source.name)
    if source.is_dir():
        if encoder is None:
            raise ValueError("An encoder is required when the input is an image directory")
        embeddings, _ = encoder.encode_directory(source)
        return embeddings
    raise FileNotFoundError(source)
