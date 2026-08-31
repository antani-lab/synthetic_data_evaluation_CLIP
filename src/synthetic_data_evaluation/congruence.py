"""Congruence measures used in the CMIG manuscript."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .core import image_paths


def peak_signal_to_noise_ratio(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    data_range: float = 1.0,
) -> float:
    """Return PSNR for a meaningful, pixel-aligned image pair.

    The arrays must have identical shapes. Infinite PSNR is returned for an
    exact match. In this study PSNR is interpretable only for image-to-image
    outputs with a direct source-image reference.
    """
    first = np.asarray(reference, dtype=np.float64)
    second = np.asarray(generated, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(f"Paired images must have the same shape: {first.shape} != {second.shape}")
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    error = float(np.mean((first - second) ** 2))
    if error == 0:
        return float("inf")
    return float(10.0 * np.log10((data_range**2) / error))


def paired_directory_psnr(
    reference_directory: str | Path,
    generated_directory: str | Path,
) -> np.ndarray:
    """Compute PSNR for images paired by relative filename."""
    from PIL import Image

    reference_root = Path(reference_directory).expanduser().resolve()
    generated_root = Path(generated_directory).expanduser().resolve()
    reference_paths = image_paths(reference_root)
    generated_by_name = {
        path.relative_to(generated_root).as_posix(): path for path in image_paths(generated_root)
    }
    scores: list[float] = []
    missing: list[str] = []
    for reference_path in reference_paths:
        key = reference_path.relative_to(reference_root).as_posix()
        generated_path = generated_by_name.get(key)
        if generated_path is None:
            missing.append(key)
            continue
        reference = np.asarray(Image.open(reference_path).convert("F"), dtype=np.float64)
        generated = np.asarray(Image.open(generated_path).convert("F"), dtype=np.float64)
        scores.append(peak_signal_to_noise_ratio(reference, generated, data_range=255.0))
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} generated counterparts, including: {preview}")
    return np.asarray(scores, dtype=np.float64)


def clean_fid_kid(
    real_directory: str | Path,
    synthetic_directory: str | Path,
    *,
    mode: str = "clean",
    num_workers: int = 0,
) -> dict[str, float]:
    """Compute conventional Inception-feature FID and KID with CleanFID."""
    try:
        from cleanfid import fid
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install the project with the 'metrics' extra to compute FID/KID") from exc

    real = str(Path(real_directory).expanduser().resolve())
    synthetic = str(Path(synthetic_directory).expanduser().resolve())
    fid_value = fid.compute_fid(real, synthetic, mode=mode, num_workers=num_workers)
    kid_value = fid.compute_kid(real, synthetic, mode=mode, num_workers=num_workers)
    return {"fid": float(fid_value), "kid": float(kid_value)}


def clip_prompt_similarity(
    image_files: Sequence[str | Path],
    *,
    prompt: str = "pneumonia, chest x-ray",
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
    device: str | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Return raw image-text cosine similarities for the fixed CLIP prompt."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    try:
        import open_clip
        import torch
        import torch.nn.functional as functional
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install the project with the 'metrics' extra to compute CLIP similarity") from exc

    paths = [Path(path).expanduser() for path in image_files]
    if not paths:
        raise ValueError("At least one image is required")
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=resolved_device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    with torch.inference_mode():
        text = tokenizer([prompt]).to(resolved_device)
        text_features = functional.normalize(model.encode_text(text), dim=-1)
        values: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            pixels = torch.stack(
                [preprocess(Image.open(path).convert("RGB")) for path in paths[start : start + batch_size]]
            ).to(resolved_device)
            image_features = functional.normalize(model.encode_image(pixels), dim=-1)
            values.append((image_features @ text_features.T).squeeze(1).cpu().numpy())
    return np.concatenate(values).astype(np.float64, copy=False)
