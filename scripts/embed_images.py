#!/usr/bin/env python3
"""Encode an image directory with BiomedCLIP and save a portable NPZ file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from synthetic_data_evaluation.embedding import BiomedCLIPEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, help="Optional fine-tuned visual-encoder weights")
    parser.add_argument("--device", help="Torch device, for example cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    encoder = BiomedCLIPEncoder(weights=args.weights, device=args.device, batch_size=args.batch_size)
    embeddings, paths = encoder.encode_directory(args.images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    relative_paths = [str(path.relative_to(args.images.resolve())) for path in paths]
    np.savez_compressed(args.output, embeddings=embeddings, paths=np.asarray(relative_paths))
    print(f"Saved {len(paths)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
