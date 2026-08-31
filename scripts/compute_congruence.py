#!/usr/bin/env python3
"""Compute conventional congruence measures for one real/synthetic comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from synthetic_data_evaluation.congruence import (
    clean_fid_kid,
    clip_prompt_similarity,
    paired_directory_psnr,
)
from synthetic_data_evaluation.core import image_paths


def _summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True, type=Path)
    parser.add_argument("--synthetic", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--paired-psnr", action="store_true", help="Use only for paired image-to-image outputs")
    parser.add_argument("--prompt", default="pneumonia, chest x-ray")
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    result: dict[str, object] = clean_fid_kid(
        args.real,
        args.synthetic,
        num_workers=args.num_workers,
    )
    clip_values = clip_prompt_similarity(
        image_paths(args.synthetic),
        prompt=args.prompt,
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
        device=args.device,
        batch_size=args.batch_size,
    )
    result["clip_prompt"] = args.prompt
    result["clip_similarity"] = _summary(clip_values)
    if args.paired_psnr:
        result["psnr"] = _summary(paired_directory_psnr(args.real, args.synthetic))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved congruence results to {args.output}")


if __name__ == "__main__":
    main()
