#!/usr/bin/env python3
"""Compute the manuscript's BiomedCLIP coverage curve (Algorithm S1)."""

from __future__ import annotations

import argparse
from pathlib import Path

from synthetic_data_evaluation.coverage import DEFAULT_THRESHOLDS, coverage_curve
from synthetic_data_evaluation.embedding import BiomedCLIPEncoder, load_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True, type=Path, help="Real-reference embeddings or image directory")
    parser.add_argument("--synthetic", required=True, type=Path, help="Synthetic embeddings or image directory")
    parser.add_argument("--generator", required=True)
    parser.add_argument("--severity", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--include-boundary", action="store_true", help="Use distance <= threshold")
    parser.add_argument("--weights", type=Path, help="Optional fine-tuned BiomedCLIP visual weights")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    needs_encoder = args.real.is_dir() or args.synthetic.is_dir()
    encoder = (
        BiomedCLIPEncoder(weights=args.weights, device=args.device, batch_size=args.batch_size)
        if needs_encoder
        else None
    )
    real = load_embeddings(args.real, encoder)
    synthetic = load_embeddings(args.synthetic, encoder)
    result = coverage_curve(
        real,
        synthetic,
        thresholds=args.thresholds,
        generator=args.generator,
        severity=args.severity,
        strict=not args.include_boundary,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} coverage points to {args.output}")


if __name__ == "__main__":
    main()
