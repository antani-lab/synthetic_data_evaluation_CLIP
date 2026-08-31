#!/usr/bin/env python3
"""Compute centroid-based constraint and consistency summaries (Algorithm S2)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from synthetic_data_evaluation.constraint_consistency import constraint_consistency
from synthetic_data_evaluation.embedding import BiomedCLIPEncoder, load_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True, type=Path, help="Real-reference embeddings or image directory")
    parser.add_argument("--synthetic", required=True, type=Path, help="Synthetic embeddings or image directory")
    parser.add_argument("--generator", required=True)
    parser.add_argument("--severity", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--leave-one-out-real", action="store_true")
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
    result = constraint_consistency(
        load_embeddings(args.real, encoder),
        load_embeddings(args.synthetic, encoder),
        generator=args.generator,
        severity=args.severity,
        leave_one_out_real=args.leave_one_out_real,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(args.output, index=False)
    print(f"Saved constraint/consistency results to {args.output}")


if __name__ == "__main__":
    main()
