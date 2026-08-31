#!/usr/bin/env python3
"""Analyze pooled and severity-adjusted quality-utility associations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from synthetic_data_evaluation.downstream import holm_adjust
from synthetic_data_evaluation.quality_utility import (
    generator_cluster_bootstrap,
    spearman_permutation_test,
)


REQUIRED_COLUMNS = {"dimension", "generator", "severity", "quality", "utility"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Long-form CSV with quality and utility values")
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    rows = []
    for dimension, subset in data.groupby("dimension", sort=False):
        for adjusted in (False, True):
            groups = subset["severity"].to_numpy() if adjusted else None
            test = spearman_permutation_test(
                subset["quality"],
                subset["utility"],
                groups=groups,
                n_permutations=args.permutations,
                seed=args.seed,
            )
            lower, upper = generator_cluster_bootstrap(
                subset["quality"],
                subset["utility"],
                subset["generator"],
                groups=groups,
                n_resamples=args.bootstrap_resamples,
                seed=args.seed,
            )
            rows.append(
                {
                    "dimension": dimension,
                    "analysis": "severity_adjusted" if adjusted else "pooled",
                    "n": len(subset),
                    **test,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    results = pd.DataFrame(rows)
    results["holm_adjusted_p_value"] = results.groupby("analysis")["permutation_p_value"].transform(
        lambda values: holm_adjust(values.to_numpy())
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"Saved {len(results)} association estimates to {args.output}")


if __name__ == "__main__":
    main()
