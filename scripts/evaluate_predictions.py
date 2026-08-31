#!/usr/bin/env python3
"""Compare one condition with a matched real-only baseline using paired bootstrap."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_data_evaluation.downstream import (
    classification_metrics,
    paired_stratified_bootstrap,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--condition", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--id-column", default="image_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--probability-columns", nargs="+", required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline)
    condition = pd.read_csv(args.condition)
    required = [args.id_column, args.label_column, *args.probability_columns]
    for name, frame in (("baseline", baseline), ("condition", condition)):
        missing = set(required).difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {', '.join(sorted(missing))}")
        if frame[args.id_column].duplicated().any():
            raise ValueError(f"{name} contains duplicate image identifiers")

    merged = baseline[required].merge(
        condition[required],
        on=args.id_column,
        how="inner",
        validate="one_to_one",
        suffixes=("_baseline", "_condition"),
    )
    if len(merged) != len(baseline) or len(merged) != len(condition):
        raise ValueError("Baseline and condition files do not contain the same image identifiers")
    baseline_labels = merged[f"{args.label_column}_baseline"].to_numpy()
    condition_labels = merged[f"{args.label_column}_condition"].to_numpy()
    if not np.array_equal(baseline_labels, condition_labels):
        raise ValueError("Class labels differ between baseline and condition files")
    classes = np.unique(baseline_labels)
    baseline_probabilities = merged[
        [f"{column}_baseline" for column in args.probability_columns]
    ].to_numpy()
    condition_probabilities = merged[
        [f"{column}_condition" for column in args.probability_columns]
    ].to_numpy()

    baseline_metrics = classification_metrics(baseline_labels, baseline_probabilities, classes=classes)
    metric_names = list(baseline_metrics)
    rows = []
    for metric_name in metric_names:
        def metric(labels: np.ndarray, probabilities: np.ndarray, name: str = metric_name) -> float:
            return classification_metrics(labels, probabilities, classes=classes)[name]

        summary = paired_stratified_bootstrap(
            baseline_labels,
            baseline_probabilities,
            condition_probabilities,
            metric,
            n_resamples=args.resamples,
            seed=args.seed,
        )
        rows.append({"metric": metric_name, **summary})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} paired metric comparisons to {args.output}")


if __name__ == "__main__":
    main()
