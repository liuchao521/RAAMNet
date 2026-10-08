"""Aggregate OA, Kappa, F1, Precision, and Recall across seed result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RAAM-Net seed results")
    parser.add_argument("--results", nargs="+", required=True, help="one or more seed*.json files")
    args = parser.parse_args()
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.results]
    if not records:
        raise ValueError("no result files were supplied")
    metrics = ("oa", "kappa", "f1", "precision", "recall")
    print(f"Seeds: {len(records)}")
    print("| Metric | Mean ± sample std. (%) |")
    print("|---|---:|")
    for name in metrics:
        values = np.asarray([record["metrics"][name] for record in records], dtype=np.float64) * 100.0
        std = values.std(ddof=1) if values.size > 1 else 0.0
        print(f"| {name.upper()} | {values.mean():.3f} ± {std:.3f} |")


if __name__ == "__main__":
    main()
