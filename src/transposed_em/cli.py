"""Fit a binary response CSV and export estimates and diagnostics."""

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from . import LaplaceIRT


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Fit the preprint's Laplace 1PL or 2PL model."
    )
    parser.add_argument(
        "responses",
        type=Path,
        help="CSV: first column subject ID; other columns binary items",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="new or empty output directory"
    )
    parser.add_argument("--model", choices=["1pl", "2pl"], default="2pl")
    parser.add_argument("--covariance", choices=["diagonal", "full"])
    parser.add_argument("--max-iterations", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error(
            "output must be a new or empty directory; existing results are not overwritten"
        )
    try:
        matrix = pd.read_csv(args.responses, index_col=0)
        fit = LaplaceIRT(
            model=args.model, covariance=args.covariance, max_iterations=args.max_iterations
        ).fit(matrix)
    except (OSError, ValueError, FloatingPointError) as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    fit.subject_frame().to_csv(args.output / "subjects.csv", index=False)
    fit.item_frame().to_csv(args.output / "items.csv", index=False)
    fit.history_frame().to_csv(args.output / "history.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps(_json_safe(fit.summary()), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"{fit.stopping_reason_}: {len(fit.history_)} iterations; results in {args.output}"
    )


if __name__ == "__main__":
    main()
