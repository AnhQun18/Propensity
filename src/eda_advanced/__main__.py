"""CLI for running advanced EDA scans (correlations, target signal, drift, categorical profiles)."""

from __future__ import annotations

import argparse
from pathlib import Path

from .categorical import periodic_categorical_profile
from .drift import compare_distributions
from .relationships import correlation_analysis, target_signal_scan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Advanced EDA CLI (correlations, target signal, drift, profiles).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_corr = sub.add_parser("corr", help="Compute correlation matrix and top correlated pairs")
    p_corr.add_argument("source")
    p_corr.add_argument("--method", default="spearman", choices=("pearson", "spearman", "kendall"))
    p_corr.add_argument("--threshold", type=float, default=0.5)
    p_corr.add_argument("--out", type=Path, default=None)

    p_sig = sub.add_parser("target-scan", help="Scan univariate IV/AUC/KS/Cramer's V against a binary target")
    p_sig.add_argument("source")
    p_sig.add_argument("--target", required=True)
    p_sig.add_argument("--out", type=Path, default=None)

    p_drift = sub.add_parser("drift", help="Compare distributions (PSI/KS/Wasserstein/JS) between 2 datasets")
    p_drift.add_argument("reference")
    p_drift.add_argument("current")
    p_drift.add_argument("--out", type=Path, default=None)

    p_cat = sub.add_parser("categorical-profile", help="Periodic entropy/HHI/novelty/switching profile")
    p_cat.add_argument("source")
    p_cat.add_argument("--entity-col", required=True)
    p_cat.add_argument("--time-col", required=True)
    p_cat.add_argument("--category-col", required=True)
    p_cat.add_argument("--period", default="month", choices=("day", "week", "month", "quarter", "year"))
    p_cat.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "corr":
        res = correlation_analysis(args.source, method=args.method, threshold=args.threshold)
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            res["matrix"].to_csv(args.out / "correlation_matrix.csv")
            res["top_pairs"].to_csv(args.out / "top_correlated_pairs.csv", index=False)
        else:
            print(res["top_pairs"].to_string(index=False))
        return 0
    if args.command == "target-scan":
        df = target_signal_scan(args.source, target_col=args.target)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.out, index=False)
        else:
            print(df.to_string(index=False))
        return 0
    if args.command == "drift":
        df = compare_distributions(args.reference, args.current)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.out, index=False)
        else:
            print(df.to_string(index=False))
        return 0
    if args.command == "categorical-profile":
        df = periodic_categorical_profile(
            args.source, entity_col=args.entity_col, time_col=args.time_col,
            category_col=args.category_col, period=args.period,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.suffix.lower() == ".csv":
            df.to_csv(args.out, index=False)
        else:
            df.to_parquet(args.out, index=False)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
