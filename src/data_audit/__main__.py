"""CLI for running standalone data quality audits on any supported format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .null_patterns import audit_null_patterns
from .outliers import audit_within_group_outliers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deep data quality auditing CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_null = sub.add_parser("nulls", help="Audit per-column and joint co-missingness patterns")
    p_null.add_argument("source", help="Input file, directory, glob, or DB URI")
    p_null.add_argument("--group-by", nargs="*", default=())
    p_null.add_argument("--out", type=Path, default=None)

    p_out = sub.add_parser("outliers", help="Audit within-group robust (log-MAD) outliers")
    p_out.add_argument("source")
    p_out.add_argument("--entity-col", required=True)
    p_out.add_argument("--value-col", required=True)
    p_out.add_argument("--group-cols", nargs="*", default=())
    p_out.add_argument("--time-col", default=None)
    p_out.add_argument("--min-group-n", type=int, default=20)
    p_out.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "nulls":
        res = audit_null_patterns(args.source, group_by=args.group_by, output_dir=args.out)
        print(json.dumps(res["summary"], ensure_ascii=False, indent=2))
        return 0
    if args.command == "outliers":
        res = audit_within_group_outliers(
            args.source,
            entity_col=args.entity_col,
            value_col=args.value_col,
            group_cols=args.group_cols,
            time_col=args.time_col,
            min_group_n=args.min_group_n,
            output_dir=args.out,
        )
        print(json.dumps({"methodology": res["methodology"], "flagged_rows": len(res["top_flagged"])},
                         ensure_ascii=False, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
