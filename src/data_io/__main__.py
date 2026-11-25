"""CLI for inspecting or converting multi-format datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .integrity import json_safe
from .inventory import inspect_inventory
from .writers import write_table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-format dataset inspection and atomic conversion.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="Inspect files, row counts, and schemas")
    p_inspect.add_argument("source", help="File, directory, glob, or db URI (e.g. data.db::orders)")
    p_inspect.add_argument("--no-count", action="store_true", help="Skip row counting on non-Parquet formats")

    p_conv = sub.add_parser("convert", help="Atomically convert a dataset between formats")
    p_conv.add_argument("source", help="Input source")
    p_conv.add_argument("destination", type=Path, help="Output file path (.parquet, .csv, .jsonl, .feather, .xlsx, .sqlite)")
    p_conv.add_argument("--table", help="Output table name when writing SQLite")

    args = parser.parse_args(argv)
    if args.command == "inspect":
        result = inspect_inventory(args.source, count_rows=not args.no_count)
        print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))
        return 0
    if args.command == "convert":
        info = write_table(args.source, args.destination, table=args.table)
        print(json.dumps(json_safe(info), ensure_ascii=False, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
