"""CLI for quick single-table out-of-core profiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import profile_table
from .spec import ProfilerConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Out-of-core single-pass profiler for any tabular source.")
    parser.add_argument("source", help="Input file, directory, glob, or db URI")
    parser.add_argument("--name", default="table")
    parser.add_argument("--entity-key", default=None)
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--targets", nargs="*", default=())
    parser.add_argument("--duplicate-mode", choices=("exact", "sample", "off"), default="sample")
    parser.add_argument("--out", type=Path, default=None, help="Directory to save CSV tables + metadata.json")
    args = parser.parse_args(argv)

    cfg = ProfilerConfig(duplicate_mode=args.duplicate_mode)
    result = profile_table(
        args.source,
        name=args.name,
        entity_key=args.entity_key,
        time_column=args.time_column,
        target_columns=tuple(args.targets),
        config=cfg,
    )
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        for sec_name, sec in result.sections.items():
            folder = args.out / sec_name
            folder.mkdir(parents=True, exist_ok=True)
            for tbl_name, df in sec.tables.items():
                df.to_csv(folder / f"{tbl_name}.csv", index=False, encoding="utf-8-sig")
        (args.out / "metadata.json").write_text(
            json.dumps(result.metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"Profile tables written to: {args.out.resolve()}")
    else:
        print(json.dumps(result.metadata, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
