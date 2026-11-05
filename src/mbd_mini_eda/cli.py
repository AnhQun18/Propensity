"""CLI xem kế hoạch hoặc tính và xuất báo cáo EDA."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

from .config import MBDMiniConfig, load_config
from .pipeline import build_plan, run_analysis
from .reporting import render_plan, export_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Khung thống kê mô tả và EDA cho MBD-mini")
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan", help="Xem kế hoạch; không đọc dữ liệu gốc")
    plan_parser.add_argument("--config", type=Path, help="JSON cấu hình, có thể bỏ để dùng mặc định")
    plan_parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    run_parser = commands.add_parser("run", help="Phân tích và xuất bảng, metadata, báo cáo")
    run_parser.add_argument("--config", type=Path)
    run_parser.add_argument("--output-dir", type=Path)
    run_parser.add_argument("--plots", action="store_true", help="Xuất PNG (cần extra plots)")
    run_parser.add_argument("--label-alignment", choices=("unresolved", "calendar_month", "explicit"))
    run_parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config) if args.config is not None else MBDMiniConfig()
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    if args.command == "plan":
        plan = build_plan(config)
        print(json.dumps(plan, ensure_ascii=False, indent=2) if args.format == "json" else render_plan(plan))
        return 0
    try:
        overrides = {}
        if args.output_dir is not None:
            overrides["output_dir"] = args.output_dir.resolve()
        if args.label_alignment is not None:
            overrides["label_alignment"] = args.label_alignment
        config = replace(config, **overrides)
        progress = None if args.quiet else lambda key: print(key, file=sys.stderr, flush=True)
        analysis = run_analysis(config, progress=progress)
        destination = export_report(analysis, plots=args.plots)
    except (OSError, TypeError, ValueError, ImportError) as error:
        parser.error(str(error))
    print(f"EDA report: {destination / 'report.md'}")
    return 0
