"""CLI entry point for propensity_model (plan / run / compare)."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from .bootstrap import paired_cluster_bootstrap
from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Dual-horizon (H=1, H=3) MBD-mini propensity baseline pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="Validate and print config without reading data")
    p_plan.add_argument("--config", required=True)

    p_run = sub.add_parser("run", help="Run full dual-horizon baseline validation & holdout tests")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--keep-enriched", action="store_true", help="Keep enriched_panel.parquet on disk")

    p_cmp = sub.add_parser("compare", help="Paired client-clustered bootstrap comparing two run dirs")
    p_cmp.add_argument("--run-a", required=True, help="First run directory (e.g. baseline_top50_v1)")
    p_cmp.add_argument("--run-b", required=True, help="Second run directory (e.g. baseline_full252_v1)")
    p_cmp.add_argument("--out", required=True, help="Output CSV path for cross-run comparison")
    p_cmp.add_argument("--replicates", type=int, default=1000)
    p_cmp.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):
        parser.error("propensity_model requires Python >= 3.11")

    if args.command == "plan":
        cfg = load_config(args.config)
        print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "run":
        from .runner import run_baseline

        cfg = load_config(args.config)
        print(run_baseline(cfg, keep_enriched_panel=args.keep_enriched))
        return

    if args.command == "compare":
        run_a = Path(args.run_a).resolve()
        run_b = Path(args.run_b).resolve()
        man_a = json.loads((run_a / "manifest.json").read_text(encoding="utf-8"))
        man_b = json.loads((run_b / "manifest.json").read_text(encoding="utf-8"))
        if man_a.get("status") != "ready" or man_b.get("status") != "ready":
            raise ValueError("Both runs must have status == 'ready' before comparison")

        rows = []
        # Compare validation winners (H=1 and H=3)
        for h in (1, 3):
            wa = man_a["winners"][f"h{h}"]
            wb = man_b["winners"][f"h{h}"]
            df_a = pd.concat(
                [pd.read_parquet(p) for p in sorted((run_a / "predictions" / f"h{h}_val" / wa).glob("fold_*.parquet"))],
                ignore_index=True,
            )
            df_b = pd.concat(
                [pd.read_parquet(p) for p in sorted((run_b / "predictions" / f"h{h}_val" / wb).glob("fold_*.parquet"))],
                ignore_index=True,
            )
            pb = paired_cluster_bootstrap(df_a, df_b, True, True, 0.01, args.replicates, args.seed)
            rows.append(
                {
                    "horizon": h,
                    "scope": "validation_pooled",
                    "run_a": man_a["run_name"],
                    "model_a": wa,
                    "run_b": man_b["run_name"],
                    "model_b": wb,
                    **pb,
                }
            )

        # Compare Final Holdout tests (H=1 2023-01 raw & cal_intercept; H=3 2022-11 monthly)
        final_pairs = [
            (1, "h1_final", "H1_final_raw_2023-01.parquet", "H1_final_raw"),
            (1, "h1_final", "H1_final_cal_intercept_2023-01.parquet", "H1_final_cal_intercept"),
            (3, "h3_final", "H3_final_monthly_2022-11.parquet", "H3_final_monthly"),
            (3, "h3_final", "H3_final_quarterly_2022-11.parquet", "H3_final_quarterly"),
        ]
        for h, subdir, fname, label in final_pairs:
            pa = run_a / "predictions" / subdir / fname
            pb_path = run_b / "predictions" / subdir / fname
            if pa.exists() and pb_path.exists():
                df_a = pd.read_parquet(pa)
                df_b = pd.read_parquet(pb_path)
                res = paired_cluster_bootstrap(df_a, df_b, True, True, 0.01, args.replicates, args.seed)
                rows.append(
                    {
                        "horizon": h,
                        "scope": "final_test",
                        "run_a": man_a["run_name"],
                        "model_a": label,
                        "run_b": man_b["run_name"],
                        "model_b": label,
                        **res,
                    }
                )

        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(out_path)
