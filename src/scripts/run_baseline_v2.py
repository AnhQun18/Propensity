"""CLI entrypoint for Baseline v2 (Protocol B Monthly Rolling OOC + Protocol A Reference)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from baseline_v2.runner import run_baseline_v2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Baseline v2 (Protocol B Monthly Rolling OOC + Protocol A Reference).")
    p.add_argument("--phase", choices=["val", "test", "all"], default="all", help="Run Val-only tuning/ablations, Locked Test, or all.")
    p.add_argument("--panel-path", type=Path, default=None, help="Path to panel_moscow.parquet.")
    p.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: outputs/baseline_v2).")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Seeds override (default: 42 142 242).")
    p.add_argument("--no-protocol-a", action="store_true", help="Skip Protocol A flat reference runs.")
    p.add_argument("--threads", type=int, default=-1, help="CatBoost CPU threads (-1 = all).")
    p.add_argument("--n-bootstrap", type=int, default=None, help="Bootstrap replicates override (default: 1000).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    res = run_baseline_v2(
        phase=args.phase,
        panel_path=args.panel_path,
        out_dir=args.output_dir,
        seeds_override=args.seeds,
        include_protocol_a=not args.no_protocol_a,
        thread_count=args.threads,
        n_bootstrap_override=args.n_bootstrap,
    )
    print(f"\n[Baseline v2] DONE in {res['elapsed_seconds']}s -> phase={res['phase_last_run']}", flush=True)


if __name__ == "__main__":
    main()
