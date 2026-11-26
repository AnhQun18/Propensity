"""End-to-end Baseline v2 orchestrator supporting Val-only tuning/ablations and one-shot Test evaluation."""
from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .diagnostics import (
    run_val_ablations,
    run_val_catboost_tuning,
    run_val_permutation_importance,
    run_val_shuffled_sanity,
    run_val_slices,
)
from .metrics import (
    bootstrap_multiseed_macro_monthly,
    bootstrap_multiseed_per_month,
    bootstrap_multiseed_pooled,
    holm_bonferroni,
    paired_bootstrap_macro_monthly,
)
from .protocol import META_KEEP_COLS, run_protocol_experiment
from .spec import load_spec, resolve_raw_panel_columns, verify_or_create_freeze_lock

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_panel_for_v2(panel_path: Path, spec: dict[str, Any]) -> pd.DataFrame:
    """Load minimal required columns from panel_moscow.parquet for Baseline v2."""
    all_feat_cols: list[str] = []
    for cols in spec["feature_sets"].values():
        all_feat_cols.extend(resolve_raw_panel_columns(cols))
    needed_cols = list(dict.fromkeys(list(META_KEEP_COLS) + [spec["data_contract"]["target_col"]] + all_feat_cols))
    df = pd.read_parquet(panel_path, columns=needed_cols)
    df["month"] = df["month"].astype(str)
    return df


def _run_milestone_suite(
    spec: dict[str, Any],
    protocol_name: str,
    stage_name: str,
    panel_df: pd.DataFrame,
    tuned_cb_params: dict[str, Any],
    seeds: list[int],
    out_dir: Path,
    thread_count: int = -1,
) -> dict[str, list[pd.DataFrame]]:
    """Run all 6 Baseline v2 milestones across specified seeds for a (protocol, stage)."""
    feat_sets = spec["feature_sets"]
    pred_dir = out_dir / "predictions"
    preds_by_model: dict[str, list[pd.DataFrame]] = {}
    cb_params = {**tuned_cb_params, "thread_count": int(thread_count)}

    for mspec in spec["models"]:
        eid = mspec["exp_id"]
        fam = mspec["family"]
        fcols = feat_sets[mspec["feature_set"]]
        uses_lag = bool(mspec["uses_target_lag"])
        model_seeds = seeds if mspec["stochastic_across_seeds"] else [int(spec["evaluation"]["canonical_seed"])]
        is_cb = (fam == "catboost_tuned")

        s_dfs: list[pd.DataFrame] = []
        for s in model_seeds:
            pdf, info = run_protocol_experiment(
                spec=spec,
                protocol_name=protocol_name,
                stage_name=stage_name,
                panel_df=panel_df,
                exp_id=eid,
                family=fam,
                feature_cols=fcols,
                uses_target_lag=uses_lag,
                seed=s,
                model_params=cb_params if is_cb else None,
                pred_cache_dir=pred_dir,
                use_temporal_es=is_cb,
            )
            s_dfs.append(pdf)
            status = "cached" if info.get("cached") else f"{info.get('elapsed_sec', 0.0):.1f}s"
            print(f"  [{protocol_name} | {stage_name.upper():4s}] {eid:22s} seed={s:3d} ({status})", flush=True)
        preds_by_model[eid] = s_dfs
    return preds_by_model


def _summarize_stage(
    spec: dict[str, Any],
    protocol_name: str,
    stage_name: str,
    preds_by_model: dict[str, list[pd.DataFrame]],
    n_boot: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    months = tuple(spec["data_contract"]["val_months"] if stage_name == "val" else spec["data_contract"]["test_months"])
    boot_seed = int(spec["evaluation"]["bootstrap_seed"])
    tie_seed = int(spec["evaluation"]["tie_break_seed"])

    monthly_rows, macro_rows, pooled_rows = [], [], []
    for mspec in spec["models"]:
        eid = mspec["exp_id"]
        is_prob = bool(mspec["is_probabilistic"])
        n_feats = len(spec["feature_sets"][mspec["feature_set"]])
        s_dfs = preds_by_model[eid]

        meta = {
            "protocol": protocol_name,
            "stage": stage_name,
            "exp_id": eid,
            "role": mspec["role"],
            "family": mspec["family"],
            "n_features": n_feats,
            "uses_target_lag": bool(mspec["uses_target_lag"]),
            "is_reference_only": bool(mspec["is_reference_only"]),
        }
        for m in months:
            mm = bootstrap_multiseed_per_month(s_dfs, month=m, is_prob=is_prob, n_replicates=n_boot, boot_seed=boot_seed, tie_seed=tie_seed)
            monthly_rows.append({**meta, **mm})

        macro = bootstrap_multiseed_macro_monthly(s_dfs, months=months, is_prob=is_prob, n_replicates=n_boot, boot_seed=boot_seed, tie_seed=tie_seed)
        macro_rows.append({**meta, **macro})

        pooled = bootstrap_multiseed_pooled(s_dfs, is_prob=is_prob, n_replicates=n_boot, boot_seed=boot_seed, tie_seed=tie_seed)
        pooled_rows.append({**meta, **pooled})

    return monthly_rows, macro_rows, pooled_rows


def generate_v2_figures(out_dir: Path) -> list[str]:
    """Generate summary plots in outputs/baseline_v2/figures/."""
    m_dir = out_dir / "metrics"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    monthly_path = m_dir / "monthly_metrics_all.csv"
    macro_path = m_dir / "macro_monthly_summary.csv"
    if not monthly_path.exists() or not macro_path.exists():
        return saved

    mdf = pd.read_csv(monthly_path)
    macro_df = pd.read_csv(macro_path)

    # 1. Monthly Lift@1% and PR-AUC under Protocol B (with 95% CI)
    sub_b = mdf[mdf["protocol"] == "B_rolling_ooc"]
    if len(sub_b) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
        months = sorted(sub_b["month"].unique())
        x = np.arange(len(months))
        for eid in sub_b["exp_id"].unique():
            ed = sub_b[sub_b["exp_id"] == eid].set_index("month").reindex(months)
            ls = "--" if eid == "L1_rule_target_lag" else "-"
            axes[0].plot(x, ed["lift_1pct_mean"], marker="o", linestyle=ls, label=eid)
            axes[1].plot(x, ed["average_precision_mean"], marker="o", linestyle=ls, label=eid)
        for ax, title, ylabel in (
            (axes[0], "Protocol B (Rolling OOC): Monthly Lift@1% (K ~ 1,003/mo)", "Lift@1%"),
            (axes[1], "Protocol B (Rolling OOC): Monthly PR-AUC", "PR-AUC (Average Precision)"),
        ):
            ax.set_xticks(x)
            ax.set_xticklabels(months, rotation=20)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[1].legend(fontsize=8, loc="best")
        fig.tight_layout()
        p1 = fig_dir / "01_protocol_b_monthly_trends.png"
        fig.savefig(p1, dpi=140, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p1))

    # 2. Protocol B vs Protocol A on Test Macro-Monthly
    test_macro = macro_df[macro_df["stage"] == "test"]
    if len(test_macro) > 0 and test_macro["protocol"].nunique() >= 1:
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        exps = [e for e in test_macro["exp_id"].unique()]
        x = np.arange(len(exps))
        w = 0.36
        for idx, proto in enumerate(sorted(test_macro["protocol"].unique())):
            sub_p = test_macro[test_macro["protocol"] == proto].set_index("exp_id").reindex(exps)
            vals = sub_p["lift_1pct_mean"].to_numpy(dtype=float)
            lo = np.maximum(0.0, vals - sub_p["lift_1pct_ci_low"].to_numpy(dtype=float))
            hi = np.maximum(0.0, sub_p["lift_1pct_ci_high"].to_numpy(dtype=float) - vals)
            offset = -w / 2 if idx == 0 else w / 2
            ax.bar(x + offset, vals, width=w, yerr=[lo, hi], capsize=3, label=proto, alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels(exps, rotation=20, ha="right")
        ax.set_ylabel("Macro-Monthly Lift@1% (95% CI)")
        ax.set_title("Locked Test: Protocol B (Monthly Rolling) vs Protocol A (Flat)", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend()
        fig.tight_layout()
        p2 = fig_dir / "02_test_protocol_b_vs_a_lift.png"
        fig.savefig(p2, dpi=140, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p2))

    return saved


def run_baseline_v2(
    phase: str = "all",
    panel_path: Path | None = None,
    out_dir: Path | None = None,
    seeds_override: list[int] | None = None,
    include_protocol_a: bool = True,
    thread_count: int = -1,
    n_bootstrap_override: int | None = None,
    spec_json_path: Path | None = None,
    spec_md_path: Path | None = None,
) -> dict[str, Any]:
    """Execute Baseline v2 (`val`, `test`, or `all`) strictly honoring frozen spec & Test-once gate."""
    t_all = time.perf_counter()
    spec = load_spec(spec_json_path)
    p_path = Path(panel_path) if panel_path is not None else (REPO_ROOT / spec["data_contract"]["default_panel_parquet"])
    o_dir = Path(out_dir) if out_dir is not None else (REPO_ROOT / "outputs" / "baseline_v2")
    m_dir = o_dir / "metrics"
    m_dir.mkdir(parents=True, exist_ok=True)

    lock = verify_or_create_freeze_lock(o_dir, spec_json_path=spec_json_path, spec_md_path=spec_md_path)
    seeds = [int(s) for s in (seeds_override or spec["evaluation"]["seeds"])]
    n_boot = int(n_bootstrap_override or spec["evaluation"]["bootstrap_replicates"])
    boot_seed = int(spec["evaluation"]["bootstrap_seed"])
    tie_seed = int(spec["evaluation"]["tie_break_seed"])

    print(f"[Baseline v2] Spec SHA-256 locked: {lock['spec_json_sha256'][:12]}...", flush=True)
    print(f"[Baseline v2] Loading panel from {p_path} ...", flush=True)
    panel_df = load_panel_for_v2(p_path, spec)
    print(f"[Baseline v2] Panel loaded: {panel_df.shape}", flush=True)

    all_monthly: list[dict[str, Any]] = []
    all_macro: list[dict[str, Any]] = []
    all_pooled: list[dict[str, Any]] = []

    val_preds_b: dict[str, list[pd.DataFrame]] = {}
    test_preds_b: dict[str, list[pd.DataFrame]] = {}
    val_preds_a: dict[str, list[pd.DataFrame]] = {}
    test_preds_a: dict[str, list[pd.DataFrame]] = {}

    if phase in ("val", "all"):
        print("\n=== PHASE 1 (VAL ONLY): CATBOOST TUNING, 6 BASELINES, ABLATIONS & DIAGNOSTICS ===", flush=True)
        tuned_cb_params, grid_df = run_val_catboost_tuning(spec, panel_df, o_dir, thread_count=thread_count, n_boot=n_boot)
        print(
            f"[Val Tuning Winner] lr={tuned_cb_params['learning_rate']}, depth={tuned_cb_params['depth']}, "
            f"l2={tuned_cb_params['l2_leaf_reg']}, median_best_iter={tuned_cb_params['median_best_iter_val']} "
            f"| Val Macro Lift@1%={tuned_cb_params['val_macro_lift_1pct']:.2f}x",
            flush=True,
        )

        val_preds_b = _run_milestone_suite(spec, "B_rolling_ooc", "val", panel_df, tuned_cb_params, seeds, o_dir, thread_count)
        mr, mac, pol = _summarize_stage(spec, "B_rolling_ooc", "val", val_preds_b, n_boot)
        all_monthly.extend(mr)
        all_macro.extend(mac)
        all_pooled.extend(pol)

        if include_protocol_a:
            val_preds_a = _run_milestone_suite(spec, "A_flat_ooc", "val", panel_df, tuned_cb_params, seeds, o_dir, thread_count)
            mr_a, mac_a, pol_a = _summarize_stage(spec, "A_flat_ooc", "val", val_preds_a, n_boot)
            all_monthly.extend(mr_a)
            all_macro.extend(mac_a)
            all_pooled.extend(pol_a)

        val_seed42 = {eid: dfs[0] for eid, dfs in val_preds_b.items()}
        run_val_ablations(spec, panel_df, tuned_cb_params, val_seed42, o_dir, thread_count=thread_count, n_boot=n_boot)
        run_val_permutation_importance(spec, panel_df, tuned_cb_params, o_dir, thread_count=thread_count)
        run_val_slices(val_seed42, o_dir, tie_seed=tie_seed)
        run_val_shuffled_sanity(spec, panel_df, tuned_cb_params, o_dir, thread_count=thread_count)

    if phase in ("test", "all"):
        cfg_path = o_dir / "tuned_catboost_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"Cannot open Locked Test before Val tuning! Missing {cfg_path}. Run --phase val first."
            )
        tuned_cb_params = json.loads(cfg_path.read_text(encoding="utf-8"))
        print("\n=== PHASE 2 (LOCKED TEST — OPENED ONCE): PROTOCOL B & PROTOCOL A REFERENCE ===", flush=True)
        test_preds_b = _run_milestone_suite(spec, "B_rolling_ooc", "test", panel_df, tuned_cb_params, seeds, o_dir, thread_count)
        mr, mac, pol = _summarize_stage(spec, "B_rolling_ooc", "test", test_preds_b, n_boot)
        all_monthly.extend(mr)
        all_macro.extend(mac)
        all_pooled.extend(pol)

        if include_protocol_a:
            test_preds_a = _run_milestone_suite(spec, "A_flat_ooc", "test", panel_df, tuned_cb_params, seeds, o_dir, thread_count)
            mr_a, mac_a, pol_a = _summarize_stage(spec, "A_flat_ooc", "test", test_preds_a, n_boot)
            all_monthly.extend(mr_a)
            all_macro.extend(mac_a)
            all_pooled.extend(pol_a)

    # Merge with existing metrics if running val and test separately
    def _save_merged(rows: list[dict[str, Any]], fname: str, key_cols: list[str]) -> pd.DataFrame:
        p = m_dir / fname
        new_df = pd.DataFrame(rows)
        if p.exists() and len(new_df) > 0:
            old_df = pd.read_csv(p)
            comb = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(subset=key_cols, keep="last")
        elif len(new_df) > 0:
            comb = new_df
        elif p.exists():
            comb = pd.read_csv(p)
        else:
            comb = pd.DataFrame()
        if len(comb) > 0:
            comb.to_csv(p, index=False)
        return comb

    monthly_df = _save_merged(all_monthly, "monthly_metrics_all.csv", ["protocol", "stage", "month", "exp_id"])
    macro_df = _save_merged(all_macro, "macro_monthly_summary.csv", ["protocol", "stage", "exp_id"])
    pooled_df = _save_merged(all_pooled, "pooled_oof_summary.csv", ["protocol", "stage", "exp_id"])

    # Paired bootstrap comparisons (Model pairs on Protocol B + Protocol B vs A)
    paired_model_rows: list[dict[str, Any]] = []
    paired_proto_rows: list[dict[str, Any]] = []
    model_meta = {m["exp_id"]: m for m in spec["models"]}

    pair_defs = [
        ("L0_base_rate", "L2_rule_trx_rfm", "Luật RFM TRX vs Sàn ngẫu nhiên"),
        ("L2_rule_trx_rfm", "B_logreg_T", "LogReg TRX (8v) vs Luật RFM TRX"),
        ("B_logreg_T", "B_logreg_TGD", "Thêm GEO+DIALOG thô trên LogReg (8 -> 14v)"),
        ("B_logreg_TGD", "B_catboost_TGD_tuned", "CatBoost tune nhẹ vs LogReg (cùng 14v thô)"),
        ("L1_rule_target_lag", "B_logreg_TGD", "LogReg 14v (không nhãn) vs L1 (tham chiếu dùng nhãn)"),
        ("L1_rule_target_lag", "B_catboost_TGD_tuned", "CatBoost tuned 14v (không nhãn) vs L1 (tham chiếu dùng nhãn)"),
    ]

    for stage_name, preds_b, preds_a in (
        ("val", val_preds_b, val_preds_a),
        ("test", test_preds_b, test_preds_a),
    ):
        if not preds_b:
            continue
        months = tuple(spec["data_contract"]["val_months"] if stage_name == "val" else spec["data_contract"]["test_months"])
        stage_model_rows = []
        for a_id, b_id, label in pair_defs:
            pb = paired_bootstrap_macro_monthly(
                preds_b[a_id], preds_b[b_id], months,
                bool(model_meta[a_id]["is_probabilistic"]),
                bool(model_meta[b_id]["is_probabilistic"]),
                n_replicates=n_boot, boot_seed=boot_seed, tie_seed=tie_seed,
            )
            stage_model_rows.append({"stage": stage_name, "protocol": "B_rolling_ooc", "model_a": a_id, "model_b": b_id, "question": label, **pb})
        holm_ap = holm_bonferroni([r["delta_average_precision_p_value"] for r in stage_model_rows])
        holm_lift = holm_bonferroni([r["delta_lift_1pct_p_value"] for r in stage_model_rows])
        for r, ha, hl in zip(stage_model_rows, holm_ap, holm_lift):
            r["delta_average_precision_p_holm"] = ha
            r["delta_lift_1pct_p_holm"] = hl
        paired_model_rows.extend(stage_model_rows)

        if preds_a:
            for eid in preds_b:
                pba = paired_bootstrap_macro_monthly(
                    preds_a[eid], preds_b[eid], months,
                    bool(model_meta[eid]["is_probabilistic"]),
                    bool(model_meta[eid]["is_probabilistic"]),
                    n_replicates=n_boot, boot_seed=boot_seed, tie_seed=tie_seed,
                )
                paired_proto_rows.append({"stage": stage_name, "exp_id": eid, "comparison": "Protocol_B_minus_A", **pba})

    _save_merged(paired_model_rows, "paired_models_protocol_b.csv", ["stage", "protocol", "model_a", "model_b"])
    _save_merged(paired_proto_rows, "paired_protocol_b_vs_a.csv", ["stage", "exp_id"])

    figs = generate_v2_figures(o_dir)
    report_path = write_v2_markdown_report(o_dir, spec)
    manifest = {
        "status": "ready",
        "phase_last_run": phase,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - t_all, 2),
        "spec_id": spec["spec_id"],
        "spec_freeze_lock": lock,
        "seeds": seeds,
        "bootstrap_replicates": n_boot,
        "include_protocol_a": include_protocol_a,
        "figures": figs,
        "report_path": str(report_path),
    }
    (o_dir / "v2_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    del panel_df
    gc.collect()
    return manifest


def write_v2_markdown_report(out_dir: Path, spec: dict[str, Any]) -> Path:
    """Write a concise Markdown report summarizing Val diagnostics and Locked Test results."""
    m_dir = out_dir / "metrics"
    lines = [
        "# Baseline v2 Report — Protocol B (Monthly Rolling × 5-Fold OOC)",
        "",
        f"- **Spec ID:** `{spec['spec_id']}` (`v{spec['spec_version']}`)",
        "- **Timezone / Cutoff:** `Europe/Moscow (UTC+3)`, strict `event_time_msk < YYYY-MM-01 00:00:00 MSK`",
        "- **Primary Metrics:** `Lift@1%` (monthly budget `K = ceil(1% * N_month) ~ 1,003` calls/month) & `PR-AUC`, reported **per-month first** with 95% CI, then **Macro-Monthly** & **Pooled OOF**.",
        "- **L1 Caveat:** `L1_rule_target_lag` assumes `label_delay = 0` (`m-1` label available before month `m` cutoff); treat as reference upper bound until bank label SLA is verified.",
        "",
    ]
    cfg_path = out_dir / "tuned_catboost_config.json"
    if cfg_path.exists():
        cb_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        lines.extend([
            "## Tuned CatBoost Config (Selected Strictly on Val)",
            f"- `learning_rate={cb_cfg['learning_rate']}`, `depth={cb_cfg['depth']}`, `l2_leaf_reg={cb_cfg['l2_leaf_reg']}`, `median_best_iter_val={cb_cfg['median_best_iter_val']}`",
            "",
        ])

    for title, fname, filter_expr in (
        ("Locked Test — Per-Month Metrics (Protocol B)", "monthly_metrics_all.csv", ("B_rolling_ooc", "test")),
        ("Macro-Monthly Summary (Protocol B vs A)", "macro_monthly_summary.csv", None),
        ("Val-Only Ablations (Source T+G/T+D, Tenure, Linear+Tree Blend)", "val_ablation_macro.csv", None),
    ):
        fpath = m_dir / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        if filter_expr is not None:
            df = df[(df["protocol"] == filter_expr[0]) & (df["stage"] == filter_expr[1])]
        if len(df) == 0:
            continue
        lines.append(f"## {title}")
        cols = [c for c in ("protocol", "stage", "group", "month", "exp_id", "n_features", "lift_1pct_mean", "lift_1pct_ci_low", "lift_1pct_ci_high", "average_precision_mean", "average_precision_ci_low", "average_precision_ci_high", "roc_auc_mean") if c in df.columns]
        sub = df[cols].copy()
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for _, r in sub.iterrows():
            vals = []
            for c in cols:
                v = r[c]
                if pd.isna(v):
                    vals.append("—")
                elif isinstance(v, (float, np.floating)):
                    vals.append(f"{float(v):.4f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    rep_path = out_dir / "BASELINE_V2_REPORT.md"
    rep_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rep_path
