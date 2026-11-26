"""Colab Stage 1: Run 5 Top-50 H=3 Baselines + 6 Project Warning Verifications on GPU."""
from __future__ import annotations

import argparse
import gc
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from baseline_v2.metrics import (
    compute_slice_point_metrics,
    deterministic_tie_hash,
    holm_bonferroni,
    paired_bootstrap_macro_monthly,
)
from h3_pipeline.models_gpu import USE_GPU
from h3_pipeline.protocol_h3 import (
    assemble_h3_stage1_panel,
    ensure_stage1_delta_extracted,
    load_h3_spec,
    quick_point_summary,
    run_h3_rolling_experiment,
    summarize_h3_predictions,
)


def _sync_to_drive(local_dir: Path, drive_dir: Path | None) -> None:
    if drive_dir is None:
        return
    drive_dir.mkdir(parents=True, exist_ok=True)
    for p in local_dir.glob("*"):
        if p.is_file():
            shutil.copy2(p, drive_dir / p.name)


def run_stage1_baselines(
    spec: dict[str, Any],
    panel: pd.DataFrame,
    out_dir: Path,
    drive_dir: Path | None,
    seeds: list[int],
) -> tuple[dict[str, list[pd.DataFrame]], dict[str, list[pd.DataFrame]]]:
    """Run all 5 Top-50 Baselines across Val (05..08) and Locked OOT Test (2022-11)."""
    top50 = list(spec["top50_features"])
    assert len(top50) == 50, f"Expected 50 features, got {len(top50)}"
    missing = [c for c in top50 if c not in panel.columns]
    assert not missing, f"Missing Top-50 cols: {missing}"

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    n_boot = int(spec["bootstrap_replicates"])
    boot_seed = int(spec["bootstrap_seed"])
    tie_seed = int(spec["tie_break_seed"])

    baseline_defs = [
        ("B1_rule_prior_rfm", "rule_prior_rfm", False, [int(spec["canonical_seed"])]),
        ("B2_logreg_top50", "logreg", True, [int(spec["canonical_seed"])]),
        ("B3_lgbm_top50", "lgbm", True, seeds),
        ("B4_xgb_gpu_top50", "xgb_gpu", True, seeds),
        ("B5_catboost_gpu_top50", "catboost_gpu", True, seeds),
    ]

    val_months = tuple(str(x["eval_month"]) for x in spec["val_schedule"])
    oot_months = tuple(str(x["eval_month"]) for x in spec["oot_test_schedule"])

    val_preds: dict[str, list[pd.DataFrame]] = {}
    oot_preds: dict[str, list[pd.DataFrame]] = {}
    timing_rows: list[dict[str, Any]] = []
    val_monthly_rows, val_macro_rows = [], []
    oot_rows = []

    for exp_id, fam, is_prob, exp_seeds in baseline_defs:
        print(f"\n=== [Stage 1 Baseline] {exp_id} (family={fam}, seeds={exp_seeds}, USE_GPU={USE_GPU}) ===", flush=True)
        v_list, o_list = [], []
        t_exp = time.perf_counter()
        for s in exp_seeds:
            v_df, v_info = run_h3_rolling_experiment(
                spec, panel, "val_schedule", exp_id, fam, top50, seed=s, pred_cache_dir=pred_dir
            )
            o_df, o_info = run_h3_rolling_experiment(
                spec, panel, "oot_test_schedule", exp_id, fam, top50, seed=s, pred_cache_dir=pred_dir
            )
            v_list.append(v_df)
            o_list.append(o_df)
            timing_rows.append({
                "stage": "stage1_baseline",
                "exp_id": exp_id,
                "family": fam,
                "seed": s,
                "n_features": 50,
                "val_sec": v_info["elapsed_sec"],
                "oot_sec": o_info["elapsed_sec"],
                "total_sec": round(v_info["elapsed_sec"] + o_info["elapsed_sec"], 2),
                "use_gpu": USE_GPU,
            })

        val_preds[exp_id] = v_list
        oot_preds[exp_id] = o_list

        pm_v, mac_v = summarize_h3_predictions(v_list, val_months, is_prob, n_boot=n_boot, boot_seed=boot_seed, tie_seed=tie_seed)
        for row in pm_v:
            val_monthly_rows.append({"exp_id": exp_id, "family": fam, "n_features": 50, **row})
        val_macro_rows.append({"exp_id": exp_id, "family": fam, "n_features": 50, "n_seeds": len(exp_seeds), **mac_v})

        pm_o, mac_o = summarize_h3_predictions(o_list, oot_months, is_prob, n_boot=n_boot, boot_seed=boot_seed, tie_seed=tie_seed)
        oot_rows.append({"exp_id": exp_id, "family": fam, "n_features": 50, "n_seeds": len(exp_seeds), **mac_o})

        print(
            f"  -> {exp_id} DONE in {time.perf_counter()-t_exp:.1f}s | "
            f"Val Macro PR-AUC={mac_v['average_precision_mean']:.5f} Lift@1%={mac_v['lift_1pct_mean']:.2f}x | "
            f"OOT(2022-11) PR-AUC={mac_o['average_precision_mean']:.5f} Lift@1%={mac_o['lift_1pct_mean']:.2f}x",
            flush=True,
        )

        # Checkpoint after each baseline
        pd.DataFrame(timing_rows).to_csv(out_dir / "stage1_model_timings.csv", index=False)
        pd.DataFrame(val_monthly_rows).to_csv(out_dir / "stage1_baselines_val_monthly.csv", index=False)
        pd.DataFrame(val_macro_rows).to_csv(out_dir / "stage1_baselines_val_macro.csv", index=False)
        pd.DataFrame(oot_rows).to_csv(out_dir / "stage1_baselines_oot_test_locked.csv", index=False)
        _sync_to_drive(out_dir, drive_dir)

    # Paired bootstrap on Val and Locked OOT Test across baselines vs B2_logreg_top50 & B1_rule_prior_rfm
    pair_rows = []
    for st_name, p_map, m_tuple in (("val", val_preds, val_months), ("oot_test", oot_preds, oot_months)):
        for a_id, b_id, desc in (
            ("B1_rule_prior_rfm", "B2_logreg_top50", "LogReg Top50 vs Rule Prior+RFM"),
            ("B2_logreg_top50", "B3_lgbm_top50", "LightGBM Top50 vs LogReg Top50"),
            ("B2_logreg_top50", "B4_xgb_gpu_top50", "XGBoost GPU Top50 vs LogReg Top50"),
            ("B2_logreg_top50", "B5_catboost_gpu_top50", "CatBoost GPU Top50 vs LogReg Top50"),
            ("B4_xgb_gpu_top50", "B5_catboost_gpu_top50", "CatBoost GPU vs XGBoost GPU (Top50)"),
        ):
            is_pa = (a_id != "B1_rule_prior_rfm")
            res = paired_bootstrap_macro_monthly(
                p_map[a_id], p_map[b_id], m_tuple, is_pa, True, n_replicates=n_boot, boot_seed=boot_seed, tie_seed=tie_seed
            )
            pair_rows.append({"stage": st_name, "model_a": a_id, "model_b": b_id, "desc": desc, **res})
    pair_df = pd.DataFrame(pair_rows)
    for st_name in ("val", "oot_test"):
        mk = pair_df["stage"] == st_name
        pair_df.loc[mk, "delta_pr_p_holm"] = holm_bonferroni(pair_df.loc[mk, "delta_average_precision_p_value"].tolist())
        pair_df.loc[mk, "delta_l1_p_holm"] = holm_bonferroni(pair_df.loc[mk, "delta_lift_1pct_p_value"].tolist())
    pair_df.to_csv(out_dir / "stage1_baselines_paired_bootstrap.csv", index=False)
    _sync_to_drive(out_dir, drive_dir)

    return val_preds, oot_preds


def run_warning_verifications(
    spec: dict[str, Any],
    panel: pd.DataFrame,
    aux_path: Path,
    out_dir: Path,
    drive_dir: Path | None,
) -> None:
    """Verify all 6 project warnings strictly under H=3 (Val 05..08, seed=42)."""
    print("\n=== [Stage 1 Verification] Auditing 6 Project Warnings under H=3 ===", flush=True)
    seed = int(spec["canonical_seed"])
    top50 = list(spec["top50_features"])
    pred_dir = out_dir / "predictions_verify"
    pred_dir.mkdir(parents=True, exist_ok=True)

    aux = pd.read_parquet(aux_path)
    aux["month"] = aux["month"].astype(str)
    panel_aug = panel.merge(aux.drop(columns=["fold"], errors="ignore"), on=["client_id", "month"], how="left", validate="one_to_one")
    del aux
    gc.collect()

    # Ensure cap365 tenure col
    panel_aug["trx_all_days_since_first_event_cap365"] = np.minimum(
        panel_aug["trx_all_days_since_first_event"].to_numpy(dtype=np.float32),
        np.float32(365.0),
    )

    rows: list[dict[str, Any]] = []

    def _run_check(check_group: str, variant: str, fcols: list[str], sched_key: str = "val_schedule", shuffle: bool = False) -> pd.DataFrame:
        t0 = time.perf_counter()
        pdf, info = run_h3_rolling_experiment(
            spec,
            panel_aug,
            sched_key,
            f"V_{check_group}_{variant}",
            "catboost_gpu",
            fcols,
            seed=seed,
            pred_cache_dir=pred_dir,
            shuffle_labels=shuffle,
        )
        qm = quick_point_summary(pdf, tie_seed=seed, is_prob=True)
        rows.append({
            "check_group": check_group,
            "variant": variant,
            "schedule": sched_key,
            "n_features": len(fcols),
            "elapsed_sec": info["elapsed_sec"] or round(time.perf_counter() - t0, 2),
            **qm,
        })
        print(f"  [{check_group:18s} | {variant:26s}] PR-AUC={qm['macro_pr_auc']:.5f} | Lift@1%={qm['macro_lift_1pct']:.2f}x | AUC={qm['macro_roc_auc']:.4f}", flush=True)
        return pdf

    # Reference baseline on Val
    base_val_pdf = _run_check("0_reference", "top50_full_delay0", top50)

    # 1. Dialog near-cutoff lag audit (0d vs 7d, 14d, 30d, no_dialog)
    dlg_9 = [c for c in top50 if c.startswith("dialog_")]
    base_no_dlg = [c for c in top50 if not c.startswith("dialog_")]
    _run_check("1_dialog_lag_audit", "no_dialog", base_no_dlg)
    for lag in (7, 14, 30):
        lag_cols = [
            f"dlg_lag{lag}_30d_has", f"dlg_lag{lag}_30d_cnt", f"dlg_lag{lag}_30d_rec",
            f"dlg_lag{lag}_90d_has", f"dlg_lag{lag}_90d_cnt", f"dlg_lag{lag}_90d_rec",
            f"dlg_lag{lag}_all_cnt", f"dlg_lag{lag}_all_rec",
        ]
        _run_check("1_dialog_lag_audit", f"dialog_lag_{lag}d", base_no_dlg + lag_cols)

    # 2. Target-lag (prior_*) delay & label-overlap audit
    prior_cols = [c for c in top50 if c.startswith("prior_")]
    no_prior_cols = [c for c in top50 if not c.startswith("prior_")]
    delay1m_cols = [
        "prior_delay1m_months_observed",
        "prior_delay1m_prev_bought",
        "prior_delay1m_3m_buy_months",
        "prior_delay1m_all_buy_months",
        "prior_delay1m_months_since_last_buy",
    ]
    _run_check("2_prior_delay", "prior_delay_1m", no_prior_cols + delay1m_cols)
    _run_check("2_prior_delay", "no_prior_pure_behav", no_prior_cols)

    # Overlap audit on eval_month=2022-08: rolling [02..05] vs non-overlapping step-3 [02, 05]
    spec_overlap = {
        **spec,
        "val_schedule": [{"eval_month": "2022-08", "train_months": ["2022-02", "2022-03", "2022-04", "2022-05"]}],
        "oot_test_schedule": [{"eval_month": "2022-08", "train_months": ["2022-02", "2022-05"]}],
    }
    for label, sk in (("rolling_02_to_05_eval08", "val_schedule"), ("non_overlap_02_05_eval08", "oot_test_schedule")):
        pdf_ov, info_ov = run_h3_rolling_experiment(spec_overlap, panel_aug, sk, f"V_overlap_{label}", "catboost_gpu", top50, seed=seed, pred_cache_dir=pred_dir)
        qm_ov = quick_point_summary(pdf_ov, tie_seed=seed, is_prob=True)
        rows.append({"check_group": "2b_label_overlap", "variant": label, "schedule": sk, "n_features": 50, "elapsed_sec": info_ov["elapsed_sec"], **qm_ov})
        print(f"  [{'2b_label_overlap':18s} | {label:26s}] PR-AUC={qm_ov['macro_pr_auc']:.5f} | Lift@1%={qm_ov['macro_lift_1pct']:.2f}x", flush=True)

    # 3. Timezone audit (MSK UTC+3 vs UTC 00:00 on TRX RFM cols)
    msk_trx_rfm = [
        "trx_30d_event_count", "trx_30d_log1p_event_count", "trx_30d_days_since_last_event",
        "trx_30d_log1p_main_amount_abs_sum", "trx_90d_log1p_event_count", "trx_90d_days_since_last_event",
        "trx_all_event_count", "trx_all_days_since_last_event", "trx_all_days_since_first_event",
    ]
    utc_trx_rfm = [f"utc_{c}" for c in msk_trx_rfm]
    _run_check("3_timezone_audit", "msk_trx_rfm_only", msk_trx_rfm)
    _run_check("3_timezone_audit", "utc_trx_rfm_only", utc_trx_rfm)

    # 4. Tenure left-censoring audit
    no_ten = [c for c in top50 if c != "trx_all_days_since_first_event"]
    cap_ten = no_ten + ["trx_all_days_since_first_event_cap365"]
    _run_check("4_tenure_censor", "drop_trx_tenure", no_ten)
    _run_check("4_tenure_censor", "cap365_trx_tenure", cap_ten)

    # 5. Geohash drift + 0-TRX active-dialog slice stability on Val
    no_gh = [c for c in top50 if "geohash" not in c]
    gh_pdf = _run_check("5_geohash_drift", "drop_geohash_cols", no_gh)
    gh_mo_rows = []
    for label, pdf in (("with_geohash", base_val_pdf), ("no_geohash", gh_pdf)):
        for mo, g in pdf.groupby("month"):
            th = deterministic_tie_hash(g["client_id"].to_numpy(), seed=seed)
            mm = compute_slice_point_metrics(g["y_true"].to_numpy(), g["score"].to_numpy(), g["client_id"].to_numpy(), seed=seed, tie_hashes=th)
            gh_mo_rows.append({"variant": label, "month": mo, "pr_auc": mm["average_precision"], "lift_1pct": mm["lift_1pct"], "roc_auc": mm["roc_auc"]})
    pd.DataFrame(gh_mo_rows).to_csv(out_dir / "verify_geohash_monthly_val.csv", index=False)

    # Slice stability on Val (cold_start, 0-TRX + active dialog, prior_buyer vs new_buyer)
    meta_val = panel_aug[panel_aug["month"].isin(["2022-05", "2022-06", "2022-07", "2022-08"])][
        ["client_id", "month", "cold_start", "trx_90d_has_event", "trx_all_has_event", "dialog_30d_has_event", "prior_all_bought_any"]
    ]
    mrg = base_val_pdf.merge(meta_val, on=["client_id", "month"], how="left")
    sl_defs = [
        ("all_val", np.ones(len(mrg), dtype=bool)),
        ("cold_start_1", mrg["cold_start"].to_numpy() == 1),
        ("trx_all_zero", mrg["trx_all_has_event"].to_numpy() == 0),
        ("trx90_zero_dlg30_act", (mrg["trx_90d_has_event"].to_numpy() == 0) & (mrg["dialog_30d_has_event"].to_numpy() == 1)),
        ("prior_buyer_1", mrg["prior_all_bought_any"].fillna(0).to_numpy() == 1),
        ("prior_buyer_0", mrg["prior_all_bought_any"].fillna(0).to_numpy() == 0),
    ]
    sl_rows = []
    for sname, mk in sl_defs:
        sub = mrg.loc[mk]
        if len(sub) < 50 or sub["y_true"].sum() == 0:
            continue
        sm = compute_slice_point_metrics(sub["y_true"].to_numpy(), sub["score"].to_numpy(), sub["client_id"].to_numpy(), seed=seed)
        sl_rows.append({"slice": sname, "n": sm["n"], "positives": sm["positives"], "prevalence": sm["prevalence"], "pr_auc": sm["average_precision"], "lift_1pct": sm["lift_1pct"], "roc_auc": sm["roc_auc"]})
    pd.DataFrame(sl_rows).to_csv(out_dir / "verify_val_slices.csv", index=False)

    # 6. Shuffled-label sanity check (Val & OOT Test)
    shuf_val = _run_check("6_shuffled_sanity", "shuffled_y_train_val", top50, sched_key="val_schedule", shuffle=True)
    shuf_oot = _run_check("6_shuffled_sanity", "shuffled_y_train_oot", top50, sched_key="oot_test_schedule", shuffle=True)
    assert rows[-2]["macro_lift_1pct"] < 1.8, f"LEAKAGE WARNING: Shuffled Val Lift@1%={rows[-2]['macro_lift_1pct']:.2f}x"
    assert rows[-1]["macro_lift_1pct"] < 1.8, f"LEAKAGE WARNING: Shuffled OOT Lift@1%={rows[-1]['macro_lift_1pct']:.2f}x"

    pd.DataFrame(rows).to_csv(out_dir / "stage1_warning_verifications.csv", index=False)
    _sync_to_drive(out_dir, drive_dir)
    del panel_aug
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--delta-zip", type=Path, default=None)
    parser.add_argument("--drive-save-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 142, 242])
    args = parser.parse_args()

    t_all = time.perf_counter()
    work_dir = args.work_dir.resolve()
    spec = load_h3_spec(work_dir / "configs" / "h3_propensity_spec.json")
    out_dir = work_dir / "outputs" / "h3_propensity" / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)

    delta_zip = args.delta_zip or (work_dir / "outputs" / "h3_propensity" / "h3_stage1_delta.zip")
    delta_dir = ensure_stage1_delta_extracted(delta_zip, work_dir)
    print(f"[Stage 1] USE_GPU={USE_GPU} | work_dir={work_dir} | delta_dir={delta_dir}", flush=True)

    panel = assemble_h3_stage1_panel(work_dir)
    print(f"[Stage 1] Unified H=3 panel assembled: shape={panel.shape} | months={sorted(panel['month'].unique())}", flush=True)

    run_stage1_baselines(spec, panel, out_dir, args.drive_save_dir, seeds=[int(s) for s in args.seeds])
    run_warning_verifications(spec, panel, delta_dir / "h3_verify_aux.parquet", out_dir, args.drive_save_dir)

    print(f"\n[Stage 1 ALL DONE] Total elapsed={time.perf_counter()-t_all:.1f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
