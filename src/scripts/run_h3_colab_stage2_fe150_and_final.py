"""Colab Stage 2: Assemble EDA features, Val-only ~150 selection, GPU tuning, Final Ensemble & Locked OOT Test."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from baseline_v2.metrics import (
    compute_slice_point_metrics,
    deterministic_tie_hash,
    holm_bonferroni,
    paired_bootstrap_macro_monthly,
)
from h3_pipeline.models_gpu import USE_GPU, fit_predict_h3_model
from h3_pipeline.progress import ModelProgressTracker
from h3_pipeline.protocol_h3 import (
    META_COLS,
    assert_fold_isolation,
    assemble_h3_stage1_panel,
    ensure_stage1_delta_extracted,
    load_h3_spec,
    quick_point_summary,
    summarize_h3_predictions,
    validate_h3_schedule,
)
from h3_pipeline.selection_and_tuning import (
    EMB_COL_NAMES,
    assemble_matrix_fast,
    ensure_h3_pca_cache,
    select_features_val_only_150,
    tune_150_models_on_val,
)


def _sync_to_drive(local_dir: Path, drive_dir: Path | None) -> None:
    if drive_dir is None:
        return
    drive_dir.mkdir(parents=True, exist_ok=True)
    for p in local_dir.glob("*"):
        if p.is_file():
            shutil.copy2(p, drive_dir / p.name)


def run_pca_model_schedule(
    spec: dict[str, Any],
    panel: pd.DataFrame,
    schedule_key: str,
    exp_id: str,
    family: str,
    tab_cols: list[str],
    emb_idx: list[int],
    pca_dir: Path,
    seed: int,
    params_override: dict[str, Any] | None,
    pred_cache_dir: Path,
    shuffle_labels: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validate_h3_schedule(spec)
    steps = spec[schedule_key]
    tag = f"{schedule_key}_{exp_id}{'_shuf' if shuffle_labels else ''}_seed{seed}"
    cache_p = pred_cache_dir / f"{tag}.parquet"
    if cache_p.exists():
        return pd.read_parquet(cache_p), {"cached": True, "elapsed_sec": 0.0, "best_iterations": []}

    n_folds = int(spec["n_folds"])
    target_col = str(spec["target_col"])
    fold_col = str(spec["fold_col"])
    is_gbdt = family in ("lgbm", "xgb_gpu", "catboost_gpu")
    tracker = ModelProgressTracker(schedule_key, f"{exp_id}_s{seed}", len(steps) * n_folds, float(spec["max_seconds_per_model"]))

    step_dfs: list[pd.DataFrame] = []
    best_iters: list[int] = []
    feat_names = tab_cols + [EMB_COL_NAMES[i] for i in emb_idx]

    for st in steps:
        ev_m = str(st["eval_month"])
        tr_ms = [str(m) for m in st["train_months"]]
        df_tr = panel[panel["month"].isin(tr_ms)].reset_index(drop=True)
        df_ev = panel[panel["month"] == ev_m].reset_index(drop=True)
        tr_f = df_tr[fold_col].to_numpy()
        ev_f = df_ev[fold_col].to_numpy()

        for k in range(n_folds):
            sl_tr = df_tr.loc[tr_f != k]
            sl_ev = df_ev.loc[ev_f == k]
            assert_fold_isolation(sl_tr["client_id"].to_numpy(), sl_ev["client_id"].to_numpy(), f"{exp_id} {ev_m} f{k}")

            X_tr = assemble_matrix_fast(sl_tr, tab_cols, emb_idx, pca_dir / ev_m / f"f{k}_tr.npy" if emb_idx else None)
            X_ev = assemble_matrix_fast(sl_ev, tab_cols, emb_idx, pca_dir / ev_m / f"f{k}_ev.npy" if emb_idx else None)
            y_tr = sl_tr[target_col].to_numpy(dtype=np.int8, copy=True)
            y_ev = sl_ev[target_col].to_numpy(dtype=np.int8, copy=True)

            if shuffle_labels:
                rng = np.random.default_rng(9500 + seed * 100 + k)
                y_tr = rng.permutation(y_tr)

            es_mask = None
            if is_gbdt:
                es_fold = (k + 1) % n_folds
                if len(tr_ms) >= 2:
                    es_mask = (sl_tr[fold_col].to_numpy() == es_fold) & np.isin(sl_tr["month"].to_numpy(), tr_ms[-2:])
                else:
                    es_mask = sl_tr[fold_col].to_numpy() == es_fold

            sc, best_n = fit_predict_h3_model(
                family=family, X_tr=X_tr, y_tr=y_tr, X_ev=X_ev,
                feature_names=feat_names, seed=seed, es_mask=es_mask,
                params_override=params_override,
            )
            if best_n is not None:
                best_iters.append(int(best_n))

            step_dfs.append(pd.DataFrame({
                "client_id": sl_ev["client_id"].to_numpy(),
                "fold": sl_ev[fold_col].to_numpy(),
                "month": sl_ev["month"].to_numpy(),
                "y_true": y_ev,
                "score": sc.astype(np.float64),
            }))
            del X_tr, X_ev
            tracker.step_done(f"{ev_m} f{k}", extra=f"best_n={best_n}" if best_n is not None else "")

    total_sec = tracker.finish()
    pdf = pd.concat(step_dfs, ignore_index=True).sort_values(["month", "client_id"]).reset_index(drop=True)
    tmp = cache_p.with_suffix(".tmp.parquet")
    pdf.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(cache_p)
    return pdf, {"cached": False, "elapsed_sec": round(total_sec, 2), "best_iterations": best_iters}


def blend_3_models(df_cb: pd.DataFrame, df_xgb: pd.DataFrame, df_lgb: pd.DataFrame, w_cb: float, w_xgb: float, w_lgb: float, mode: str = "prob") -> pd.DataFrame:
    out = df_cb[["client_id", "fold", "month", "y_true"]].copy()
    s1 = df_cb["score"].to_numpy(dtype=np.float64)
    s2 = df_xgb["score"].to_numpy(dtype=np.float64)
    s3 = df_lgb["score"].to_numpy(dtype=np.float64)
    if mode == "logit":
        p1 = logit(np.clip(s1, 1e-7, 1.0 - 1e-7))
        p2 = logit(np.clip(s2, 1e-7, 1.0 - 1e-7))
        p3 = logit(np.clip(s3, 1e-7, 1.0 - 1e-7))
        out["score"] = expit(w_cb * p1 + w_xgb * p2 + w_lgb * p3)
    else:
        out["score"] = w_cb * s1 + w_xgb * s2 + w_lgb * s3
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--stage1-delta-zip", type=Path, default=None)
    parser.add_argument("--stage2-delta-zip", type=Path, default=None)
    parser.add_argument("--drive-save-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 142, 242])
    args = parser.parse_args()

    t_all = time.perf_counter()
    work_dir = args.work_dir.resolve()
    spec = load_h3_spec(work_dir / "configs" / "h3_propensity_spec.json")
    out_dir = work_dir / "outputs" / "h3_propensity" / "stage2"
    pred_dir = out_dir / "predictions"
    pca_dir = out_dir / "pca_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    pca_dir.mkdir(parents=True, exist_ok=True)

    s1_zip = args.stage1_delta_zip or (work_dir / "outputs" / "h3_propensity" / "h3_stage1_delta.zip")
    delta_dir = ensure_stage1_delta_extracted(s1_zip, work_dir)
    s2_zip = args.stage2_delta_zip or (work_dir / "outputs" / "h3_propensity" / "h3_stage2_fe_delta.zip")
    if not (delta_dir / "h3_stage2_fe_delta.parquet").exists():
        with zipfile.ZipFile(s2_zip, "r") as zf:
            zf.extractall(delta_dir)

    panel_s1 = assemble_h3_stage1_panel(work_dir)
    s2_df = pd.read_parquet(delta_dir / "h3_stage2_fe_delta.parquet")
    s2_df["month"] = s2_df["month"].astype(str)
    s2_new_cols = [c for c in s2_df.columns if c not in panel_s1.columns]
    panel = panel_s1.merge(s2_df[["client_id", "month"] + s2_new_cols], on=["client_id", "month"], how="inner", validate="one_to_one")
    del panel_s1, s2_df
    gc.collect()
    print(f"[Stage 2] Full H3 Candidate Panel ready: shape={panel.shape}", flush=True)

    ensure_h3_pca_cache(
        spec,
        panel[["client_id", "fold", "month"]],
        delta_dir / "h3_early_02_03_emb.npz",
        work_dir / "outputs" / "baseline_moscow_v1" / "dialog_emb_moscow.npz",
        pca_dir,
    )

    cand_tab = [c for c in panel.columns if c not in set(META_COLS) and not c.startswith("prior_delay1m_")]
    locked_feat = select_features_val_only_150(spec, panel, cand_tab, pca_dir, out_dir, target_k=150)
    tab_cols = list(locked_feat["tab_cols"])
    emb_idx = list(locked_feat["emb_indices"])
    _sync_to_drive(out_dir, args.drive_save_dir)

    locked_hp = tune_150_models_on_val(spec, panel, tab_cols, emb_idx, pca_dir, out_dir)
    _sync_to_drive(out_dir, args.drive_save_dir)

    seeds = [int(s) for s in args.seeds]
    model_specs = [
        ("M1_logreg_150", "logreg", locked_hp["logreg_params"], [int(spec["canonical_seed"])]),
        ("M2_lgbm_150_tuned", "lgbm", locked_hp["lgbm_params"], [int(spec["canonical_seed"])]),
        ("M3_xgb_gpu_150_tuned", "xgb_gpu", locked_hp["xgb_params"], seeds),
        ("M4_catboost_gpu_150_tuned", "catboost_gpu", locked_hp["cb_params"], seeds),
    ]

    val_months = tuple(str(x["eval_month"]) for x in spec["val_schedule"])
    oot_months = tuple(str(x["eval_month"]) for x in spec["oot_test_schedule"])
    n_boot = int(spec["bootstrap_replicates"])
    boot_seed = int(spec["bootstrap_seed"])
    tie_seed = int(spec["tie_break_seed"])

    val_preds: dict[str, list[pd.DataFrame]] = {}
    oot_preds: dict[str, list[pd.DataFrame]] = {}
    timing_rows: list[dict[str, Any]] = []

    for exp_id, fam, p_ov, exp_seeds in model_specs:
        v_list, o_list = [], []
        for s in exp_seeds:
            v_df, v_info = run_pca_model_schedule(spec, panel, "val_schedule", exp_id, fam, tab_cols, emb_idx, pca_dir, s, p_ov, pred_dir)
            o_df, o_info = run_pca_model_schedule(spec, panel, "oot_test_schedule", exp_id, fam, tab_cols, emb_idx, pca_dir, s, p_ov, pred_dir)
            v_list.append(v_df)
            o_list.append(o_df)
            timing_rows.append({
                "stage": "stage2_150", "exp_id": exp_id, "family": fam, "seed": s,
                "n_features": locked_feat["n_total"], "val_sec": v_info["elapsed_sec"],
                "oot_sec": o_info["elapsed_sec"], "total_sec": round(v_info["elapsed_sec"] + o_info["elapsed_sec"], 2),
                "use_gpu": USE_GPU,
            })
        val_preds[exp_id] = v_list
        oot_preds[exp_id] = o_list

    # Tune Final Blend weights strictly on Val (seed=42) BEFORE evaluating OOT Blend
    best_blend = None
    best_b_score = (-1.0, -1.0)
    for mode in ("prob", "logit"):
        for w_cb in (0.3, 0.4, 0.5, 0.6):
            for w_xgb in (0.2, 0.3, 0.4, 0.5):
                w_lgb = round(1.0 - w_cb - w_xgb, 2)
                if w_lgb < 0.0 or w_lgb > 0.5:
                    continue
                bdf_v = blend_3_models(val_preds["M4_catboost_gpu_150_tuned"][0], val_preds["M3_xgb_gpu_150_tuned"][0], val_preds["M2_lgbm_150_tuned"][0], w_cb, w_xgb, w_lgb, mode)
                qm = quick_point_summary(bdf_v, tie_seed=tie_seed)
                sc = (qm["macro_pr_auc"], qm["macro_lift_1pct"])
                if sc > best_b_score:
                    best_b_score = sc
                    best_blend = {"mode": mode, "w_cb": w_cb, "w_xgb": w_xgb, "w_lgb": w_lgb}
    assert best_blend is not None
    (out_dir / "locked_ensemble_weights.json").write_text(json.dumps(best_blend, indent=2), encoding="utf-8")
    print(f"[Stage 2C] Val-Locked Blend: {best_blend}", flush=True)

    val_preds["M5_final_blend_150"] = [
        blend_3_models(
            val_preds["M4_catboost_gpu_150_tuned"][i],
            val_preds["M3_xgb_gpu_150_tuned"][i],
            val_preds["M2_lgbm_150_tuned"][min(i, len(val_preds["M2_lgbm_150_tuned"]) - 1)],
            **best_blend,
        )
        for i in range(len(seeds))
    ]
    oot_preds["M5_final_blend_150"] = [
        blend_3_models(
            oot_preds["M4_catboost_gpu_150_tuned"][i],
            oot_preds["M3_xgb_gpu_150_tuned"][i],
            oot_preds["M2_lgbm_150_tuned"][min(i, len(oot_preds["M2_lgbm_150_tuned"]) - 1)],
            **best_blend,
        )
        for i in range(len(seeds))
    ]

    # No-prior ablation on best single GPU model (CatBoost & XGBoost) to isolate EDA GEO+DIALOG gain
    tab_no_prior = [c for c in tab_cols if not c.startswith("prior_")]
    v_np, _ = run_pca_model_schedule(spec, panel, "val_schedule", "M6_xgb_150_no_prior", "xgb_gpu", tab_no_prior, emb_idx, pca_dir, 42, locked_hp["xgb_params"], pred_dir)
    o_np, _ = run_pca_model_schedule(spec, panel, "oot_test_schedule", "M6_xgb_150_no_prior", "xgb_gpu", tab_no_prior, emb_idx, pca_dir, 42, locked_hp["xgb_params"], pred_dir)
    val_preds["M6_xgb_150_no_prior"] = [v_np]
    oot_preds["M6_xgb_150_no_prior"] = [o_np]

    # Shuffled-label sanity check on 150-feature XGB GPU (OOT Test)
    shuf_o, _ = run_pca_model_schedule(spec, panel, "oot_test_schedule", "M3_xgb_gpu_150_tuned", "xgb_gpu", tab_cols, emb_idx, pca_dir, 42, locked_hp["xgb_params"], pred_dir, shuffle_labels=True)
    shuf_qm = quick_point_summary(shuf_o, tie_seed=42)
    pd.DataFrame([{"model": "M3_xgb_gpu_150_tuned_shuffled_oot", **shuf_qm}]).to_csv(out_dir / "stage2_shuffled_sanity_oot.csv", index=False)
    print(f"[Stage 2 Sanity] Shuffled OOT PR-AUC={shuf_qm['macro_pr_auc']:.5f} | Lift@1%={shuf_qm['macro_lift_1pct']:.2f}x", flush=True)

    # Summarize all Stage 2 models (Val & Locked OOT Test)
    val_m_rows, val_mac_rows, oot_mac_rows = [], [], []
    for eid in ("M1_logreg_150", "M2_lgbm_150_tuned", "M3_xgb_gpu_150_tuned", "M4_catboost_gpu_150_tuned", "M5_final_blend_150", "M6_xgb_150_no_prior"):
        nf = len(tab_no_prior) + len(emb_idx) if eid == "M6_xgb_150_no_prior" else locked_feat["n_total"]
        pm_v, mac_v = summarize_h3_predictions(val_preds[eid], val_months, True, n_boot=n_boot, boot_seed=boot_seed, tie_seed=tie_seed)
        for r in pm_v:
            val_m_rows.append({"exp_id": eid, "n_features": nf, **r})
        val_mac_rows.append({"exp_id": eid, "n_features": nf, "n_seeds": len(val_preds[eid]), **mac_v})

        _, mac_o = summarize_h3_predictions(oot_preds[eid], oot_months, True, n_boot=n_boot, boot_seed=boot_seed, tie_seed=tie_seed)
        oot_mac_rows.append({"exp_id": eid, "n_features": nf, "n_seeds": len(oot_preds[eid]), **mac_o})
        print(f"  -> {eid:26s} | Val PR={mac_v['average_precision_mean']:.5f} L1={mac_v['lift_1pct_mean']:.2f}x | OOT(2022-11) PR={mac_o['average_precision_mean']:.5f} L1={mac_o['lift_1pct_mean']:.2f}x", flush=True)

    pd.DataFrame(timing_rows).to_csv(out_dir / "stage2_model_timings.csv", index=False)
    pd.DataFrame(val_m_rows).to_csv(out_dir / "stage2_150_val_monthly.csv", index=False)
    pd.DataFrame(val_mac_rows).to_csv(out_dir / "stage2_150_val_macro.csv", index=False)
    pd.DataFrame(oot_mac_rows).to_csv(out_dir / "stage2_150_oot_test_summary.csv", index=False)

    # Paired bootstrap on Locked OOT Test: Final 150 vs Stage-1 Top-50 Baselines
    s1_pred_dir = work_dir / "outputs" / "h3_propensity" / "stage1" / "predictions"
    s1_cb_oot = [pd.read_parquet(s1_pred_dir / f"oot_test_schedule_B5_catboost_gpu_top50_seed{s}.parquet") for s in seeds if (s1_pred_dir / f"oot_test_schedule_B5_catboost_gpu_top50_seed{s}.parquet").exists()]
    s1_lr_oot = [pd.read_parquet(s1_pred_dir / "oot_test_schedule_B2_logreg_top50_seed42.parquet")]

    pair_defs = [
        ("M3_xgb_150_vs_B5_cb_top50", oot_preds["M3_xgb_gpu_150_tuned"], s1_cb_oot, "XGB GPU 150 vs CatBoost Top50"),
        ("M4_cb_150_vs_B5_cb_top50", oot_preds["M4_catboost_gpu_150_tuned"], s1_cb_oot, "CatBoost GPU 150 vs CatBoost Top50"),
        ("M5_blend_150_vs_B5_cb_top50", oot_preds["M5_final_blend_150"], s1_cb_oot, "Final Blend 150 vs CatBoost Top50"),
        ("M5_blend_150_vs_B2_lr_top50", oot_preds["M5_final_blend_150"], s1_lr_oot, "Final Blend 150 vs LogReg Top50"),
        ("M3_xgb_150_vs_M6_no_prior", oot_preds["M3_xgb_gpu_150_tuned"], oot_preds["M6_xgb_150_no_prior"], "XGB 150 (with prior) vs XGB 150 (no prior)"),
    ]
    pair_rows = []
    for pid, df_b_list, df_a_list, desc in pair_defs:
        if not df_a_list:
            continue
        res = paired_bootstrap_macro_monthly(df_a_list, df_b_list, oot_months, True, True, n_replicates=n_boot, boot_seed=boot_seed, tie_seed=tie_seed)
        pair_rows.append({"pair_id": pid, "desc": desc, **res})
    if pair_rows:
        pdf_p = pd.DataFrame(pair_rows)
        pdf_p["delta_pr_p_holm"] = holm_bonferroni(pdf_p["delta_average_precision_p_value"].tolist())
        pdf_p["delta_l1_p_holm"] = holm_bonferroni(pdf_p["delta_lift_1pct_p_value"].tolist())
        pdf_p.to_csv(out_dir / "stage2_oot_paired_bootstrap.csv", index=False)

    # Per-fold and Slice metrics on OOT Test for Final Blend (seed=42)
    oot_best = oot_preds["M5_final_blend_150"][0]
    fold_rows = []
    for k in range(5):
        fk = oot_best[oot_best["fold"] == k]
        fm = compute_slice_point_metrics(fk["y_true"].to_numpy(), fk["score"].to_numpy(), fk["client_id"].to_numpy(), seed=42)
        fold_rows.append({"fold": k, **{x: fm[x] for x in ("n", "positives", "prevalence", "average_precision", "roc_auc", "lift_1pct", "hits_1pct", "lift_5pct")}})
    pd.DataFrame(fold_rows).to_csv(out_dir / "stage2_oot_per_fold.csv", index=False)

    oot_meta = panel[panel["month"] == "2022-11"][["client_id", "cold_start", "trx_90d_has_event", "trx_all_has_event", "dialog_30d_has_event", "prior_all_bought_any"]]
    mrg = oot_best.merge(oot_meta, on="client_id", how="left")
    srows = []
    for sname, mk in (
        ("all_oot_2022_11", np.ones(len(mrg), dtype=bool)),
        ("cold_start_1", mrg["cold_start"].to_numpy() == 1),
        ("trx_all_zero", mrg["trx_all_has_event"].to_numpy() == 0),
        ("trx90_zero_dlg30_act", (mrg["trx_90d_has_event"].to_numpy() == 0) & (mrg["dialog_30d_has_event"].to_numpy() == 1)),
        ("prior_buyer_1", mrg["prior_all_bought_any"].fillna(0).to_numpy() == 1),
        ("prior_buyer_0", mrg["prior_all_bought_any"].fillna(0).to_numpy() == 0),
    ):
        sub = mrg.loc[mk]
        if len(sub) < 50 or sub["y_true"].sum() == 0:
            continue
        sm = compute_slice_point_metrics(sub["y_true"].to_numpy(), sub["score"].to_numpy(), sub["client_id"].to_numpy(), seed=42)
        srows.append({"slice": sname, "n": sm["n"], "positives": sm["positives"], "prevalence": sm["prevalence"], "pr_auc": sm["average_precision"], "lift_1pct": sm["lift_1pct"], "roc_auc": sm["roc_auc"]})
    pd.DataFrame(srows).to_csv(out_dir / "stage2_oot_slices.csv", index=False)

    _sync_to_drive(out_dir, args.drive_save_dir)
    print(f"\n[Stage 2 ALL DONE] Total elapsed={time.perf_counter()-t_all:.1f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
