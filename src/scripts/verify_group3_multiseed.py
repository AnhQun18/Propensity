"""Group 3 Multi-Seed & Per-Fold Verification:
1. Completes missing seeds (142, 242) for L5c_cb and 5 seeds (42, 142, 242, 342, 442) for L5c_lgbm_dialog_meta
   under Protocol A (Val & Test), plus seeds 342, 442 for B_catboost_TGD_tuned under Protocol B (5 seeds total).
2. Computes exact multi-seed summaries (mean +/- sd across seeds) and per-fold breakdowns (fold=0..4),
   explicitly comparing Pooled OOF vs Mean-Across-Folds.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from baseline_v2.models import CatBoostTunedModel
from baseline_v2.spec import load_spec
from propensity_model.metrics import compute_metrics, deterministic_tie_hash

import shutil
import subprocess

def _detect_gpu() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=5).returncode == 0
    except Exception:
        return False

USE_GPU = _detect_gpu()

PANEL_PATH = REPO_ROOT / "outputs" / "baseline_moscow_v1" / "panel_moscow.parquet"
FE1_PRED_DIR = REPO_ROOT / "outputs" / "fe_v1_ablation" / "predictions"
V2_PRED_DIR = REPO_ROOT / "outputs" / "baseline_v2" / "predictions"
OUT_DIR = REPO_ROOT / "outputs" / "verification_audit"
MS_PRED_DIR = OUT_DIR / "multiseed_preds"
MS_PRED_DIR.mkdir(parents=True, exist_ok=True)


def run_missing_protocol_a_seeds(seeds: tuple[int, ...] = (42,)) -> None:
    print(f"[MultiSeed 1/2] Checking Protocol A seeds={seeds} for L5c_cb & L5c_lgbm_dialog_meta ...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    pcols = list(panel.columns)
    cols_112 = [c for c in pcols if c.startswith(("trx_", "geo_", "dialog_"))] + ["cold_start"]
    base_pred_dir = REPO_ROOT / "outputs" / "baseline_moscow_v1" / "predictions"

    for stage, tr_m, ev_m in (
        ("val", ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08"], ["2022-09", "2022-10"]),
        ("test", ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08", "2022-09", "2022-10"], ["2022-11", "2022-12", "2023-01"]),
    ):
        df_tr = panel[panel["month"].isin(tr_m)].reset_index(drop=True)
        df_ev = panel[panel["month"].isin(ev_m)].reset_index(drop=True)

        # L5c_cb (seed 42 already in FE1_PRED_DIR; only run extra seeds if requested)
        for seed in [s for s in seeds if s != 42]:
            out_p = MS_PRED_DIR / f"{stage}_L5c_cb_seed{seed}.parquet"
            fe_p = FE1_PRED_DIR / stage / f"L5c_cb_seed{seed}.parquet"
            if out_p.exists() or fe_p.exists():
                continue
            t0 = time.perf_counter()
            fdfs = []
            for k in range(5):
                tr_k = df_tr["fold"].to_numpy() != k
                ev_k = df_ev["fold"].to_numpy() == k
                cb_kw = dict(iterations=500, random_seed=seed, border_count=128, loss_function="Logloss", verbose=False, allow_writing_files=False)
                if USE_GPU:
                    cb_kw["task_type"] = "GPU"
                else:
                    cb_kw["thread_count"] = -1
                clf = CatBoostClassifier(**cb_kw)
                clf.fit(df_tr.loc[tr_k, cols_112].to_numpy(dtype=np.float32), df_tr.loc[tr_k, "y_any"].to_numpy(dtype=np.int8))
                sc = clf.predict_proba(df_ev.loc[ev_k, cols_112].to_numpy(dtype=np.float32))[:, 1].astype(np.float64)
                fdfs.append(pd.DataFrame({
                    "client_id": df_ev.loc[ev_k, "client_id"].to_numpy(),
                    "fold": df_ev.loc[ev_k, "fold"].to_numpy(),
                    "month": df_ev.loc[ev_k, "month"].to_numpy(),
                    "y_true": df_ev.loc[ev_k, "y_any"].to_numpy(dtype=np.int8),
                    "score": sc,
                }))
            pdf = pd.concat(fdfs, ignore_index=True).sort_values(["month", "client_id"]).reset_index(drop=True)
            pdf.to_parquet(out_p, index=False, compression="zstd")
            print(f"  [{stage}] L5c_cb seed={seed} done ({time.perf_counter()-t0:.1f}s)", flush=True)

        # L5c_lgbm_dialog_meta (seed 42 already in baseline_moscow_v1/predictions; run only missing requested seeds)
        for seed in seeds:
            out_p = MS_PRED_DIR / f"{stage}_L5c_lgbm_seed{seed}.parquet"
            base_p = base_pred_dir / stage / "L5c_lgbm_dialog_meta.parquet" if seed == 42 else None
            if out_p.exists() or (base_p is not None and base_p.exists()):
                continue
            t0 = time.perf_counter()
            fdfs = []
            for k in range(5):
                tr_k = df_tr["fold"].to_numpy() != k
                ev_k = df_ev["fold"].to_numpy() == k
                clf = lgb.LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1)
                clf.fit(df_tr.loc[tr_k, cols_112].to_numpy(dtype=np.float32), df_tr.loc[tr_k, "y_any"].to_numpy(dtype=np.int8))
                sc = clf.predict_proba(df_ev.loc[ev_k, cols_112].to_numpy(dtype=np.float32))[:, 1].astype(np.float64)
                fdfs.append(pd.DataFrame({
                    "client_id": df_ev.loc[ev_k, "client_id"].to_numpy(),
                    "fold": df_ev.loc[ev_k, "fold"].to_numpy(),
                    "month": df_ev.loc[ev_k, "month"].to_numpy(),
                    "y_true": df_ev.loc[ev_k, "y_any"].to_numpy(dtype=np.int8),
                    "score": sc,
                }))
            pdf = pd.concat(fdfs, ignore_index=True).sort_values(["month", "client_id"]).reset_index(drop=True)
            pdf.to_parquet(out_p, index=False, compression="zstd")
            print(f"  [{stage}] L5c_lgbm seed={seed} done ({time.perf_counter()-t0:.1f}s)", flush=True)

    del panel
    gc.collect()


def run_extra_protocol_b_seeds(seeds: tuple[int, ...] = (42,)) -> None:
    extra = [s for s in seeds if s in (342, 442)]
    if not extra:
        print("[MultiSeed 2/2] Single-seed mode: skipping extra Protocol B seeds (using existing seed=42/142/242 predictions).", flush=True)
        return
    print(f"[MultiSeed 2/2] Running Protocol B B_catboost_TGD_tuned extra seeds={extra} ...", flush=True)
    spec = load_spec(REPO_ROOT / "configs" / "baseline_v2_evaluation_spec.json")
    tuned = json.loads((REPO_ROOT / "outputs" / "baseline_v2" / "tuned_catboost_config.json").read_text(encoding="utf-8"))
    fcols = list(spec["feature_sets"]["raw_TGD_14"])
    panel = pd.read_parquet(PANEL_PATH, columns=["client_id", "fold", "month", "y_any"] + fcols)

    for seed in extra:
        for st, sched in (("val", spec["protocols"]["B_rolling_ooc"]["val_schedule"]), ("test", spec["protocols"]["B_rolling_ooc"]["test_schedule"])):
            out_p = MS_PRED_DIR / f"protB_{st}_B_catboost_TGD_tuned_seed{seed}.parquet"
            if out_p.exists():
                continue
            t0 = time.perf_counter()
            step_dfs = []
            for step in sched:
                ev_m = step["eval_month"]
                tr_ms = step["train_months"]
                df_c = panel[panel["month"].isin(tr_ms[:-1])]
                df_es = panel[panel["month"].isin([tr_ms[-1]])]
                df_ev = panel[panel["month"] == ev_m]
                for k in range(5):
                    tc = df_c[df_c["fold"] != k]
                    te = df_es[df_es["fold"] != k]
                    ev = df_ev[df_ev["fold"] == k]
                    m = CatBoostTunedModel(
                        seed=seed, learning_rate=tuned["learning_rate"], depth=tuned["depth"],
                        l2_leaf_reg=tuned["l2_leaf_reg"], max_iterations=tuned["max_iterations"],
                        early_stopping_rounds=tuned["early_stopping_rounds"], thread_count=4,
                    )
                    m.fit(
                        tc[fcols].to_numpy(dtype=np.float32), tc["y_any"].to_numpy(dtype=np.int8),
                        X_es=te[fcols].to_numpy(dtype=np.float32), y_es=te["y_any"].to_numpy(dtype=np.int8),
                    )
                    sc = m.predict_scores(ev[fcols].to_numpy(dtype=np.float32))
                    step_dfs.append(pd.DataFrame({
                        "client_id": ev["client_id"].to_numpy(),
                        "fold": ev["fold"].to_numpy(),
                        "month": ev["month"].to_numpy(),
                        "y_true": ev["y_any"].to_numpy(dtype=np.int8),
                        "score": sc,
                    }))
            pdf = pd.concat(step_dfs, ignore_index=True).sort_values(["month", "client_id"]).reset_index(drop=True)
            pdf.to_parquet(out_p, index=False, compression="zstd")
            print(f"  [ProtB {st}] B_catboost_TGD_tuned seed={seed} done ({time.perf_counter()-t0:.1f}s)", flush=True)


def summarize_all_multiseed() -> None:
    print("[MultiSeed Summary] Aggregating Pooled OOF vs Mean-Across-Folds across all seeds ...", flush=True)
    # Protocol A summary
    prot_a_rows = []
    for st in ("val", "test"):
        models_paths: dict[str, list[tuple[int, Path]]] = {}
        for eid in ("B-raw-T", "B-raw-TGD", "L4_cb", "FE-v1"):
            models_paths[eid] = [(s, FE1_PRED_DIR / st / f"{eid}_seed{s}.parquet") for s in (42, 142, 242) if (FE1_PRED_DIR / st / f"{eid}_seed{s}.parquet").exists()]
        # Deterministic LogReg (lbfgs): always report single seed 42 only
        for eid in ("L3_logreg_rfm", "B-raw-TGD_logreg"):
            p42 = FE1_PRED_DIR / st / f"{eid}_seed42.parquet"
            if p42.exists():
                models_paths[eid] = [(42, p42)]
        models_paths["L5c_cb"] = [
            (42, FE1_PRED_DIR / st / "L5c_cb_seed42.parquet"),
            (142, MS_PRED_DIR / f"{st}_L5c_cb_seed142.parquet"),
            (242, MS_PRED_DIR / f"{st}_L5c_cb_seed242.parquet"),
        ]
        models_paths["L5c_lgbm_dialog_meta"] = [
            (42, REPO_ROOT / "outputs" / "baseline_moscow_v1" / "predictions" / st / "L5c_lgbm_dialog_meta.parquet"),
            *[(s, MS_PRED_DIR / f"{st}_L5c_lgbm_seed{s}.parquet") for s in (42, 142, 242, 342, 442)],
        ]

        for eid, s_list in models_paths.items():
            for s, p in s_list:
                if not p.exists():
                    continue
                df = pd.read_parquet(p)
                th = deterministic_tie_hash(df["client_id"].to_numpy(), seed=42)
                m_pool, _ = compute_metrics(df["y_true"].to_numpy(), df["score"].to_numpy(), df["client_id"].to_numpy(),
                                            budget_fractions=(0.01, 0.05), primary_budget=0.01, seed=42, is_probabilistic=True, tie_hashes=th)
                fold_aps, fold_l1s, fold_aucs = [], [], []
                for k in range(5):
                    fk = df[df["fold"] == k]
                    mk, _ = compute_metrics(fk["y_true"].to_numpy(), fk["score"].to_numpy(), fk["client_id"].to_numpy(),
                                            budget_fractions=(0.01,), primary_budget=0.01, seed=42, is_probabilistic=True)
                    fold_aps.append(mk["average_precision"])
                    fold_l1s.append(mk["lift_1pct"])
                    fold_aucs.append(mk["roc_auc"])
                prot_a_rows.append({
                    "protocol": "A_flat_ooc", "stage": st, "exp_id": eid, "seed": s,
                    "pooled_pr_auc": m_pool["average_precision"], "pooled_roc_auc": m_pool["roc_auc"],
                    "pooled_lift_1pct": m_pool["lift_1pct"], "pooled_hits_1pct": m_pool["hits_1pct"],
                    "mean_fold_pr_auc": float(np.mean(fold_aps)), "std_fold_pr_auc": float(np.std(fold_aps, ddof=1)),
                    "mean_fold_lift_1pct": float(np.mean(fold_l1s)), "mean_fold_roc_auc": float(np.mean(fold_aucs)),
                })

    # Protocol B B_catboost_TGD_tuned 5 seeds
    for st in ("val", "test"):
        for s in (42, 142, 242, 342, 442):
            p = (V2_PRED_DIR / "B_rolling_ooc" / st / f"B_catboost_TGD_tuned_seed{s}.parquet") if s in (42, 142, 242) else (MS_PRED_DIR / f"protB_{st}_B_catboost_TGD_tuned_seed{s}.parquet")
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            m_pool, _ = compute_metrics(df["y_true"].to_numpy(), df["score"].to_numpy(), df["client_id"].to_numpy(),
                                        budget_fractions=(0.01, 0.05), primary_budget=0.01, seed=42, is_probabilistic=True)
            m_aps, m_l1s, m_aucs = [], [], []
            for mo, grp in df.groupby("month"):
                mm, _ = compute_metrics(grp["y_true"].to_numpy(), grp["score"].to_numpy(), grp["client_id"].to_numpy(),
                                        budget_fractions=(0.01,), primary_budget=0.01, seed=42, is_probabilistic=True)
                m_aps.append(mm["average_precision"])
                m_l1s.append(mm["lift_1pct"])
                m_aucs.append(mm["roc_auc"])
            prot_a_rows.append({
                "protocol": "B_rolling_ooc", "stage": st, "exp_id": "B_catboost_TGD_tuned", "seed": s,
                "pooled_pr_auc": m_pool["average_precision"], "pooled_roc_auc": m_pool["roc_auc"],
                "pooled_lift_1pct": m_pool["lift_1pct"], "pooled_hits_1pct": m_pool["hits_1pct"],
                "mean_fold_pr_auc": float(np.mean(m_aps)), "std_fold_pr_auc": float(np.std(m_aps, ddof=1)),
                "mean_fold_lift_1pct": float(np.mean(m_l1s)), "mean_fold_roc_auc": float(np.mean(m_aucs)),
            })

    out_df = pd.DataFrame(prot_a_rows)
    out_df.to_csv(OUT_DIR / "group3_multiseed_detailed.csv", index=False)
    agg = out_df.groupby(["protocol", "stage", "exp_id"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        pooled_pr_auc_mean=("pooled_pr_auc", "mean"),
        pooled_pr_auc_sd=("pooled_pr_auc", "std"),
        pooled_lift_1pct_mean=("pooled_lift_1pct", "mean"),
        pooled_lift_1pct_sd=("pooled_lift_1pct", "std"),
        pooled_roc_auc_mean=("pooled_roc_auc", "mean"),
        mean_fold_or_mo_pr_auc=("mean_fold_pr_auc", "mean"),
        mean_fold_or_mo_lift_1pct=("mean_fold_lift_1pct", "mean"),
    )
    agg.to_csv(OUT_DIR / "group3_multiseed_summary.csv", index=False)
    print(agg.to_string(index=False), flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds (default 42 single-seed policy)")
    args = parser.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip()) or (42,)
    run_missing_protocol_a_seeds(seeds)
    run_extra_protocol_b_seeds(seeds)
    summarize_all_multiseed()


if __name__ == "__main__":
    main()
