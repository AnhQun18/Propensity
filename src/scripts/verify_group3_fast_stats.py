"""Fast verification script for Group 2.4 (missing 1.5% no-TRX slice) and Group 3:
- Missing paired comparisons: L4_cb - L1_rule_target_lag, L4_cb - L2_rule_trx_rfm, FE-v1 - L5c_lgbm_dialog_meta
- Holm-Bonferroni AND Benjamini-Hochberg (BH) FDR adjusted p-values
- Per-fold (fold=0..4) breakdown for all models (v1 & v2)
- Calibration Slope & Intercept (logistic recalibration y ~ a + b * logit(p)) replacing BSS
- Monthly buyer composition (repeat vs new buyer, base rate, call/trx/geo activity by month) for 2022-04..2023-01.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from propensity_model.metrics import compute_metrics, deterministic_tie_hash
from scripts.run_fe_v1_ablation import (
    BOOT_KEYS,
    _prep_sorted,
    _w_metrics_multi_k,
    holm_bonferroni,
)

PANEL_PATH = REPO_ROOT / "outputs" / "baseline_moscow_v1" / "panel_moscow.parquet"
FE_V1_PRED_DIR = REPO_ROOT / "outputs" / "fe_v1_ablation" / "predictions"
MOSCOW_PRED_DIR = REPO_ROOT / "outputs" / "baseline_moscow_v1" / "predictions"
V2_PRED_DIR = REPO_ROOT / "outputs" / "baseline_v2" / "predictions" / "B_rolling_ooc"
OUT_DIR = REPO_ROOT / "outputs" / "verification_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def benjamini_hochberg(p_values: list[float | None]) -> list[float | None]:
    """Step-up Benjamini-Hochberg FDR adjusted q-values."""
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None and np.isfinite(p)]
    m = len(indexed)
    adj: list[float | None] = [None] * len(p_values)
    if m == 0:
        return adj
    indexed.sort(key=lambda x: x[1])
    running_min = 1.0
    for rank_rev, (orig_i, p_raw) in enumerate(reversed(indexed)):
        rank = m - rank_rev
        val = min(1.0, (m / rank) * float(p_raw))
        running_min = min(running_min, val)
        adj[orig_i] = running_min
    return adj


def calibration_slope_intercept(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Fit logistic recalibration: logit(P(y=1)) = intercept + slope * logit(p_hat)."""
    y = np.asarray(y_true, dtype=np.int8)
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    lp = logit(p).reshape(-1, 1)
    if np.std(lp) < 1e-12:
        # Constant prediction (e.g. L0_base_rate) -> slope undefined (nan), intercept is logit(obs) - logit(p)
        obs_logit = float(logit(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6)))
        pred_logit = float(lp[0, 0])
        return float("nan"), obs_logit - pred_logit
    clf = LogisticRegression(C=1e9, solver="lbfgs", max_iter=500)
    clf.fit(lp, y)
    return float(clf.coef_[0, 0]), float(clf.intercept_[0])


def compute_missing_trx_slice_and_per_fold_and_cal() -> None:
    panel_meta = pd.read_parquet(
        PANEL_PATH,
        columns=[
            "client_id", "month", "fold", "y_any",
            "trx_all_has_event", "trx_90d_has_event", "cold_start",
            "prior_1m_bought_any", "prior_all_bought_any",
            "dialog_90d_has_event", "geo_90d_has_event",
        ],
    )
    panel_meta["month"] = panel_meta["month"].astype(str)

    # 1. Monthly composition analysis across all 10 modeling months (2022-04..2023-01)
    monthly_comp = (
        panel_meta.groupby("month")
        .agg(
            n=("client_id", "count"),
            positives=("y_any", "sum"),
            base_rate=("y_any", "mean"),
            share_trx_90d=("trx_90d_has_event", "mean"),
            share_trx_all=("trx_all_has_event", "mean"),
            share_geo_90d=("geo_90d_has_event", "mean"),
            share_dialog_90d=("dialog_90d_has_event", "mean"),
            share_cold_start=("cold_start", "mean"),
            share_prior_1m=("prior_1m_bought_any", "mean"),
            share_prior_all=("prior_all_bought_any", "mean"),
            pos_repeat_prior_all=("y_any", lambda s: int(s[panel_meta.loc[s.index, "prior_all_bought_any"] == 1].sum())),
            pos_repeat_prior_1m=("y_any", lambda s: int(s[panel_meta.loc[s.index, "prior_1m_bought_any"] == 1].sum())),
        )
        .reset_index()
    )
    monthly_comp["pos_new_no_prior_all"] = monthly_comp["positives"] - monthly_comp["pos_repeat_prior_all"]
    monthly_comp["repeat_share_of_positives"] = monthly_comp["pos_repeat_prior_all"] / monthly_comp["positives"]
    monthly_comp.to_csv(OUT_DIR / "monthly_composition_04_to_01.csv", index=False)

    # Models to inspect from fe_v1_ablation + L5c_lgbm_dialog_meta from baseline_moscow_v1
    models = [
        ("L0_base_rate", [42], FE_V1_PRED_DIR, True),
        ("L1_rule_target_lag", [42], FE_V1_PRED_DIR, False),
        ("L2_rule_trx_rfm", [42], FE_V1_PRED_DIR, False),
        ("L3_logreg_rfm", [42, 142, 242], FE_V1_PRED_DIR, True),
        ("B-raw-T", [42, 142, 242], FE_V1_PRED_DIR, True),
        ("B-raw-TGD", [42, 142, 242], FE_V1_PRED_DIR, True),
        ("B-raw-TGD_logreg", [42, 142, 242], FE_V1_PRED_DIR, True),
        ("L4_cb", [42, 142, 242], FE_V1_PRED_DIR, True),
        ("M_deriv_cb", [42], FE_V1_PRED_DIR, True),
        ("L5c_cb", [42], FE_V1_PRED_DIR, True),
        ("L5c_lgbm_dialog_meta", [42], MOSCOW_PRED_DIR, True),
        ("FE-v1", [42, 142, 242], FE_V1_PRED_DIR, True),
    ]

    slice_rows = []
    fold_rows = []
    cal_rows = []

    for stage in ("val", "test"):
        meta_stage = panel_meta[panel_meta["month"].isin(["2022-09", "2022-10"] if stage == "val" else ["2022-11", "2022-12", "2023-01"])].copy()
        meta_stage = meta_stage.sort_values(["month", "client_id"]).reset_index(drop=True)
        tie_h = deterministic_tie_hash(meta_stage["client_id"].to_numpy(), seed=42)
        y_true = meta_stage["y_any"].to_numpy(dtype=np.int8)

        for eid, seeds, pdir, is_prob in models:
            for s in seeds:
                pfile = pdir / stage / (f"{eid}.parquet" if pdir == MOSCOW_PRED_DIR else f"{eid}_seed{s}.parquet")
                pdf = pd.read_parquet(pfile).sort_values(["month", "client_id"]).reset_index(drop=True)
                assert np.array_equal(pdf["client_id"].to_numpy(), meta_stage["client_id"].to_numpy())
                assert np.array_equal(pdf["month"].to_numpy(), meta_stage["month"].to_numpy())
                scores = pdf["score"].to_numpy(dtype=np.float64)

                # Per-fold metrics (fold=0..4)
                for fk in range(5):
                    fmask = meta_stage["fold"].to_numpy() == fk
                    fm, _ = compute_metrics(
                        y_true[fmask], scores[fmask], meta_stage["client_id"].to_numpy()[fmask],
                        budget_fractions=(0.01, 0.05), primary_budget=0.01,
                        seed=42, is_probabilistic=is_prob, tie_hashes=tie_h[fmask],
                    )
                    fold_rows.append({
                        "stage": stage, "exp_id": eid, "seed": s, "fold": fk,
                        "n": fm["n"], "positives": fm["positives"], "prevalence": fm["prevalence"],
                        "pr_auc": fm["average_precision"], "roc_auc": fm["roc_auc"],
                        "lift_1pct": fm["lift_1pct"], "hits_1pct": fm["hits_1pct"],
                        "lift_5pct": fm["lift_5pct"],
                    })

                # Calibration Slope & Intercept (replaces BSS)
                if is_prob:
                    slope, intercept = calibration_slope_intercept(y_true, scores)
                    pm_all, _ = compute_metrics(
                        y_true, scores, meta_stage["client_id"].to_numpy(),
                        budget_fractions=(0.01,), primary_budget=0.01,
                        seed=42, is_probabilistic=True, tie_hashes=tie_h,
                    )
                    cal_rows.append({
                        "protocol": "v1_flat_ooc", "stage": stage, "exp_id": eid, "seed": s,
                        "n": pm_all["n"], "positives": pm_all["positives"], "prevalence": pm_all["prevalence"],
                        "mean_pred": float(np.mean(scores)),
                        "calibration_ratio": pm_all["calibration_ratio"],
                        "calibration_slope": slope,
                        "calibration_intercept": intercept,
                        "log_loss": pm_all["log_loss"],
                        "ece_quantile": pm_all["ece_quantile"],
                    })

                # Slice including missing 1.5% TRX (trx_all_has_event == 0), trx_90d_has_event == 0, cold_start == 1
                for scol in ("trx_all_has_event", "trx_90d_has_event", "cold_start"):
                    for sval in (0, 1):
                        smask = meta_stage[scol].to_numpy() == sval
                        sm, _ = compute_metrics(
                            y_true[smask], scores[smask], meta_stage["client_id"].to_numpy()[smask],
                            budget_fractions=(0.01, 0.05), primary_budget=0.01,
                            seed=42, is_probabilistic=is_prob, tie_hashes=tie_h[smask],
                        )
                        slice_rows.append({
                            "stage": stage, "exp_id": eid, "seed": s,
                            "slice_col": scol, "slice_val": sval,
                            "n": sm["n"], "positives": sm["positives"], "prevalence": sm["prevalence"],
                            "pr_auc": sm["average_precision"], "roc_auc": sm["roc_auc"],
                            "lift_1pct": sm["lift_1pct"], "hits_1pct": sm["hits_1pct"],
                            "lift_5pct": sm["lift_5pct"],
                        })

        # Also compute Calibration Slope & Intercept for Baseline v2 Protocol B models
        for eid_v2, is_prob_v2 in (
            ("L0_base_rate", True),
            ("B_logreg_T", True),
            ("B_logreg_TGD", True),
            ("B_catboost_TGD_tuned", True),
        ):
            v2_seeds = [42] if eid_v2 == "L0_base_rate" else [42, 142, 242]
            for s in v2_seeds:
                pfile = V2_PRED_DIR / stage / f"{eid_v2}_seed{s}.parquet"
                if not pfile.exists():
                    continue
                pdf = pd.read_parquet(pfile).sort_values(["month", "client_id"]).reset_index(drop=True)
                y_v2 = pdf["y_true"].to_numpy(dtype=np.int8)
                sc_v2 = pdf["score"].to_numpy(dtype=np.float64)
                slope, intercept = calibration_slope_intercept(y_v2, sc_v2)
                pm_v2, _ = compute_metrics(
                    y_v2, sc_v2, pdf["client_id"].to_numpy(),
                    budget_fractions=(0.01,), primary_budget=0.01,
                    seed=42, is_probabilistic=True,
                )
                cal_rows.append({
                    "protocol": "v2_rolling_ooc", "stage": stage, "exp_id": eid_v2, "seed": s,
                    "n": pm_v2["n"], "positives": pm_v2["positives"], "prevalence": pm_v2["prevalence"],
                    "mean_pred": float(np.mean(sc_v2)),
                    "calibration_ratio": pm_v2["calibration_ratio"],
                    "calibration_slope": slope,
                    "calibration_intercept": intercept,
                    "log_loss": pm_v2["log_loss"],
                    "ece_quantile": pm_v2["ece_quantile"],
                })

    pd.DataFrame(fold_rows).to_csv(OUT_DIR / "per_fold_metrics_detailed.csv", index=False)
    fold_agg = (
        pd.DataFrame(fold_rows)
        .groupby(["stage", "exp_id", "fold"])
        .agg(
            n=("n", "mean"),
            positives=("positives", "mean"),
            pr_auc_mean=("pr_auc", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            lift_1pct_mean=("lift_1pct", "mean"),
            hits_1pct_mean=("hits_1pct", "mean"),
        )
        .reset_index()
    )
    fold_agg.to_csv(OUT_DIR / "per_fold_metrics_summary.csv", index=False)

    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(OUT_DIR / "calibration_slope_intercept_all.csv", index=False)
    cal_agg = (
        cal_df.groupby(["protocol", "stage", "exp_id"])
        .agg(
            n_seeds=("seed", "count"),
            prevalence=("prevalence", "mean"),
            mean_pred=("mean_pred", "mean"),
            calibration_ratio=("calibration_ratio", "mean"),
            calibration_slope_mean=("calibration_slope", "mean"),
            calibration_slope_sd=("calibration_slope", "std"),
            calibration_intercept_mean=("calibration_intercept", "mean"),
            calibration_intercept_sd=("calibration_intercept", "std"),
            log_loss=("log_loss", "mean"),
            ece_quantile=("ece_quantile", "mean"),
        )
        .reset_index()
    )
    cal_agg.to_csv(OUT_DIR / "calibration_slope_intercept_summary.csv", index=False)

    sl_agg = (
        pd.DataFrame(slice_rows)
        .groupby(["stage", "slice_col", "slice_val", "exp_id"])
        .agg(
            n_seeds=("seed", "count"),
            n=("n", "mean"),
            positives=("positives", "mean"),
            prevalence=("prevalence", "mean"),
            pr_auc_mean=("pr_auc", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            lift_1pct_mean=("lift_1pct", "mean"),
            hits_1pct_mean=("hits_1pct", "mean"),
            lift_5pct_mean=("lift_5pct", "mean"),
        )
        .reset_index()
    )
    sl_agg.to_csv(OUT_DIR / "slice_metrics_with_missing_trx.csv", index=False)


def run_missing_paired_tests_with_holm_and_bh() -> None:
    """Compute paired client-clustered bootstrap (1,000 reps) for full primary + missing pairs + Holm & BH."""
    pairs = [
        # Original 7 primary pairs from fe_v1_evaluation_spec.json
        ("C1", "B-raw-T", "L3_logreg_rfm", "Đổi họ mô hình trên 8 biến TRX (CatBoost - LogReg)"),
        ("C2", "L4_cb", "B-raw-T", "FE trên riêng TRX (63 - 8 biến, CatBoost)"),
        ("C3", "B-raw-TGD", "B-raw-T", "Thêm GEO + DIALOG thô (14 - 8 biến, CatBoost)"),
        ("C4", "FE-v1", "B-raw-TGD", "FE-v1 (139 biến) - B-raw-TGD (14 biến)"),
        ("C5", "FE-v1", "L4_cb", "Đóng góp tổng GEO + DIALOG (FE-v1 - L4_cb)"),
        ("C6a", "FE-v1", "L1_rule_target_lag", "FE-v1 - L1_rule_target_lag"),
        ("C6b", "FE-v1", "L2_rule_trx_rfm", "FE-v1 - L2_rule_trx_rfm"),
        # Missing pairs requested in Group 3.1
        ("M1_L4_vs_L1", "L4_cb", "L1_rule_target_lag", "L4_cb (TRX 63) - L1_rule_target_lag"),
        ("M2_L4_vs_L2", "L4_cb", "L2_rule_trx_rfm", "L4_cb (TRX 63) - L2_rule_trx_rfm"),
        ("M3_FEv1_vs_L5cLGBM", "FE-v1", "L5c_lgbm_dialog_meta", "FE-v1 (CatBoost 139) - L5c_lgbm_dialog_meta (112)"),
        # Also auxiliary 2x2 pairs
        ("A1", "B-raw-TGD_logreg", "L3_logreg_rfm", "LogReg: + GEO+DIALOG thô (14 - 8)"),
        ("A2", "B-raw-TGD", "B-raw-TGD_logreg", "14 biến thô: CatBoost - LogReg"),
    ]

    model_info = {
        "L0_base_rate": ([42], FE_V1_PRED_DIR, True),
        "L1_rule_target_lag": ([42], FE_V1_PRED_DIR, False),
        "L2_rule_trx_rfm": ([42], FE_V1_PRED_DIR, False),
        "L3_logreg_rfm": ([42, 142, 242], FE_V1_PRED_DIR, True),
        "B-raw-T": ([42, 142, 242], FE_V1_PRED_DIR, True),
        "B-raw-TGD": ([42, 142, 242], FE_V1_PRED_DIR, True),
        "B-raw-TGD_logreg": ([42, 142, 242], FE_V1_PRED_DIR, True),
        "L4_cb": ([42, 142, 242], FE_V1_PRED_DIR, True),
        "L5c_lgbm_dialog_meta": ([42], MOSCOW_PRED_DIR, True),
        "FE-v1": ([42, 142, 242], FE_V1_PRED_DIR, True),
    }

    all_rows = []
    n_boot = 1000
    for stage in ("val", "test"):
        ref_df = pd.read_parquet(FE_V1_PRED_DIR / stage / "L0_base_rate_seed42.parquet").sort_values(["month", "client_id"]).reset_index(drop=True)
        codes, uniq = pd.factorize(ref_df["client_id"].to_numpy(), sort=True)
        n_clients = len(uniq)
        tie_h = deterministic_tie_hash(ref_df["client_id"].to_numpy(), seed=42)
        y_ref = ref_df["y_true"].to_numpy(dtype=np.int8)

        rng = np.random.default_rng(42)
        boot_weights = [np.bincount(rng.integers(0, n_clients, size=n_clients), minlength=n_clients) for _ in range(n_boot)]
        ones_w = np.ones(n_clients, dtype=np.int64)

        boot_arrs: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        pt_vals: dict[tuple[str, int], dict[str, float]] = {}

        needed_eids = sorted({p[1] for p in pairs} | {p[2] for p in pairs})
        for eid in needed_eids:
            seeds, pdir, is_prob = model_info[eid]
            for s in seeds:
                pfile = pdir / stage / (f"{eid}.parquet" if pdir == MOSCOW_PRED_DIR else f"{eid}_seed{s}.parquet")
                pdf = pd.read_parquet(pfile).sort_values(["month", "client_id"]).reset_index(drop=True)
                pr = _prep_sorted(y_ref, pdf["score"].to_numpy(dtype=np.float64), codes, tie_h, is_prob)
                pt_vals[(eid, s)] = _w_metrics_multi_k(pr, ones_w)
                arrs = {k: np.empty(n_boot, dtype=np.float64) for k in BOOT_KEYS}
                for b_idx, cw in enumerate(boot_weights):
                    wm = _w_metrics_multi_k(pr, cw)
                    for k in BOOT_KEYS:
                        arrs[k][b_idx] = wm[k]
                boot_arrs[(eid, s)] = arrs

        stage_rows = []
        for pid, bid, aid, qtext in pairs:
            sa = model_info[aid][0]
            sb = model_info[bid][0]
            pt_a = {k: float(np.mean([pt_vals[(aid, s)][k] for s in sa])) for k in BOOT_KEYS}
            pt_b = {k: float(np.mean([pt_vals[(bid, s)][k] for s in sb])) for k in BOOT_KEYS}
            row = {
                "stage": stage, "pair_id": pid, "model_b": bid, "model_a": aid, "question": qtext,
                "pr_auc_a": pt_a["average_precision"], "pr_auc_b": pt_b["average_precision"],
                "lift_1pct_a": pt_a["lift_1pct"], "lift_1pct_b": pt_b["lift_1pct"],
                "roc_auc_a": pt_a["roc_auc"], "roc_auc_b": pt_b["roc_auc"],
            }
            for met in ("average_precision", "lift_1pct", "lift_5pct", "roc_auc"):
                tag = "pr_auc" if met == "average_precision" else met
                ba = np.mean([boot_arrs[(aid, s)][met] for s in sa], axis=0)
                bb = np.mean([boot_arrs[(bid, s)][met] for s in sb], axis=0)
                deltas = bb - ba
                delta_pt = pt_b[met] - pt_a[met]
                rel_pct = (delta_pt / abs(pt_a[met]) * 100.0) if abs(pt_a[met]) > 1e-12 else float("nan")
                p_raw = min(1.0, 2.0 * min(float(np.mean(deltas <= 0.0)), float(np.mean(deltas >= 0.0))))
                row[f"delta_{tag}"] = delta_pt
                row[f"rel_delta_{tag}_pct"] = rel_pct
                row[f"delta_{tag}_ci_low"] = float(np.percentile(deltas, 2.5))
                row[f"delta_{tag}_ci_high"] = float(np.percentile(deltas, 97.5))
                row[f"delta_{tag}_p_raw"] = p_raw
            stage_rows.append(row)

        # Apply Holm & BH across the 10 main/missing pairs (exclude A1, A2 auxiliary so family M=10)
        main_idx = [i for i, r in enumerate(stage_rows) if not r["pair_id"].startswith("A")]
        for met_tag in ("pr_auc", "lift_1pct", "roc_auc"):
            p_raws = [stage_rows[i][f"delta_{met_tag}_p_raw"] for i in main_idx]
            p_holm = holm_bonferroni(p_raws)
            p_bh = benjamini_hochberg(p_raws)
            for idx_pos, i in enumerate(main_idx):
                stage_rows[i][f"delta_{met_tag}_p_holm_m10"] = p_holm[idx_pos]
                stage_rows[i][f"delta_{met_tag}_p_bh_m10"] = p_bh[idx_pos]
        all_rows.extend(stage_rows)

    pd.DataFrame(all_rows).to_csv(OUT_DIR / "paired_full_with_missing_holm_bh.csv", index=False)


def main():
    compute_missing_trx_slice_and_per_fold_and_cal()
    print("[1/2] Per-fold, missing 1.5% TRX slice, calibration slope/intercept, monthly comp done.", flush=True)
    run_missing_paired_tests_with_holm_and_bh()
    print("[2/2] Paired bootstrap with missing pairs + Holm + BH done.", flush=True)


if __name__ == "__main__":
    main()
