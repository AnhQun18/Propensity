"""Validation-only CatBoost tuning, ablations (T+G, T+D, tenure, blend), PI, slices, and sanity checks."""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import (
    bootstrap_multiseed_macro_monthly,
    bootstrap_multiseed_per_month,
    compute_slice_point_metrics,
)
from .models import blend_prediction_frames, build_model
from .protocol import iter_protocol_steps, run_protocol_experiment
from .spec import extract_feature_matrix


def run_val_catboost_tuning(
    spec: dict[str, Any],
    panel_df: pd.DataFrame,
    out_dir: Path,
    thread_count: int = -1,
    n_boot: int = 1000,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Tune CatBoost (learning_rate, depth, l2_leaf_reg) strictly on Val using Protocol B."""
    cfg_path = out_dir / "tuned_catboost_config.json"
    grid_csv = out_dir / "metrics" / "val_catboost_tuning_grid.csv"
    if cfg_path.exists() and grid_csv.exists():
        return json.loads(cfg_path.read_text(encoding="utf-8")), pd.read_csv(grid_csv)

    tune_cfg = spec["catboost_tuning"]
    fixed = dict(tune_cfg["fixed_params"])
    feat_cols = spec["feature_sets"]["raw_TGD_14"]
    val_months = tuple(spec["data_contract"]["val_months"])
    seed = int(spec["evaluation"]["canonical_seed"])
    tie_seed = int(spec["evaluation"]["tie_break_seed"])
    boot_seed = int(spec["evaluation"]["bootstrap_seed"])

    rows: list[dict[str, Any]] = []
    best_score = (-1.0, -1.0)
    best_cfg: dict[str, Any] | None = None

    for idx, cand in enumerate(tune_cfg["grid"], start=1):
        params = {
            **fixed,
            "learning_rate": float(cand["learning_rate"]),
            "depth": int(cand["depth"]),
            "l2_leaf_reg": float(cand["l2_leaf_reg"]),
            "max_iterations": int(tune_cfg["max_iterations"]),
            "early_stopping_rounds": int(tune_cfg["early_stopping_rounds"]),
            "thread_count": int(thread_count),
            "eval_metric": "Logloss",
        }
        exp_tag = f"TuneCB_{idx}_lr{cand['learning_rate']}_d{cand['depth']}_l2_{cand['l2_leaf_reg']}"
        pdf, run_info = run_protocol_experiment(
            spec=spec,
            protocol_name="B_rolling_ooc",
            stage_name="val",
            panel_df=panel_df,
            exp_id=exp_tag,
            family="catboost_tuned",
            feature_cols=feat_cols,
            uses_target_lag=False,
            seed=seed,
            model_params=params,
            pred_cache_dir=out_dir / "predictions" / "_tuning_cache",
            use_temporal_es=True,
        )
        macro = bootstrap_multiseed_macro_monthly(
            [pdf], months=val_months, is_prob=True, n_replicates=n_boot, boot_seed=boot_seed, tie_seed=tie_seed
        )
        iters = run_info.get("best_iterations", [])
        med_iter = int(np.median(iters)) if iters else int(tune_cfg["max_iterations"])
        row = {
            "candidate_id": exp_tag,
            "learning_rate": float(cand["learning_rate"]),
            "depth": int(cand["depth"]),
            "l2_leaf_reg": float(cand["l2_leaf_reg"]),
            "median_best_iter": med_iter,
            "min_best_iter": int(min(iters)) if iters else med_iter,
            "max_best_iter": int(max(iters)) if iters else med_iter,
            "val_macro_lift_1pct": macro["lift_1pct_mean"],
            "val_macro_lift_1pct_ci_low": macro["lift_1pct_ci_low"],
            "val_macro_lift_1pct_ci_high": macro["lift_1pct_ci_high"],
            "val_macro_pr_auc": macro["average_precision_mean"],
            "val_macro_pr_auc_ci_low": macro["average_precision_ci_low"],
            "val_macro_pr_auc_ci_high": macro["average_precision_ci_high"],
            "val_macro_roc_auc": macro["roc_auc_mean"],
            "val_macro_log_loss": macro["log_loss_mean"],
            "elapsed_sec": run_info.get("elapsed_sec", 0.0),
        }
        rows.append(row)
        score = (float(macro["lift_1pct_mean"] or 0.0), float(macro["average_precision_mean"] or 0.0))
        if score > best_score:
            best_score = score
            best_cfg = {
                **fixed,
                "learning_rate": float(cand["learning_rate"]),
                "depth": int(cand["depth"]),
                "l2_leaf_reg": float(cand["l2_leaf_reg"]),
                "max_iterations": int(tune_cfg["max_iterations"]),
                "early_stopping_rounds": int(tune_cfg["early_stopping_rounds"]),
                "median_best_iter_val": med_iter,
                "selected_on": "val_protocol_b_macro_lift_1pct_then_pr_auc",
                "val_macro_lift_1pct": row["val_macro_lift_1pct"],
                "val_macro_pr_auc": row["val_macro_pr_auc"],
                "eval_metric": "Logloss",
            }

    grid_df = pd.DataFrame(rows).sort_values(["val_macro_lift_1pct", "val_macro_pr_auc"], ascending=[False, False]).reset_index(drop=True)
    grid_csv.parent.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(grid_csv, index=False)
    assert best_cfg is not None
    cfg_path.write_text(json.dumps(best_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return best_cfg, grid_df


def run_val_ablations(
    spec: dict[str, Any],
    panel_df: pd.DataFrame,
    tuned_cb_params: dict[str, Any],
    val_preds_seed42: dict[str, pd.DataFrame],
    out_dir: Path,
    thread_count: int = -1,
    n_boot: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Val-only ablations: T+G, T+D, tenure (drop/cap365), and LogReg+CatBoost ensembles."""
    val_months = tuple(spec["data_contract"]["val_months"])
    seed = int(spec["evaluation"]["canonical_seed"])
    tie_seed = int(spec["evaluation"]["tie_break_seed"])
    boot_seed = int(spec["evaluation"]["bootstrap_seed"])
    feat_sets = spec["feature_sets"]
    pred_dir = out_dir / "predictions"

    cb_params = {**tuned_cb_params, "thread_count": int(thread_count)}
    ablation_groups = [
        ("source_ablation", spec["val_only_ablations"]["source_ablations"]),
        ("tenure_ablation", spec["val_only_ablations"]["tenure_ablations"]),
    ]

    macro_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []

    def _record(group_name: str, exp_id: str, family: str, n_feats: int, pdf: pd.DataFrame) -> None:
        macro = bootstrap_multiseed_macro_monthly(
            [pdf], months=val_months, is_prob=True, n_replicates=n_boot, boot_seed=boot_seed, tie_seed=tie_seed
        )
        macro_rows.append({"group": group_name, "exp_id": exp_id, "family": family, "n_features": n_feats, **macro})
        for m in val_months:
            mm = bootstrap_multiseed_per_month(
                [pdf], month=m, is_prob=True, n_replicates=n_boot, boot_seed=boot_seed, tie_seed=tie_seed
            )
            monthly_rows.append({"group": group_name, "exp_id": exp_id, "family": family, "n_features": n_feats, **mm})

    # Include main Val reference points (T, TGD for LogReg & Tuned CatBoost) for side-by-side table
    for ref_id, fam, fkey in (
        ("B_logreg_T", "logreg", "raw_T_8"),
        ("B_logreg_TGD", "logreg", "raw_TGD_14"),
        ("B_catboost_TGD_tuned", "catboost_tuned", "raw_TGD_14"),
    ):
        if ref_id in val_preds_seed42:
            _record("reference_main", ref_id, fam, len(feat_sets[fkey]), val_preds_seed42[ref_id])

    for group_name, items in ablation_groups:
        for item in items:
            eid = item["exp_id"]
            fam = item["family"]
            fcols = feat_sets[item["feature_set"]]
            is_cb = (fam == "catboost_tuned")
            pdf, _ = run_protocol_experiment(
                spec=spec,
                protocol_name="B_rolling_ooc",
                stage_name="val",
                panel_df=panel_df,
                exp_id=eid,
                family=fam,
                feature_cols=fcols,
                uses_target_lag=False,
                seed=seed,
                model_params=cb_params if is_cb else None,
                pred_cache_dir=pred_dir,
                use_temporal_es=is_cb,
            )
            _record(group_name, eid, fam, len(fcols), pdf)

    # Linear + Tree ensemble ablations on Val
    if "B_logreg_TGD" in val_preds_seed42 and "B_catboost_TGD_tuned" in val_preds_seed42:
        df_lr = val_preds_seed42["B_logreg_TGD"]
        df_cb = val_preds_seed42["B_catboost_TGD_tuned"]
        for ens in spec["val_only_ablations"]["ensemble_ablations"]:
            eid = ens["exp_id"]
            bpdf = blend_prediction_frames(df_lr, df_cb, method=ens["method"], weight_logreg=float(ens["weight_logreg"]))
            _record("ensemble_ablation", eid, f"blend_{ens['method']}", 14, bpdf)

    macro_df = pd.DataFrame(macro_rows)
    monthly_df = pd.DataFrame(monthly_rows)
    m_dir = out_dir / "metrics"
    m_dir.mkdir(parents=True, exist_ok=True)
    macro_df.to_csv(m_dir / "val_ablation_macro.csv", index=False)
    monthly_df.to_csv(m_dir / "val_ablation_monthly.csv", index=False)
    return macro_df, monthly_df


def run_val_permutation_importance(
    spec: dict[str, Any],
    panel_df: pd.DataFrame,
    tuned_cb_params: dict[str, Any],
    out_dir: Path,
    thread_count: int = -1,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Compute OOF Permutation Importance strictly on Val under Protocol B (5 repeats)."""
    pi_path = out_dir / "metrics" / "val_permutation_importance.csv"
    if pi_path.exists():
        return pd.read_csv(pi_path)

    feat_cols = list(spec["feature_sets"]["raw_TGD_14"])
    seed = int(spec["evaluation"]["canonical_seed"])
    tie_seed = int(spec["evaluation"]["tie_break_seed"])
    n_folds = int(spec["data_contract"]["n_folds"])
    target_col = str(spec["data_contract"]["target_col"])
    fold_col = str(spec["data_contract"]["fold_col"])
    steps = iter_protocol_steps(spec, "B_rolling_ooc", "val")
    cb_params = {**tuned_cb_params, "thread_count": int(thread_count)}

    rows: list[dict[str, Any]] = []
    for exp_id, family, use_es in (
        ("B_logreg_TGD", "logreg", False),
        ("B_catboost_TGD_tuned", "catboost_tuned", True),
    ):
        # Train rolling models per (month, fold) and hold eval arrays for OOF permutation
        eval_blocks: list[dict[str, Any]] = []
        for tr_months, ev_months in steps:
            df_ev_step = panel_df.loc[panel_df["month"].isin(ev_months)].reset_index(drop=True)
            can_es = bool(use_es and len(tr_months) >= 2)
            if can_es:
                df_tr_core = panel_df.loc[panel_df["month"].isin(tr_months[:-1])].reset_index(drop=True)
                df_tr_es = panel_df.loc[panel_df["month"].isin([tr_months[-1]])].reset_index(drop=True)
            else:
                df_tr_full = panel_df.loc[panel_df["month"].isin(tr_months)].reset_index(drop=True)

            for k in range(n_folds):
                ev_slice = df_ev_step.loc[df_ev_step[fold_col].to_numpy() == k]
                X_ev = extract_feature_matrix(ev_slice, feat_cols)
                y_ev = ev_slice[target_col].to_numpy(dtype=np.int8)
                cids_ev = ev_slice["client_id"].to_numpy()

                if can_es:
                    tr_core = df_tr_core.loc[df_tr_core[fold_col].to_numpy() != k]
                    tr_es = df_tr_es.loc[df_tr_es[fold_col].to_numpy() != k]
                    model = build_model(family, seed=seed, model_params=cb_params)
                    model.fit(
                        extract_feature_matrix(tr_core, feat_cols),
                        tr_core[target_col].to_numpy(dtype=np.int8),
                        X_es=extract_feature_matrix(tr_es, feat_cols),
                        y_es=tr_es[target_col].to_numpy(dtype=np.int8),
                    )
                else:
                    tr_full = df_tr_full.loc[df_tr_full[fold_col].to_numpy() != k]
                    model = build_model(family, seed=seed)
                    model.fit(extract_feature_matrix(tr_full, feat_cols), tr_full[target_col].to_numpy(dtype=np.int8))

                base_scores = model.predict_scores(X_ev)
                eval_blocks.append({
                    "month": ev_months[0],
                    "X_ev": X_ev,
                    "y_ev": y_ev,
                    "cids_ev": cids_ev,
                    "model": model,
                    "base_scores": base_scores,
                })

        def _macro_ap_lift(score_by_block: list[np.ndarray]) -> tuple[float, float]:
            by_m: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
            for blk, sc in zip(eval_blocks, score_by_block):
                by_m.setdefault(blk["month"], []).append((blk["y_ev"], sc, blk["cids_ev"]))
            aps, lifts = [], []
            for m, parts in by_m.items():
                ym = np.concatenate([p[0] for p in parts])
                sm = np.concatenate([p[1] for p in parts])
                cm = np.concatenate([p[2] for p in parts])
                mm = compute_slice_point_metrics(ym, sm, cm, seed=tie_seed, is_probabilistic=True)
                aps.append(mm["average_precision"])
                lifts.append(mm["lift_1pct"])
            return float(np.mean(aps)), float(np.mean(lifts))

        base_ap, base_lift = _macro_ap_lift([b["base_scores"] for b in eval_blocks])
        rng = np.random.default_rng(seed + 900)
        for j, fname in enumerate(feat_cols):
            ap_drops, lift_drops = [], []
            for _ in range(n_repeats):
                perm_scores = []
                for blk in eval_blocks:
                    Xp = blk["X_ev"].copy()
                    Xp[:, j] = rng.permutation(Xp[:, j])
                    perm_scores.append(blk["model"].predict_scores(Xp))
                p_ap, p_lift = _macro_ap_lift(perm_scores)
                ap_drops.append(base_ap - p_ap)
                lift_drops.append(base_lift - p_lift)

            src = "TRX" if fname.startswith("trx_") else ("GEO" if fname.startswith("geo_") else "DIALOG")
            rows.append({
                "exp_id": exp_id,
                "feature": fname,
                "source": src,
                "base_val_macro_pr_auc": base_ap,
                "base_val_macro_lift_1pct": base_lift,
                "pr_auc_drop_mean": float(np.mean(ap_drops)),
                "pr_auc_drop_sd": float(np.std(ap_drops, ddof=1)) if n_repeats > 1 else 0.0,
                "lift_1pct_drop_mean": float(np.mean(lift_drops)),
                "lift_1pct_drop_sd": float(np.std(lift_drops, ddof=1)) if n_repeats > 1 else 0.0,
            })
        del eval_blocks
        gc.collect()

    pi_df = pd.DataFrame(rows).sort_values(["exp_id", "pr_auc_drop_mean"], ascending=[True, False]).reset_index(drop=True)
    pi_path.parent.mkdir(parents=True, exist_ok=True)
    pi_df.to_csv(pi_path, index=False)
    return pi_df


def run_val_slices(
    val_preds_seed42: dict[str, pd.DataFrame],
    out_dir: Path,
    tie_seed: int = 42,
) -> pd.DataFrame:
    """Compute segment diagnostics strictly on Val predictions."""
    slice_dims = ("trx_90d_has_event", "geo_90d_has_event", "dialog_90d_has_event", "dialog_all_has_event", "cold_start")
    rows: list[dict[str, Any]] = []
    for exp_id, pdf in val_preds_seed42.items():
        for dim in slice_dims:
            if dim not in pdf.columns:
                continue
            for val in (0, 1):
                sub = pdf[pdf[dim] == val]
                if len(sub) == 0:
                    continue
                mm = compute_slice_point_metrics(
                    sub["y_true"].to_numpy(),
                    sub["score"].to_numpy(),
                    sub["client_id"].to_numpy(),
                    seed=tie_seed,
                    is_probabilistic=False,
                )
                rows.append({"exp_id": exp_id, "slice_dim": dim, "slice_val": val, **mm})
    sdf = pd.DataFrame(rows)
    s_path = out_dir / "metrics" / "val_slice_metrics.csv"
    s_path.parent.mkdir(parents=True, exist_ok=True)
    sdf.to_csv(s_path, index=False)
    return sdf


def run_val_shuffled_sanity(
    spec: dict[str, Any],
    panel_df: pd.DataFrame,
    tuned_cb_params: dict[str, Any],
    out_dir: Path,
    thread_count: int = -1,
) -> pd.DataFrame:
    """Verify shuffled training labels collapse PR-AUC to base rate under Protocol B on Val."""
    seed = int(spec["evaluation"]["canonical_seed"])
    tie_seed = int(spec["evaluation"]["tie_break_seed"])
    feat_cols = spec["feature_sets"]["raw_TGD_14"]
    cb_params = {**tuned_cb_params, "thread_count": int(thread_count)}
    rows: list[dict[str, Any]] = []

    for exp_id, family, use_es in (
        ("B_logreg_TGD", "logreg", False),
        ("B_catboost_TGD_tuned", "catboost_tuned", True),
    ):
        pdf, _ = run_protocol_experiment(
            spec=spec,
            protocol_name="B_rolling_ooc",
            stage_name="val",
            panel_df=panel_df,
            exp_id=exp_id,
            family=family,
            feature_cols=feat_cols,
            uses_target_lag=False,
            seed=seed,
            model_params=cb_params if use_es else None,
            pred_cache_dir=out_dir / "predictions" / "_sanity_cache",
            use_temporal_es=use_es,
            shuffle_labels=True,
        )
        for m, sub in pdf.groupby("month"):
            mm = compute_slice_point_metrics(
                sub["y_true"].to_numpy(), sub["score"].to_numpy(), sub["client_id"].to_numpy(), seed=tie_seed, is_probabilistic=True
            )
            gap = abs(float(mm["average_precision"]) - float(mm["prevalence"]))
            rows.append({
                "exp_id": exp_id,
                "month": m,
                "prevalence": mm["prevalence"],
                "shuffled_pr_auc": mm["average_precision"],
                "shuffled_lift_1pct": mm["lift_1pct"],
                "abs_gap_to_prevalence": gap,
                "passed": bool(gap <= 0.005),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics" / "val_shuffled_sanity.csv", index=False)
    return df
