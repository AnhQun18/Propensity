"""End-to-end dual-horizon (H=1, H=3) baseline runner with checkpointing & verification."""
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import time
from typing import Any

import catboost
import numpy as np
import pandas as pd
import psutil
import sklearn

from .bootstrap import cluster_bootstrap_single, paired_cluster_bootstrap
from .config import BaselineConfig, ExperimentSpec, FoldSpec
from .data import load_dataset_contract, load_fold_arrays, resolve_feature_sets
from .features import build_enriched_panel
from .metrics import compute_metrics, deterministic_rank_order, safe_log_loss
from .models import (
    CatBoostBaseline,
    ColdStartFallbackWrapper,
    HeuristicRFBaseline,
    PrevalenceBaseline,
    ProbabilityCalibrator,
    RidgeLogisticBaseline,
)
from .plots import generate_all_plots
from .slices import evaluate_slices


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _rss_mb() -> float:
    return float(psutil.Process().memory_info().rss / (1024 * 1024))


def _make_model(spec: ExperimentSpec, config: BaselineConfig, seed: int):
    if spec.model_type == "prevalence":
        return PrevalenceBaseline()
    if spec.model_type == "heuristic_rf":
        return HeuristicRFBaseline()
    if spec.model_type == "ridge_logreg":
        return RidgeLogisticBaseline(c=config.logreg_c, max_iter=config.logreg_max_iter, seed=seed)
    if spec.model_type == "catboost":
        return CatBoostBaseline(params=config.catboost_params, seed=seed)
    if spec.model_type == "catboost_cold_fallback":
        return ColdStartFallbackWrapper(
            params=config.catboost_params, alpha=config.cold_start_alpha, seed=seed
        )
    raise ValueError(f"Unknown model_type: {spec.model_type}")


def _fit_and_score_fold(
    spec: ExperimentSpec,
    fold: FoldSpec,
    enriched_parquet: Path,
    feature_cols: tuple[str, ...],
    config: BaselineConfig,
    pred_path: Path,
    model_path: Path,
    baseline_logloss: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fold_data = load_fold_arrays(
        enriched_parquet=enriched_parquet,
        train_months=fold.train_months,
        eval_month=fold.eval_month,
        horizon=fold.horizon,
        feature_cols=feature_cols,
        burn_in_months=config.burn_in_months,
        memory_limit=config.duckdb_memory_limit,
        threads=config.duckdb_threads,
    )
    model = _make_model(spec, config, seed=config.seed)

    t0 = time.perf_counter()
    model.fit(fold_data.X_train, fold_data.y_train, cold_start=fold_data.train_cold_start)
    train_sec = time.perf_counter() - t0

    eval_cold = fold_data.eval_meta["cold_start"].to_numpy(dtype=np.int8)
    t1 = time.perf_counter()
    scores = model.predict_scores(fold_data.X_eval, cold_start=eval_cold)
    pred_sec = time.perf_counter() - t1

    rss = _rss_mb()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    saved_model = model.save(model_path)
    model_bytes = saved_model.stat().st_size

    # Learning curve (offline diagnostic after training, zero validation leakage)
    lc_rows: list[dict[str, Any]] = []
    if isinstance(model, (CatBoostBaseline, ColdStartFallbackWrapper)):
        if isinstance(model, CatBoostBaseline):
            tr_staged = model.staged_predictions(fold_data.X_train, config.learning_curve_checkpoints)
            ev_staged = model.staged_predictions(fold_data.X_eval, config.learning_curve_checkpoints)
        else:
            tr_staged = model.staged_predictions(
                fold_data.X_train, fold_data.train_cold_start, config.learning_curve_checkpoints
            )
            ev_staged = model.staged_predictions(
                fold_data.X_eval, eval_cold, config.learning_curve_checkpoints
            )
        for cp in sorted(ev_staged.keys()):
            ev_m, _ = compute_metrics(
                y_true=fold_data.y_eval,
                scores=ev_staged[cp],
                client_ids=fold_data.eval_meta["client_id"].to_numpy(),
                budget_fractions=(config.primary_budget_fraction,),
                primary_budget=config.primary_budget_fraction,
                seed=config.seed,
                is_probabilistic=True,
            )
            lc_rows.append(
                {
                    "horizon": fold.horizon,
                    "stage": fold.stage,
                    "experiment_id": spec.experiment_id,
                    "eval_month": fold.eval_month,
                    "ntree": int(cp),
                    "train_log_loss": safe_log_loss(fold_data.y_train, tr_staged[cp]),
                    "eval_log_loss": ev_m["log_loss"],
                    "eval_roc_auc": ev_m["roc_auc"],
                    "eval_average_precision": ev_m["average_precision"],
                    "eval_recall_1pct": ev_m["recall_1pct"],
                }
            )

    # Feature importance
    fi_rows = [
        {
            "horizon": fold.horizon,
            "stage": fold.stage,
            "experiment_id": spec.experiment_id,
            "eval_month": fold.eval_month,
            "feature": fname,
            "importance": float(imp),
        }
        for fname, imp in model.feature_importance(feature_cols)
    ]

    # Save fold predictions immediately as checkpoint
    pred_df = fold_data.eval_meta.copy()
    pred_df["y_true"] = fold_data.y_eval.astype(np.int8)
    pred_df["score"] = scores.astype(np.float64)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(pred_path, index=False, compression="zstd")

    metrics, _ = compute_metrics(
        y_true=fold_data.y_eval,
        scores=scores,
        client_ids=pred_df["client_id"].to_numpy(),
        budget_fractions=config.budget_fractions,
        primary_budget=config.primary_budget_fraction,
        seed=config.seed,
        is_probabilistic=model.is_probabilistic,
        baseline_logloss=baseline_logloss,
    )
    metric_row = {
        "horizon": fold.horizon,
        "stage": fold.stage,
        "fold_name": fold.name,
        "experiment_id": spec.experiment_id,
        "model_type": spec.model_type,
        "feature_set": spec.feature_set,
        "n_features": len(feature_cols),
        "train_months": ",".join(fold.train_months),
        "n_train_rows": int(len(fold_data.y_train)),
        "eval_month": fold.eval_month,
        **metrics,
    }
    op_row = {
        "horizon": fold.horizon,
        "stage": fold.stage,
        "experiment_id": spec.experiment_id,
        "eval_month": fold.eval_month,
        "n_features": len(feature_cols),
        "n_train_rows": int(len(fold_data.y_train)),
        "n_eval_rows": int(len(fold_data.y_eval)),
        "train_seconds": round(train_sec, 3),
        "predict_seconds": round(pred_sec, 3),
        "model_bytes": int(model_bytes),
        "rss_mb": round(rss, 1),
    }

    del fold_data, model, pred_df, scores
    gc.collect()
    return metric_row, lc_rows, fi_rows, op_row


def _select_winner(bootstrap_df: pd.DataFrame, experiments: tuple[ExperimentSpec, ...], horizon: int) -> str:
    eligible_ids = {
        e.experiment_id
        for e in experiments
        if e.model_type in {"catboost", "catboost_cold_fallback", "ridge_logreg"}
    }
    sub = bootstrap_df[
        (bootstrap_df["horizon"] == horizon)
        & (bootstrap_df["scope"] == "validation_pooled")
        & (bootstrap_df["experiment_id"].isin(eligible_ids))
    ].copy()
    if len(sub) == 0:
        return experiments[-1].experiment_id
    sub = sub.sort_values(["lift_1pct", "average_precision", "roc_auc"], ascending=[False, False, False])
    return str(sub.iloc[0]["experiment_id"])


def run_baseline(config: BaselineConfig, keep_enriched_panel: bool = False) -> Path:
    out_dir = config.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)

    contract = load_dataset_contract(config.processed_dir)
    feature_sets = resolve_feature_sets(contract.schema)

    manifest: dict[str, Any] = {
        "status": "running",
        "run_name": config.run_name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": contract.manifest.get("dataset_sha256"),
        "source_processed_dir": str(config.processed_dir),
        "versions": {
            "catboost": catboost.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "config": config.to_dict(),
        "feature_counts": {k: len(v) for k, v in feature_sets.items()},
    }
    _write_json(out_dir / "manifest.json", manifest)

    t_start = time.perf_counter()
    peak_rss = _rss_mb()

    try:
        # 1. Build point-in-time enriched panel (H=1, H=3, 17 prior_purchase, 14 derived_behavioral)
        enriched_path = out_dir / "enriched_panel.parquet"
        print(f"[{config.run_name}] Building enriched panel from {contract.parquet_path} ...", flush=True)
        enrich_info = build_enriched_panel(
            processed_parquet=contract.parquet_path,
            destination=enriched_path,
            burn_in_months=config.burn_in_months,
            memory_limit=config.duckdb_memory_limit,
            threads=config.duckdb_threads,
        )
        manifest["enriched_summary"] = enrich_info
        _write_json(out_dir / "manifest.json", manifest)
        print(f"[{config.run_name}] Enriched panel ready: {enrich_info['total_rows']:,} rows", flush=True)

        monthly_rows: list[dict[str, Any]] = []
        lc_rows: list[dict[str, Any]] = []
        fi_rows: list[dict[str, Any]] = []
        op_rows: list[dict[str, Any]] = []

        # 2. Run Rolling Validation for H=1 and H=3
        for horizon, folds in ((1, config.h1_validation_folds), (3, config.h3_validation_folds)):
            prev_logloss_by_month: dict[str, float] = {}
            for spec in config.experiments:
                cols = feature_sets[spec.feature_set]
                for fold in folds:
                    pred_path = out_dir / "predictions" / f"h{horizon}_val" / spec.experiment_id / f"fold_{fold.eval_month}.parquet"
                    model_path = out_dir / "models" / f"h{horizon}_val" / spec.experiment_id / f"fold_{fold.eval_month}"
                    base_ll = prev_logloss_by_month.get(fold.eval_month)
                    m_row, lc, fi, op = _fit_and_score_fold(
                        spec=spec,
                        fold=fold,
                        enriched_parquet=enriched_path,
                        feature_cols=cols,
                        config=config,
                        pred_path=pred_path,
                        model_path=model_path,
                        baseline_logloss=base_ll,
                    )
                    if spec.model_type == "prevalence" and m_row.get("log_loss") is not None:
                        prev_logloss_by_month[fold.eval_month] = float(m_row["log_loss"])
                        m_row["log_loss_skill"] = 0.0
                    monthly_rows.append(m_row)
                    lc_rows.extend(lc)
                    fi_rows.extend(fi)
                    op_rows.append(op)
                    peak_rss = max(peak_rss, op["rss_mb"])
                    print(
                        f"  [H={horizon} VAL] {spec.experiment_id:26s} | {fold.eval_month} "
                        f"| Lift@1%={m_row['lift_1pct']:.2f}x | Rec@1%={m_row['recall_1pct']*100:.2f}% "
                        f"| AP={m_row['average_precision']:.4f} | AUC={m_row['roc_auc']:.4f} "
                        f"| {op['train_seconds']:.1f}s | RSS={op['rss_mb']:.0f}MB",
                        flush=True,
                    )

        # 3. Validation Pooled Bootstrap, Paired Comparisons, Calibration Bins & Slices
        bootstrap_rows: list[dict[str, Any]] = []
        paired_rows: list[dict[str, Any]] = []
        slice_rows: list[dict[str, Any]] = []
        cal_rows: list[dict[str, Any]] = []

        exp_by_id = {e.experiment_id: e for e in config.experiments}
        is_prob_map = {e.experiment_id: (e.model_type != "heuristic_rf") for e in config.experiments}

        for horizon in (1, 3):
            pooled_cache: dict[str, pd.DataFrame] = {}
            for spec in config.experiments:
                exp_dir = out_dir / "predictions" / f"h{horizon}_val" / spec.experiment_id
                files = sorted(exp_dir.glob("fold_*.parquet"))
                df_p = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
                pooled_cache[spec.experiment_id] = df_p
                is_prob = is_prob_map[spec.experiment_id]

                bs = cluster_bootstrap_single(
                    df=df_p,
                    is_probabilistic=is_prob,
                    primary_budget=config.primary_budget_fraction,
                    n_replicates=config.bootstrap_replicates,
                    seed=config.seed,
                )
                bootstrap_rows.append(
                    {
                        "horizon": horizon,
                        "scope": "validation_pooled",
                        "experiment_id": spec.experiment_id,
                        "n_features": len(feature_sets[spec.feature_set]),
                        **bs,
                    }
                )

                # Calibration bins on pooled validation
                if is_prob:
                    _, c_df = compute_metrics(
                        y_true=df_p["y_true"].to_numpy(dtype=np.int8),
                        scores=df_p["score"].to_numpy(dtype=np.float64),
                        client_ids=df_p["client_id"].to_numpy(),
                        budget_fractions=config.budget_fractions,
                        primary_budget=config.primary_budget_fraction,
                        seed=config.seed,
                        is_probabilistic=True,
                    )
                    c_df["horizon"] = horizon
                    c_df["scope"] = "validation_pooled"
                    c_df["experiment_id"] = spec.experiment_id
                    cal_rows.extend(c_df.to_dict("records"))

                # Slices on pooled validation
                s_recs = evaluate_slices(
                    df=df_p,
                    budget_fractions=config.budget_fractions,
                    primary_budget=config.primary_budget_fraction,
                    seed=config.seed,
                    is_probabilistic=is_prob,
                )
                for r in s_recs:
                    slice_rows.append(
                        {"horizon": horizon, "scope": "validation_pooled", "experiment_id": spec.experiment_id, **r}
                    )

            # Paired cluster bootstrap along configured ablation pairs
            for model_a, model_b in config.paired_comparisons:
                if model_a in pooled_cache and model_b in pooled_cache:
                    pb = paired_cluster_bootstrap(
                        df_a=pooled_cache[model_a],
                        df_b=pooled_cache[model_b],
                        is_prob_a=is_prob_map[model_a],
                        is_prob_b=is_prob_map[model_b],
                        primary_budget=config.primary_budget_fraction,
                        n_replicates=config.bootstrap_replicates,
                        seed=config.seed,
                    )
                    paired_rows.append(
                        {
                            "horizon": horizon,
                            "scope": "validation_pooled",
                            "model_a": model_a,
                            "model_b": model_b,
                            **pb,
                        }
                    )

        bootstrap_df = pd.DataFrame(bootstrap_rows)
        winners = {
            "h1": _select_winner(bootstrap_df, config.experiments, horizon=1),
            "h3": _select_winner(bootstrap_df, config.experiments, horizon=3),
        }
        print(f"[{config.run_name}] Selected validation winners: {winners}", flush=True)

        # 4. Final Holdout Evaluation (H=1 with 2022-12 calibration -> 2023-01 test; H=3 monthly vs quarterly -> 2022-11 test)
        final_rows: list[dict[str, Any]] = []
        calibration_params: dict[str, Any] = {}

        # --- H=1 Final ---
        h1_win_spec = exp_by_id[winners["h1"]]
        h1_win_cols = feature_sets[h1_win_spec.feature_set]

        # Prevalence & Heuristic baselines on H=1 final test
        h1_prev_ll = None
        for b_id in ("B0_prevalence", "B1_heuristic_rf"):
            if b_id not in exp_by_id:
                continue
            b_spec = exp_by_id[b_id]
            fold_t = FoldSpec(
                name=f"h1_final_{b_id}",
                train_months=config.h1_final_train_months,
                eval_month=config.h1_test_month,
                horizon=1,
                stage="final_test",
            )
            m_row, _, _, op = _fit_and_score_fold(
                spec=b_spec,
                fold=fold_t,
                enriched_parquet=enriched_path,
                feature_cols=feature_sets[b_spec.feature_set],
                config=config,
                pred_path=out_dir / "predictions" / "h1_final" / f"{b_id}_{config.h1_test_month}.parquet",
                model_path=out_dir / "models" / "h1_final" / b_id,
                baseline_logloss=h1_prev_ll,
            )
            if b_spec.model_type == "prevalence" and m_row.get("log_loss") is not None:
                h1_prev_ll = float(m_row["log_loss"])
                m_row["log_loss_skill"] = 0.0
            final_rows.append(m_row)
            op_rows.append(op)

        # Fit single H=1 final base model on 05..11 (NEVER refit after calibration!)
        cal_data = load_fold_arrays(
            enriched_parquet=enriched_path,
            train_months=config.h1_final_train_months,
            eval_month=config.h1_calibration_month,
            horizon=1,
            feature_cols=h1_win_cols,
            burn_in_months=config.burn_in_months,
            memory_limit=config.duckdb_memory_limit,
            threads=config.duckdb_threads,
        )
        h1_base = _make_model(h1_win_spec, config, seed=config.seed)
        t0 = time.perf_counter()
        h1_base.fit(cal_data.X_train, cal_data.y_train, cold_start=cal_data.train_cold_start)
        h1_train_sec = time.perf_counter() - t0
        cal_scores = h1_base.predict_scores(
            cal_data.X_eval, cold_start=cal_data.eval_meta["cold_start"].to_numpy(dtype=np.int8)
        )
        cal_y = cal_data.y_eval.copy()

        # Save calibration month predictions
        cal_pred_df = cal_data.eval_meta.copy()
        cal_pred_df["y_true"] = cal_y
        cal_pred_df["score"] = cal_scores
        (out_dir / "predictions" / "h1_final").mkdir(parents=True, exist_ok=True)
        cal_pred_df.to_parquet(
            out_dir / "predictions" / "h1_final" / f"H1_cal_month_{config.h1_calibration_month}.parquet",
            index=False,
            compression="zstd",
        )
        h1_saved = h1_base.save(out_dir / "models" / "h1_final" / f"winner_{h1_win_spec.experiment_id}")
        n_h1_train = len(cal_data.y_train)
        del cal_data
        gc.collect()

        # Fit calibrators strictly on 2022-12
        cal_intercept = ProbabilityCalibrator("intercept_shift").fit(cal_scores, cal_y)
        cal_platt = ProbabilityCalibrator("platt").fit(cal_scores, cal_y)
        calibration_params["h1_2022_12"] = {
            "base_experiment": h1_win_spec.experiment_id,
            "intercept_shift": {"slope": cal_intercept.slope_, "intercept": cal_intercept.intercept_},
            "platt": {"slope": cal_platt.slope_, "intercept": cal_platt.intercept_},
            "cal_month_obs_rate": float(np.mean(cal_y)),
            "cal_month_raw_mean_pred": float(np.mean(cal_scores)),
        }

        # Score locked test month 2023-01 with same frozen base model
        test_h1_data = load_fold_arrays(
            enriched_parquet=enriched_path,
            train_months=(config.h1_calibration_month,),  # only used for embargo check; we don't retrain!
            eval_month=config.h1_test_month,
            horizon=1,
            feature_cols=h1_win_cols,
            burn_in_months=config.burn_in_months,
            memory_limit=config.duckdb_memory_limit,
            threads=config.duckdb_threads,
        )
        t1 = time.perf_counter()
        raw_test_scores = h1_base.predict_scores(
            test_h1_data.X_eval,
            cold_start=test_h1_data.eval_meta["cold_start"].to_numpy(dtype=np.int8),
        )
        h1_pred_sec = time.perf_counter() - t1
        h1_test_variants = {
            "H1_final_raw": raw_test_scores,
            "H1_final_cal_intercept": cal_intercept.transform(raw_test_scores),
            "H1_final_cal_platt": cal_platt.transform(raw_test_scores),
        }
        op_rows.append(
            {
                "horizon": 1,
                "stage": "final_test",
                "experiment_id": "H1_final_raw",
                "eval_month": config.h1_test_month,
                "n_features": len(h1_win_cols),
                "n_train_rows": int(n_h1_train),
                "n_eval_rows": int(len(test_h1_data.y_eval)),
                "train_seconds": round(h1_train_sec, 3),
                "predict_seconds": round(h1_pred_sec, 3),
                "model_bytes": int(h1_saved.stat().st_size),
                "rss_mb": round(_rss_mb(), 1),
            }
        )

        h1_final_dfs: dict[str, pd.DataFrame] = {}
        for var_name, v_scores in h1_test_variants.items():
            pdf = test_h1_data.eval_meta.copy()
            pdf["y_true"] = test_h1_data.y_eval.astype(np.int8)
            pdf["score"] = v_scores.astype(np.float64)
            pdf.to_parquet(
                out_dir / "predictions" / "h1_final" / f"{var_name}_{config.h1_test_month}.parquet",
                index=False,
                compression="zstd",
            )
            h1_final_dfs[var_name] = pdf
            m, c_df = compute_metrics(
                y_true=test_h1_data.y_eval,
                scores=v_scores,
                client_ids=pdf["client_id"].to_numpy(),
                budget_fractions=config.budget_fractions,
                primary_budget=config.primary_budget_fraction,
                seed=config.seed,
                is_probabilistic=True,
                baseline_logloss=h1_prev_ll,
            )
            final_rows.append(
                {
                    "horizon": 1,
                    "stage": "final_test",
                    "fold_name": f"h1_test_{config.h1_test_month}",
                    "experiment_id": var_name,
                    "model_type": h1_win_spec.model_type,
                    "feature_set": h1_win_spec.feature_set,
                    "n_features": len(h1_win_cols),
                    "train_months": ",".join(config.h1_final_train_months),
                    "n_train_rows": int(n_h1_train),
                    "eval_month": config.h1_test_month,
                    **m,
                }
            )
            c_df["horizon"] = 1
            c_df["scope"] = "final_test"
            c_df["experiment_id"] = var_name
            cal_rows.extend(c_df.to_dict("records"))

            bs = cluster_bootstrap_single(
                df=pdf,
                is_probabilistic=True,
                primary_budget=config.primary_budget_fraction,
                n_replicates=config.bootstrap_replicates,
                seed=config.seed,
            )
            bootstrap_rows.append(
                {"horizon": 1, "scope": "final_test", "experiment_id": var_name, "n_features": len(h1_win_cols), **bs}
            )

        # Paired bootstrap for calibration effect on 2023-01 Log Loss
        for cal_var in ("H1_final_cal_intercept", "H1_final_cal_platt"):
            pb = paired_cluster_bootstrap(
                df_a=h1_final_dfs["H1_final_raw"],
                df_b=h1_final_dfs[cal_var],
                is_prob_a=True,
                is_prob_b=True,
                primary_budget=config.primary_budget_fraction,
                n_replicates=config.bootstrap_replicates,
                seed=config.seed,
            )
            paired_rows.append({"horizon": 1, "scope": "final_test", "model_a": "H1_final_raw", "model_b": cal_var, **pb})

        del test_h1_data, h1_base
        gc.collect()

        # --- H=3 Final (2022-11 test: Monthly sliding vs Quarterly non-overlapping) ---
        h3_win_spec = exp_by_id[winners["h3"]]
        h3_win_cols = feature_sets[h3_win_spec.feature_set]
        h3_prev_ll = None
        for b_id in ("B0_prevalence", "B1_heuristic_rf"):
            if b_id not in exp_by_id:
                continue
            b_spec = exp_by_id[b_id]
            fold_t = FoldSpec(
                name=f"h3_final_{b_id}",
                train_months=config.h3_final_monthly_train_months,
                eval_month=config.h3_test_month,
                horizon=3,
                stage="final_test",
            )
            m_row, _, _, op = _fit_and_score_fold(
                spec=b_spec,
                fold=fold_t,
                enriched_parquet=enriched_path,
                feature_cols=feature_sets[b_spec.feature_set],
                config=config,
                pred_path=out_dir / "predictions" / "h3_final" / f"{b_id}_{config.h3_test_month}.parquet",
                model_path=out_dir / "models" / "h3_final" / b_id,
                baseline_logloss=h3_prev_ll,
            )
            if b_spec.model_type == "prevalence" and m_row.get("log_loss") is not None:
                h3_prev_ll = float(m_row["log_loss"])
                m_row["log_loss_skill"] = 0.0
            final_rows.append(m_row)
            op_rows.append(op)

        h3_final_dfs: dict[str, pd.DataFrame] = {}
        for var_name, t_months in (
            ("H3_final_monthly", config.h3_final_monthly_train_months),
            ("H3_final_quarterly", config.h3_final_quarterly_train_months),
        ):
            v_spec = ExperimentSpec(
                experiment_id=var_name,
                model_type=h3_win_spec.model_type,
                feature_set=h3_win_spec.feature_set,
                stage="final_test",
                description=f"H=3 final ({var_name}) using winner {h3_win_spec.experiment_id}",
            )
            fold_t = FoldSpec(
                name=var_name,
                train_months=t_months,
                eval_month=config.h3_test_month,
                horizon=3,
                stage="final_test",
            )
            p_path = out_dir / "predictions" / "h3_final" / f"{var_name}_{config.h3_test_month}.parquet"
            m_row, _, _, op = _fit_and_score_fold(
                spec=v_spec,
                fold=fold_t,
                enriched_parquet=enriched_path,
                feature_cols=h3_win_cols,
                config=config,
                pred_path=p_path,
                model_path=out_dir / "models" / "h3_final" / var_name,
                baseline_logloss=h3_prev_ll,
            )
            final_rows.append(m_row)
            op_rows.append(op)
            pdf = pd.read_parquet(p_path)
            h3_final_dfs[var_name] = pdf
            bs = cluster_bootstrap_single(
                df=pdf,
                is_probabilistic=True,
                primary_budget=config.primary_budget_fraction,
                n_replicates=config.bootstrap_replicates,
                seed=config.seed,
            )
            bootstrap_rows.append(
                {"horizon": 3, "scope": "final_test", "experiment_id": var_name, "n_features": len(h3_win_cols), **bs}
            )
            _, c_df = compute_metrics(
                y_true=pdf["y_true"].to_numpy(dtype=np.int8),
                scores=pdf["score"].to_numpy(dtype=np.float64),
                client_ids=pdf["client_id"].to_numpy(),
                budget_fractions=config.budget_fractions,
                primary_budget=config.primary_budget_fraction,
                seed=config.seed,
                is_probabilistic=True,
                baseline_logloss=h3_prev_ll,
            )
            c_df["horizon"] = 3
            c_df["scope"] = "final_test"
            c_df["experiment_id"] = var_name
            cal_rows.extend(c_df.to_dict("records"))

        # Paired comparison: H3_final_monthly vs H3_final_quarterly on 2022-11
        pb_h3 = paired_cluster_bootstrap(
            df_a=h3_final_dfs["H3_final_quarterly"],
            df_b=h3_final_dfs["H3_final_monthly"],
            is_prob_a=True,
            is_prob_b=True,
            primary_budget=config.primary_budget_fraction,
            n_replicates=config.bootstrap_replicates,
            seed=config.seed,
        )
        paired_rows.append(
            {
                "horizon": 3,
                "scope": "final_test",
                "model_a": "H3_final_quarterly",
                "model_b": "H3_final_monthly",
                **pb_h3,
            }
        )

        # Common snapshot 2022-11 comparison (H=1 Fold 3 winner vs H=3 Final monthly winner)
        common_2022_11: dict[str, Any] = {}
        h1_nov_path = out_dir / "predictions" / "h1_val" / winners["h1"] / f"fold_{config.h3_test_month}.parquet"
        if h1_nov_path.exists() and "H3_final_monthly" in h3_final_dfs:
            df_h1_nov = pd.read_parquet(h1_nov_path)
            df_h3_nov = h3_final_dfs["H3_final_monthly"]
            merged = df_h1_nov[["client_id", "y_true", "score"]].merge(
                df_h3_nov[["client_id", "y_true", "score"]],
                on="client_id",
                suffixes=("_h1", "_h3"),
                validate="one_to_one",
            )
            n = len(merged)
            k = max(1, int(math.ceil(config.primary_budget_fraction * n)))
            ord_h1 = deterministic_rank_order(merged["score_h1"].to_numpy(), merged["client_id"].to_numpy(), config.seed)
            ord_h3 = deterministic_rank_order(merged["score_h3"].to_numpy(), merged["client_id"].to_numpy(), config.seed)
            top_h1 = set(merged["client_id"].iloc[ord_h1[:k]])
            top_h3 = set(merged["client_id"].iloc[ord_h3[:k]])
            overlap = len(top_h1 & top_h3)
            corr = float(merged["score_h1"].corr(merged["score_h3"], method="spearman"))
            # Cross-horizon capture: how well does H=1 top-1% capture H=3 buyers, and vice versa?
            m_h1_on_h3, _ = compute_metrics(
                y_true=merged["y_true_h3"].to_numpy(dtype=np.int8),
                scores=merged["score_h1"].to_numpy(dtype=np.float64),
                client_ids=merged["client_id"].to_numpy(),
                budget_fractions=(config.primary_budget_fraction,),
                primary_budget=config.primary_budget_fraction,
                seed=config.seed,
                is_probabilistic=False,
            )
            m_h3_on_h1, _ = compute_metrics(
                y_true=merged["y_true_h1"].to_numpy(dtype=np.int8),
                scores=merged["score_h3"].to_numpy(dtype=np.float64),
                client_ids=merged["client_id"].to_numpy(),
                budget_fractions=(config.primary_budget_fraction,),
                primary_budget=config.primary_budget_fraction,
                seed=config.seed,
                is_probabilistic=False,
            )
            common_2022_11 = {
                "month": config.h3_test_month,
                "n_clients": int(n),
                "k_1pct": int(k),
                "h1_winner": winners["h1"],
                "h3_winner": winners["h3"],
                "spearman_score_corr": corr,
                "top1pct_overlap_count": int(overlap),
                "top1pct_overlap_share": float(overlap / k),
                "h1_model_evaluated_on_h3_target": {
                    "lift_1pct": m_h1_on_h3["lift_1pct"],
                    "recall_1pct": m_h1_on_h3["recall_1pct"],
                    "average_precision": m_h1_on_h3["average_precision"],
                    "roc_auc": m_h1_on_h3["roc_auc"],
                },
                "h3_model_evaluated_on_h1_target": {
                    "lift_1pct": m_h3_on_h1["lift_1pct"],
                    "recall_1pct": m_h3_on_h1["recall_1pct"],
                    "average_precision": m_h3_on_h1["average_precision"],
                    "roc_auc": m_h3_on_h1["roc_auc"],
                },
            }

        # 5. Export all metric tables
        metrics_dir = out_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        monthly_df = pd.DataFrame(monthly_rows)
        bootstrap_df = pd.DataFrame(bootstrap_rows)
        paired_df = pd.DataFrame(paired_rows)
        slice_df = pd.DataFrame(slice_rows)
        cal_df = pd.DataFrame(cal_rows)
        lc_df = pd.DataFrame(lc_rows)
        fi_df = pd.DataFrame(fi_rows)
        op_df = pd.DataFrame(op_rows)
        final_df = pd.DataFrame(final_rows)

        monthly_df.to_csv(metrics_dir / "monthly_metrics.csv", index=False)
        bootstrap_df.to_csv(metrics_dir / "bootstrap_summary.csv", index=False)
        paired_df.to_csv(metrics_dir / "paired_bootstrap.csv", index=False)
        slice_df.to_csv(metrics_dir / "slice_metrics.csv", index=False)
        cal_df.to_csv(metrics_dir / "calibration_bins.csv", index=False)
        lc_df.to_csv(metrics_dir / "learning_curves.csv", index=False)
        fi_df.to_csv(metrics_dir / "feature_importance.csv", index=False)
        op_df.to_csv(metrics_dir / "operational_stats.csv", index=False)
        final_df.to_csv(metrics_dir / "final_holdout_metrics.csv", index=False)
        _write_json(metrics_dir / "calibration_params.json", calibration_params)
        _write_json(metrics_dir / "common_2022_11_comparison.json", common_2022_11)

        # 6. Generate publication figures
        exp_order = [e.experiment_id for e in config.experiments]
        figures = generate_all_plots(
            run_dir=out_dir,
            bootstrap_df=bootstrap_df,
            paired_df=paired_df,
            monthly_df=monthly_df,
            slice_df=slice_df,
            cal_df=cal_df,
            lc_df=lc_df,
            fi_df=fi_df,
            final_df=final_df,
            winners=winners,
            exp_order=exp_order,
        )

        # Clean up temporary enriched_panel.parquet unless requested
        if not keep_enriched_panel and enriched_path.exists():
            enriched_path.unlink()

        elapsed = time.perf_counter() - t_start
        manifest.update(
            status="ready",
            winners=winners,
            calibration_params=calibration_params,
            common_2022_11_comparison=common_2022_11,
            figures=[Path(f).name for f in figures],
            elapsed_seconds=round(elapsed, 2),
            peak_rss_mb=round(peak_rss, 1),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        _write_json(out_dir / "manifest.json", manifest)
        print(f"[{config.run_name}] DONE in {elapsed:.1f}s (peak RSS {peak_rss:.0f} MB) -> {out_dir}", flush=True)
        return out_dir / "manifest.json"
    except BaseException as exc:
        manifest.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            failed_at=datetime.now(timezone.utc).isoformat(),
        )
        _write_json(out_dir / "manifest.json", manifest)
        raise
