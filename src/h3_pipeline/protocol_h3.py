"""Strict zero-leakage H=3 protocol engine, panel assembly, and evaluation."""
from __future__ import annotations

import gc
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baseline_v2.metrics import (
    bootstrap_multiseed_macro_monthly,
    bootstrap_multiseed_per_month,
    compute_slice_point_metrics,
    deterministic_tie_hash,
)
from propensity_model.config import month_diff

from .models_gpu import fit_predict_h3_model
from .progress import ModelProgressTracker

OOT_QUARANTINED_MONTHS: frozenset[str] = frozenset({"2022-09", "2022-10", "2022-11", "2022-12", "2023-01"})
ACTIVE_H3_MONTHS: tuple[str, ...] = ("2022-02", "2022-03", "2022-04", "2022-05", "2022-06", "2022-07", "2022-08", "2022-11")
META_COLS: tuple[str, ...] = ("client_id", "fold", "month", "y_any", "y_h3")


def load_h3_spec(spec_path: Path) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    validate_h3_schedule(spec)
    return spec


def validate_h3_schedule(spec: dict[str, Any]) -> None:
    """Enforce zero OOT leakage across all validation and test schedules."""
    quarantine = set(spec.get("quarantined_months", OOT_QUARANTINED_MONTHS))
    horizon = int(spec["horizon"])
    assert horizon == 3, f"Expected horizon=3, got {horizon}"

    for entry in spec["val_schedule"]:
        ev = str(entry["eval_month"])
        tr_ms = [str(m) for m in entry["train_months"]]
        if ev in quarantine:
            raise AssertionError(f"OOT LEAKAGE: val eval_month {ev} overlaps OOT/quarantine {sorted(quarantine)}")
        for tm in tr_ms:
            if tm in quarantine:
                raise AssertionError(f"OOT LEAKAGE: val train_month {tm} in quarantine {sorted(quarantine)}")
            gap = month_diff(ev, tm)
            if gap < horizon:
                raise AssertionError(f"EMBARGO LEAKAGE: val {tm} -> {ev} gap={gap} < H={horizon}")

    for sched_key in ("oot_test_schedule", "non_overlapping_step3_oot_schedule"):
        if sched_key not in spec:
            continue
        for entry in spec[sched_key]:
            ev = str(entry["eval_month"])
            tr_ms = [str(m) for m in entry["train_months"]]
            for tm in tr_ms:
                if tm in quarantine:
                    raise AssertionError(f"OOT LEAKAGE: {sched_key} train_month {tm} in quarantine {sorted(quarantine)}")
                gap = month_diff(ev, tm)
                if gap < horizon:
                    raise AssertionError(f"EMBARGO LEAKAGE: {sched_key} {tm} -> {ev} gap={gap} < H={horizon}")


def assert_fold_isolation(train_cids: np.ndarray, eval_cids: np.ndarray, context: str = "") -> None:
    """Guarantee zero client overlap between train/ES and eval slices."""
    overlap = set(train_cids).intersection(set(eval_cids))
    if overlap:
        raise AssertionError(f"CLIENT OVERLAP ({len(overlap)} clients) in {context}")


def assert_prior_null_at_cold_month(panel: pd.DataFrame) -> None:
    """Verify 2022-02 has prior_months_observed == 0 and NaN prior buy features."""
    m02 = panel[panel["month"] == "2022-02"]
    if len(m02) == 0:
        return
    if "prior_months_observed" in m02.columns:
        assert int(m02["prior_months_observed"].fillna(0).max()) == 0, "2022-02 must have 0 observed prior months"
    for c in ("prior_1m_bought_any", "prior_3m_buy_months", "prior_all_bought_any"):
        if c in m02.columns:
            assert m02[c].isna().all(), f"2022-02 {c} must be NaN (unobserved), not 0"


def ensure_stage1_delta_extracted(delta_zip: Path, work_dir: Path) -> Path:
    """Extract h3_stage1_delta.zip into work_dir/outputs/h3_propensity/delta/ if needed."""
    delta_dir = work_dir / "outputs" / "h3_propensity" / "delta"
    flag = delta_dir / "_STAGE1_READY.flag"
    if flag.exists() and (delta_dir / "h3_targets_prior.parquet").exists():
        return delta_dir
    delta_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(delta_zip, "r") as zf:
        zf.extractall(delta_dir)
    flag.write_text("ok", encoding="utf-8")
    return delta_dir


def assemble_h3_stage1_panel(work_dir: Path, extra_cols: list[str] | None = None) -> pd.DataFrame:
    """Build unified 8-month H=3 panel (02..08, 11) with 112 behavioral cols + y_h3 + prior_*."""
    delta_dir = work_dir / "outputs" / "h3_propensity" / "delta"
    tgt_p = delta_dir / "h3_targets_prior.parquet"
    early_p = delta_dir / "h3_early_02_03_base112.parquet"
    moscow_p = work_dir / "outputs" / "baseline_moscow_v1" / "panel_moscow.parquet"

    tgt = pd.read_parquet(tgt_p)
    tgt["month"] = tgt["month"].astype(str)
    assert set(tgt["month"].unique()).issubset(set(ACTIVE_H3_MONTHS))

    early = pd.read_parquet(early_p)
    early["month"] = early["month"].astype(str)
    beh_cols = [c for c in early.columns if c not in ("client_id", "fold", "month")]

    moscow = pd.read_parquet(
        moscow_p,
        columns=["client_id", "month"] + beh_cols,
        filters=[("month", "in", ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08", "2022-11"])],
    )
    moscow["month"] = moscow["month"].astype(str)

    beh_all = pd.concat([early.drop(columns=["fold"], errors="ignore"), moscow], ignore_index=True)
    panel = tgt.merge(beh_all, on=["client_id", "month"], how="inner", validate="one_to_one")
    assert len(panel) == len(tgt), f"Row mismatch after behavioral merge: {len(panel)} vs {len(tgt)}"

    if extra_cols:
        # Optionally load extra columns from Stage 2 or aux files later if requested
        pass

    # Downcast float64 feature columns to float32 to minimize RAM
    skip = set(META_COLS) | {"cutoff_msk", "target_1", "target_2", "target_3", "target_4"}
    for c in panel.columns:
        if c not in skip and panel[c].dtype == np.float64:
            panel[c] = panel[c].astype(np.float32)

    assert_prior_null_at_cold_month(panel)
    panel = panel.sort_values(["month", "client_id"]).reset_index(drop=True)
    del beh_all, early, moscow, tgt
    gc.collect()
    return panel


def run_h3_rolling_experiment(
    spec: dict[str, Any],
    panel: pd.DataFrame,
    schedule_key: str,
    exp_id: str,
    family: str,
    feature_cols: list[str],
    seed: int = 42,
    params_override: dict[str, Any] | None = None,
    pred_cache_dir: Path | None = None,
    shuffle_labels: bool = False,
    use_gpu: bool | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run one (schedule_key, exp_id, seed) across all schedule steps x 5 OOC folds."""
    validate_h3_schedule(spec)
    steps = spec[schedule_key]
    tag = f"{schedule_key}_{exp_id}{'_shuf' if shuffle_labels else ''}_seed{seed}"
    cache_path = (pred_cache_dir / f"{tag}.parquet") if pred_cache_dir else None
    if cache_path is not None and cache_path.exists():
        pdf = pd.read_parquet(cache_path)
        return pdf, {"cached": True, "elapsed_sec": 0.0, "best_iterations": []}

    target_col = str(spec["target_col"])
    fold_col = str(spec["fold_col"])
    n_folds = int(spec["n_folds"])
    max_sec = float(spec.get("max_seconds_per_model", 1800))

    tracker = ModelProgressTracker(
        stage_name=schedule_key,
        exp_id=f"{exp_id}_s{seed}",
        total_units=len(steps) * n_folds,
        max_seconds=max_sec,
    )

    step_dfs: list[pd.DataFrame] = []
    best_iters: list[int] = []
    is_gbdt = family in ("lgbm", "xgb_gpu", "catboost_gpu")

    for step in steps:
        ev_m = str(step["eval_month"])
        tr_ms = [str(m) for m in step["train_months"]]
        df_tr = panel[panel["month"].isin(tr_ms)].reset_index(drop=True)
        df_ev = panel[panel["month"] == ev_m].reset_index(drop=True)

        tr_folds_arr = df_tr[fold_col].to_numpy()
        tr_months_arr = df_tr["month"].to_numpy()
        ev_folds_arr = df_ev[fold_col].to_numpy()

        for k in range(n_folds):
            tr_mask = tr_folds_arr != k
            ev_mask = ev_folds_arr == k
            sl_tr = df_tr.loc[tr_mask]
            sl_ev = df_ev.loc[ev_mask]

            assert_fold_isolation(
                sl_tr["client_id"].to_numpy(),
                sl_ev["client_id"].to_numpy(),
                context=f"{exp_id} {ev_m} fold={k}",
            )

            X_tr = sl_tr[feature_cols].to_numpy(dtype=np.float32, copy=True)
            y_tr = sl_tr[target_col].to_numpy(dtype=np.int8, copy=True)
            X_ev = sl_ev[feature_cols].to_numpy(dtype=np.float32, copy=True)
            y_ev = sl_ev[target_col].to_numpy(dtype=np.int8, copy=True)

            if shuffle_labels:
                rng = np.random.default_rng(9100 + seed * 100 + k)
                y_tr = rng.permutation(y_tr)

            es_mask = None
            if is_gbdt:
                es_fold = (k + 1) % n_folds
                if len(tr_ms) >= 2:
                    es_months = set(tr_ms[-2:])
                    es_mask = (sl_tr[fold_col].to_numpy() == es_fold) & np.isin(sl_tr["month"].to_numpy(), list(es_months))
                else:
                    es_mask = sl_tr[fold_col].to_numpy() == es_fold

            sc, best_n = fit_predict_h3_model(
                family=family,
                X_tr=X_tr,
                y_tr=y_tr,
                X_ev=X_ev,
                feature_names=feature_cols,
                seed=seed,
                es_mask=es_mask,
                params_override=params_override,
                use_gpu=use_gpu,
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
            del X_tr, X_ev, y_tr
            tracker.step_done(f"{ev_m} f{k}", extra=f"best_n={best_n}" if best_n is not None else "")

    total_sec = tracker.finish(f"median_best_n={int(np.median(best_iters))}" if best_iters else "")
    pdf = pd.concat(step_dfs, ignore_index=True).sort_values(["month", "client_id"]).reset_index(drop=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp.parquet")
        pdf.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(cache_path)

    return pdf, {
        "cached": False,
        "elapsed_sec": round(total_sec, 2),
        "best_iterations": best_iters,
    }


def summarize_h3_predictions(
    seed_dfs: list[pd.DataFrame],
    months: tuple[str, ...],
    is_prob: bool,
    n_boot: int = 1000,
    boot_seed: int = 42,
    tie_seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute per-month and macro-monthly bootstrap metrics for a model across seeds."""
    per_month = [
        bootstrap_multiseed_per_month(
            seed_dfs, month=m, is_prob=is_prob, n_replicates=n_boot, boot_seed=boot_seed, tie_seed=tie_seed
        )
        for m in months
    ]
    macro = bootstrap_multiseed_macro_monthly(
        seed_dfs, months=months, is_prob=is_prob, n_replicates=n_boot, boot_seed=boot_seed, tie_seed=tie_seed
    )
    return per_month, macro


def quick_point_summary(pdf: pd.DataFrame, tie_seed: int = 42, is_prob: bool = True) -> dict[str, Any]:
    """Fast deterministic macro-monthly point metrics without bootstrap (for Val tuning/checks)."""
    aps, aucs, l1s, l5s, lls = [], [], [], [], []
    for _, g in pdf.groupby("month"):
        th = deterministic_tie_hash(g["client_id"].to_numpy(), seed=tie_seed)
        m = compute_slice_point_metrics(
            g["y_true"].to_numpy(),
            g["score"].to_numpy(),
            g["client_id"].to_numpy(),
            budget_fractions=(0.005, 0.01, 0.05),
            primary_budget=0.01,
            seed=tie_seed,
            is_probabilistic=is_prob,
            tie_hashes=th,
        )
        aps.append(m["average_precision"])
        aucs.append(m["roc_auc"])
        l1s.append(m["lift_1pct"])
        l5s.append(m["lift_5pct"])
        if m["log_loss"] is not None:
            lls.append(m["log_loss"])
    return {
        "macro_pr_auc": float(np.mean(aps)),
        "macro_roc_auc": float(np.mean(aucs)),
        "macro_lift_1pct": float(np.mean(l1s)),
        "macro_lift_5pct": float(np.mean(l5s)),
        "macro_log_loss": float(np.mean(lls)) if lls else float("nan"),
    }
