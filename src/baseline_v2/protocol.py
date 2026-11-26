"""Model-agnostic evaluation protocol engine (Protocol B Rolling OOC & Protocol A Flat OOC)."""
from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .models import build_model
from .spec import extract_feature_matrix, validate_feature_set

META_KEEP_COLS: tuple[str, ...] = (
    "client_id",
    "fold",
    "month",
    "cold_start",
    "trx_90d_has_event",
    "trx_all_has_event",
    "geo_90d_has_event",
    "geo_all_has_event",
    "dialog_90d_has_event",
    "dialog_all_has_event",
)


def iter_protocol_steps(
    spec: dict[str, Any],
    protocol_name: str,
    stage_name: str,
) -> list[tuple[list[str], list[str]]]:
    """Return list of (train_months, eval_months) steps for a protocol and stage."""
    proto = spec["protocols"][protocol_name]
    sched_key = "val_schedule" if stage_name == "val" else "test_schedule"
    steps: list[tuple[list[str], list[str]]] = []
    for entry in proto[sched_key]:
        tr_m = list(entry["train_months"])
        ev_m = [entry["eval_month"]] if "eval_month" in entry else list(entry["eval_months"])
        if max(tr_m) >= min(ev_m):
            raise AssertionError(f"Temporal leakage in {protocol_name}/{stage_name}: {tr_m} -> {ev_m}")
        steps.append((tr_m, ev_m))
    return steps


def run_protocol_experiment(
    spec: dict[str, Any],
    protocol_name: str,
    stage_name: str,
    panel_df: pd.DataFrame,
    exp_id: str,
    family: str,
    feature_cols: list[str] | tuple[str, ...],
    uses_target_lag: bool,
    seed: int = 42,
    model_params: dict[str, Any] | None = None,
    pred_cache_dir: Path | None = None,
    use_temporal_es: bool = False,
    shuffle_labels: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run one (protocol, stage, exp_id, seed) under strict 5-fold disjoint-client isolation."""
    validate_feature_set(exp_id, feature_cols, uses_target_lag, spec)
    n_folds = int(spec["data_contract"]["n_folds"])
    target_col = str(spec["data_contract"]["target_col"])
    fold_col = str(spec["data_contract"]["fold_col"])

    tag = f"{exp_id}_shuffled_seed{seed}" if shuffle_labels else f"{exp_id}_seed{seed}"
    cache_path = (pred_cache_dir / protocol_name / stage_name / f"{tag}.parquet") if pred_cache_dir else None
    if cache_path is not None and cache_path.exists():
        return pd.read_parquet(cache_path), {"cached": True, "elapsed_sec": 0.0, "best_iterations": []}

    steps = iter_protocol_steps(spec, protocol_name, stage_name)
    t0 = time.perf_counter()
    step_dfs: list[pd.DataFrame] = []
    best_iters: list[int] = []
    keep_cols = [c for c in META_KEEP_COLS if c in panel_df.columns]

    for tr_months, ev_months in steps:
        df_ev_step = panel_df.loc[panel_df["month"].isin(ev_months)].reset_index(drop=True)
        # When using temporal internal ES, split train_months[:-1] for fitting and train_months[-1] for ES
        can_es = bool(use_temporal_es and len(tr_months) >= 2)
        if can_es:
            tr_core_m = tr_months[:-1]
            tr_es_m = [tr_months[-1]]
            df_tr_core = panel_df.loc[panel_df["month"].isin(tr_core_m)].reset_index(drop=True)
            df_tr_es = panel_df.loc[panel_df["month"].isin(tr_es_m)].reset_index(drop=True)
        else:
            df_tr_full = panel_df.loc[panel_df["month"].isin(tr_months)].reset_index(drop=True)

        for k in range(n_folds):
            ev_k = (df_ev_step[fold_col].to_numpy() == k)
            ev_slice = df_ev_step.loc[ev_k]
            X_ev = extract_feature_matrix(ev_slice, feature_cols)
            y_ev = ev_slice[target_col].to_numpy(dtype=np.int8)
            ev_cids = ev_slice["client_id"].to_numpy()

            if can_es:
                core_k = (df_tr_core[fold_col].to_numpy() != k)
                es_k = (df_tr_es[fold_col].to_numpy() != k)
                tr_core_slice = df_tr_core.loc[core_k]
                tr_es_slice = df_tr_es.loc[es_k]
                tr_cids = np.concatenate([tr_core_slice["client_id"].to_numpy(), tr_es_slice["client_id"].to_numpy()])
                if not set(tr_cids).isdisjoint(set(ev_cids)):
                    raise AssertionError(f"Client overlap detected in {protocol_name}/{stage_name} fold={k}!")

                X_tr = extract_feature_matrix(tr_core_slice, feature_cols)
                y_tr = tr_core_slice[target_col].to_numpy(dtype=np.int8)
                X_es = extract_feature_matrix(tr_es_slice, feature_cols)
                y_es = tr_es_slice[target_col].to_numpy(dtype=np.int8)
                if shuffle_labels:
                    rng = np.random.default_rng(7100 + int(seed) * 100 + k)
                    y_tr = rng.permutation(y_tr)
                    y_es = rng.permutation(y_es)

                model = build_model(family=family, seed=seed, model_params=model_params)
                model.fit(X_tr, y_tr, X_es=X_es, y_es=y_es)
                if getattr(model, "best_iteration_", None) is not None:
                    best_iters.append(int(model.best_iteration_))
                del X_tr, y_tr, X_es, y_es
            else:
                tr_k = (df_tr_full[fold_col].to_numpy() != k)
                tr_slice = df_tr_full.loc[tr_k]
                if not set(tr_slice["client_id"].to_numpy()).isdisjoint(set(ev_cids)):
                    raise AssertionError(f"Client overlap detected in {protocol_name}/{stage_name} fold={k}!")

                X_tr = extract_feature_matrix(tr_slice, feature_cols)
                y_tr = tr_slice[target_col].to_numpy(dtype=np.int8)
                if shuffle_labels:
                    rng = np.random.default_rng(7100 + int(seed) * 100 + k)
                    y_tr = rng.permutation(y_tr)

                model = build_model(family=family, seed=seed, model_params=model_params)
                model.fit(X_tr, y_tr)
                del X_tr, y_tr

            scores = model.predict_scores(X_ev)
            meta = ev_slice[keep_cols].copy()
            meta["y_true"] = y_ev
            meta["score"] = scores.astype(np.float64)
            step_dfs.append(meta)
            del X_ev, model
            gc.collect()

    out_df = pd.concat(step_dfs, ignore_index=True).sort_values(["month", "client_id"]).reset_index(drop=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp.parquet")
        out_df.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(cache_path)

    return out_df, {
        "cached": False,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "best_iterations": best_iters,
    }
