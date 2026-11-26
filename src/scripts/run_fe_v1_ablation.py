"""FE-v1 Renaming, Raw Baseline (B-raw-T / B-raw-TGD) Multi-Seed Ablation,
Quality Gates, Permutation Importance, Slice Analysis, and Holm-Bonferroni Testing.

Strictly enforces the pre-registered frozen specification at:
  - configs/fe_v1_evaluation_spec.json
  - docs/FE_V1_FROZEN_SPEC.md
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import catboost
import lightgbm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
import sklearn
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from propensity_model.metrics import (
    compute_metrics,
    deterministic_rank_order,
    deterministic_tie_hash,
)

SPEC_JSON_PATH = REPO_ROOT / "configs" / "fe_v1_evaluation_spec.json"
SPEC_MD_PATH = REPO_ROOT / "docs" / "FE_V1_FROZEN_SPEC.md"

MOSCOW_DIR = REPO_ROOT / "outputs" / "baseline_moscow_v1"
PANEL_PATH = MOSCOW_DIR / "panel_moscow.parquet"
EMB_PATH = MOSCOW_DIR / "dialog_emb_moscow.npz"
LEGACY_PRED_DIR = MOSCOW_DIR / "predictions"
LEGACY_MANIFEST_PATH = MOSCOW_DIR / "ladder_manifest.json"

OUT_DIR = REPO_ROOT / "outputs" / "fe_v1_ablation"
PRED_DIR = OUT_DIR / "predictions"
MODEL_DIR = OUT_DIR / "models"
METRICS_DIR = OUT_DIR / "metrics"
FIG_DIR = OUT_DIR / "figures"
PCA_CACHE_DIR = OUT_DIR / "_pca_stage0_seed42"
FREEZE_LOCK_PATH = OUT_DIR / "spec_freeze_lock.json"

PCA_ALL_DIM = 16
PCA_90D_DIM = 8

SLICE_META_COLS: list[str] = [
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
]

RAW_FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "share",
    "fraction",
    "rate",
    "ratio",
    "geohash",
    "pca",
    "norm",
    "cosine",
    "first_activated",
    "cold_start",
    "prior_",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_or_create_spec_freeze_lock() -> dict[str, Any]:
    """Ensure Step 0 spec files exist and have NEVER been altered after locking."""
    if not SPEC_JSON_PATH.exists() or not SPEC_MD_PATH.exists():
        raise FileNotFoundError("Frozen spec files missing! Must lock Step 0 before running.")

    cur_json_hash = sha256_file(SPEC_JSON_PATH)
    cur_md_hash = sha256_file(SPEC_MD_PATH)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if FREEZE_LOCK_PATH.exists():
        lock = json.loads(FREEZE_LOCK_PATH.read_text(encoding="utf-8"))
        if lock["spec_json_sha256"] != cur_json_hash:
            raise AssertionError(
                f"VIOLATION: {SPEC_JSON_PATH} was modified after freeze lock! "
                f"Expected {lock['spec_json_sha256']}, got {cur_json_hash}"
            )
        if lock["spec_md_sha256"] != cur_md_hash:
            raise AssertionError(
                f"VIOLATION: {SPEC_MD_PATH} was modified after freeze lock! "
                f"Expected {lock['spec_md_sha256']}, got {cur_md_hash}"
            )
        return lock

    lock = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec_json_path": str(SPEC_JSON_PATH.relative_to(REPO_ROOT).as_posix()),
        "spec_json_sha256": cur_json_hash,
        "spec_md_path": str(SPEC_MD_PATH.relative_to(REPO_ROOT).as_posix()),
        "spec_md_sha256": cur_md_hash,
    }
    FREEZE_LOCK_PATH.write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
    return lock


def load_spec() -> dict[str, Any]:
    return json.loads(SPEC_JSON_PATH.read_text(encoding="utf-8"))


def resolve_model_features(spec: dict[str, Any], panel_cols: list[str]) -> dict[str, tuple[str, ...]]:
    trx_63 = tuple(c for c in panel_cols if c.startswith("trx_"))
    geo_meta_18 = tuple(c for c in panel_cols if c.startswith("geo_") and ("geohash" not in c))
    geo_geohash_12 = tuple(c for c in panel_cols if c.startswith("geo_") and ("geohash" in c))
    dialog_meta_18 = tuple(c for c in panel_cols if c.startswith("dialog_"))

    selector_map: dict[str, tuple[str, ...]] = {
        "all_trx_63": trx_63,
        "trx_63_plus_geo_meta_18_plus_dialog_meta_18_plus_cold_start_1": (
            trx_63 + geo_meta_18 + dialog_meta_18 + ("cold_start",)
        ),
        "trx_63_plus_geo_meta_18_plus_geo_geohash_12_plus_dialog_meta_18_plus_cold_start_1": (
            trx_63 + geo_meta_18 + geo_geohash_12 + dialog_meta_18 + ("cold_start",)
        ),
        "L5c_cb_112_plus_dialog_emb_pca_27": (
            trx_63 + geo_meta_18 + geo_geohash_12 + dialog_meta_18 + ("cold_start",)
        ),
    }

    resolved: dict[str, tuple[str, ...]] = {}
    for m in spec["models"]:
        eid = m["exp_id"]
        if "features" in m:
            cols = tuple(m["features"])
        else:
            cols = selector_map[m["feature_selector"]]
        resolved[eid] = cols
    return resolved


def run_static_quality_gates(spec: dict[str, Any], panel_cols: list[str], feat_map: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    """Step 6.1: Strict automated schema, leakage, feature-count, and exclusion checks."""
    forbidden = frozenset(spec["data_contract"]["forbidden_cols"])
    tlag_prefix = spec["data_contract"]["target_lag_prefix"]
    panel_set = frozenset(panel_cols)

    model_checks: dict[str, Any] = {}
    for m in spec["models"]:
        eid = m["exp_id"]
        cols = feat_map[eid]
        emb_dim = (PCA_ALL_DIM + PCA_90D_DIM + 3) if m["uses_emb"] else 0
        total_f = len(cols) + emb_dim

        missing_in_panel = [c for c in cols if c not in panel_set]
        if missing_in_panel:
            raise AssertionError(f"[{eid}] Missing columns in panel_moscow.parquet: {missing_in_panel}")
        if len(cols) != len(set(cols)):
            raise AssertionError(f"[{eid}] Duplicate columns detected!")
        if total_f != int(m["n_features"]):
            raise AssertionError(f"[{eid}] Feature count mismatch: got {total_f}, spec requires {m['n_features']}")

        bad_forbid = forbidden.intersection(cols)
        if bad_forbid:
            raise AssertionError(f"[{eid}] Forbidden leakage columns present: {bad_forbid}")

        has_tlag = [c for c in cols if c.startswith(tlag_prefix)]
        if not m["uses_target_lag"] and has_tlag:
            raise AssertionError(f"[{eid}] Target-Lag columns leaked into behavioral model: {has_tlag}")

        if eid in ("B-raw-T", "B-raw-TGD", "B-raw-TGD_logreg"):
            violated = [
                c for c in cols
                if any(sub in c.lower() for sub in RAW_FORBIDDEN_SUBSTRINGS)
            ]
            if violated:
                raise AssertionError(f"[{eid}] Violates Step 1 raw exclusion rule: {violated}")

        model_checks[eid] = {
            "expected_n_features": int(m["n_features"]),
            "actual_tabular_features": len(cols),
            "actual_emb_features": emb_dim,
            "actual_total_features": total_f,
            "forbidden_overlap": 0,
            "target_lag_columns": len(has_tlag),
            "passed": True,
        }

    # Explicit structural equivalence assertions from Step 1
    assert feat_map["B-raw-T"] == feat_map["L3_logreg_rfm"], "B-raw-T must have exact same 8 features as L3!"
    assert feat_map["B-raw-TGD"][:8] == feat_map["B-raw-T"], "B-raw-TGD must extend B-raw-T!"
    assert feat_map["B-raw-TGD"] == feat_map["B-raw-TGD_logreg"], "B-raw-TGD_logreg must match B-raw-TGD features!"

    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel_total_columns": len(panel_cols),
        "all_static_assertions_passed": True,
        "models": model_checks,
    }


def apply_step2_renaming_and_seed42_aliases(spec: dict[str, Any], panel_df: pd.DataFrame) -> dict[str, Any]:
    """Step 2: Rename L5d_cb_dialog_emb -> FE-v1 in predictions & manifest WITHOUT re-running seed=42."""
    # 1. Create FE-v1.parquet and FE-v1_cal.parquet aliases in baseline_moscow_v1/predictions
    for stage in ("val", "test"):
        src_raw = LEGACY_PRED_DIR / stage / "L5d_cb_dialog_emb.parquet"
        dst_raw = LEGACY_PRED_DIR / stage / "FE-v1.parquet"
        if src_raw.exists() and not dst_raw.exists():
            shutil.copy2(src_raw, dst_raw)
    src_cal = LEGACY_PRED_DIR / "test" / "L5d_cb_dialog_emb_cal.parquet"
    dst_cal = LEGACY_PRED_DIR / "test" / "FE-v1_cal.parquet"
    if src_cal.exists() and not dst_cal.exists():
        shutil.copy2(src_cal, dst_cal)

    # 2. Update baseline_moscow_v1/ladder_manifest.json non-destructively
    if LEGACY_MANIFEST_PATH.exists():
        leg_man = json.loads(LEGACY_MANIFEST_PATH.read_text(encoding="utf-8"))
        leg_man["tier_renaming_v2"] = {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "FE-v1": {
                "renamed_from": "L5d_cb_dialog_emb",
                "calibrated_renamed_from": "L5d_cb_dialog_emb_cal",
                "rerun_performed": False,
                "post_selection_disclosure": spec["design_decisions_locked"]["post_selection_disclosure"],
            },
            "intermediate_tiers_retained": ["L4_cb_trx", "L5c_cb_dialog_meta"],
            "next_milestone_id": "FE-v2",
        }
        LEGACY_MANIFEST_PATH.write_text(json.dumps(leg_man, indent=2, ensure_ascii=False), encoding="utf-8")

    # 3. Materialize seed=42 cached predictions in outputs/fe_v1_ablation/predictions/{stage}/
    #    enriching with slice columns (trx_90d_has_event, geo_90d_has_event, dialog_90d_has_event)
    #    strictly via 1-to-1 join on (month, client_id) from panel_moscow.parquet.
    copied: list[str] = []
    for stage, eval_months in (
        ("val", tuple(spec["data_contract"]["val_months"])),
        ("test", tuple(spec["data_contract"]["test_months"])),
    ):
        meta_sub = panel_df.loc[panel_df["month"].isin(eval_months), SLICE_META_COLS].copy()
        stage_out = PRED_DIR / stage
        stage_out.mkdir(parents=True, exist_ok=True)

        for m in spec["models"]:
            leg_id = m.get("legacy_exp_id")
            eid = m["exp_id"]
            if not leg_id:
                continue
            dst = stage_out / f"{eid}_seed42.parquet"
            if dst.exists():
                copied.append(f"{stage}/{dst.name}")
                continue
            src = LEGACY_PRED_DIR / stage / f"{leg_id}.parquet"
            if not src.exists():
                raise FileNotFoundError(f"Expected legacy prediction file missing: {src}")
            pdf = pd.read_parquet(src)[["client_id", "month", "y_true", "score"]]
            merged = meta_sub.merge(pdf, on=["client_id", "month"], how="inner", validate="one_to_one")
            assert len(merged) == len(pdf) == len(meta_sub), f"Row mismatch aliasing {leg_id} -> {eid} on {stage}"
            merged = merged.sort_values(["month", "client_id"]).reset_index(drop=True)
            tmp = dst.with_suffix(".tmp.parquet")
            merged.to_parquet(tmp, index=False, compression="zstd")
            tmp.replace(dst)
            copied.append(f"{stage}/{dst.name}")

    return {"seed42_aliased_files": copied}


def get_rss_mb() -> float:
    return float(psutil.Process().memory_info().rss / 1e6)


def ensure_stage0_pca_cached(spec: dict[str, Any]) -> None:
    """Compute frozen Stage-0 PCA slices (canonical seed=42+k) with bounded RSS (<1.45 GB)."""
    needed = False
    for stage in ("val", "test"):
        for k in range(5):
            if not (PCA_CACHE_DIR / stage / f"f{k}_tr.npy").exists() or not (PCA_CACHE_DIR / stage / f"f{k}_ev.npy").exists():
                needed = True
    if not needed:
        print("[PCA] Using cached Stage-0 frozen PCA slices.", flush=True)
        return

    print(f"[PCA] Computing Stage-0 frozen train-only PCA slices (seed=42+k) | start RSS={get_rss_mb():.1f} MB ...", flush=True)
    PCA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_df = pd.read_parquet(PANEL_PATH, columns=["client_id", "fold", "month"])
    base_seed = int(spec["protocol"]["canonical_seed"])

    tmp_all_path = PCA_CACHE_DIR / "_tmp_all_emb.npy"
    tmp_w90_path = PCA_CACHE_DIR / "_tmp_w90_emb.npy"

    # Load metadata arrays & build lightweight MultiIndex lookups (avoids huge Python dict of tuples)
    with np.load(EMB_PATH, allow_pickle=False) as z:
        all_months = z["all_months"].astype(str)
        all_clients = z["all_clients"].astype(str)
        w90_months = z["w90_months"].astype(str)
        w90_clients = z["w90_clients"].astype(str)

        idx_all = pd.MultiIndex.from_arrays([all_months, all_clients])
        idx_90 = pd.MultiIndex.from_arrays([w90_months, w90_clients])

        # Pass 1: all_emb -> compute norms and spill uncompressed float32 .npy for mmap
        all_emb = z["all_emb"].astype(np.float32, copy=False)
        all_norms = np.linalg.norm(all_emb, axis=1).astype(np.float32)
        np.save(tmp_all_path, all_emb)
        del all_emb
        gc.collect()

        # Pass 2: w90_emb -> compute norms and cosine(90d, all) via chunked mmap lookup, then spill .npy
        w90_emb = z["w90_emb"].astype(np.float32, copy=False)
        w90_norms = np.linalg.norm(w90_emb, axis=1).astype(np.float32)
        w90_to_all = idx_all.get_indexer(pd.MultiIndex.from_arrays([w90_months, w90_clients]))
        cos_w90 = np.full(len(w90_emb), np.nan, dtype=np.float32)
        all_mm = np.load(tmp_all_path, mmap_mode="r")
        valid_match = w90_to_all >= 0
        valid_positions = np.flatnonzero(valid_match)
        chunk = 25000
        for start in range(0, len(valid_positions), chunk):
            pos = valid_positions[start:start + chunk]
            a_rows = all_mm[w90_to_all[pos]]
            w_rows = w90_emb[pos]
            dot = np.sum(a_rows * w_rows, axis=1)
            denom = np.maximum(all_norms[w90_to_all[pos]] * w90_norms[pos], 1e-8)
            cos_w90[pos] = (dot / denom).astype(np.float32)
        del all_mm
        np.save(tmp_w90_path, w90_emb)
        del w90_emb, all_months, all_clients, w90_months, w90_clients, w90_to_all
        gc.collect()

    print(f"  [PCA] Precomputed norms/cosine & spilled mmap caches | RSS={get_rss_mb():.1f} MB", flush=True)

    all_mm = np.load(tmp_all_path, mmap_mode="r")
    w90_mm = np.load(tmp_w90_path, mmap_mode="r")
    n_feat = PCA_ALL_DIM + PCA_90D_DIM + 3
    peak_rss = get_rss_mb()

    for stage, tr_m, ev_m in (
        ("val", tuple(spec["data_contract"]["train_months_val"]), tuple(spec["data_contract"]["val_months"])),
        ("test", tuple(spec["data_contract"]["train_months_test"]), tuple(spec["data_contract"]["test_months"])),
    ):
        assert max(tr_m) < min(ev_m)
        sdir = PCA_CACHE_DIR / stage
        sdir.mkdir(parents=True, exist_ok=True)
        df_tr = meta_df.loc[meta_df["month"].isin(tr_m)].reset_index(drop=True)
        df_ev = meta_df.loc[meta_df["month"].isin(ev_m)].reset_index(drop=True)

        for k in range(5):
            tr_k = df_tr["fold"].to_numpy() != k
            ev_k = df_ev["fold"].to_numpy() == k
            tr_cids = df_tr.loc[tr_k, "client_id"].to_numpy()
            ev_cids = df_ev.loc[ev_k, "client_id"].to_numpy()
            assert len(set(tr_cids).intersection(set(ev_cids))) == 0

            tr_mi = pd.MultiIndex.from_arrays([df_tr.loc[tr_k, "month"].astype(str).to_numpy(), tr_cids.astype(str)])
            ev_mi = pd.MultiIndex.from_arrays([df_ev.loc[ev_k, "month"].astype(str).to_numpy(), ev_cids.astype(str)])

            tr_ia = idx_all.get_indexer(tr_mi)
            ev_ia = idx_all.get_indexer(ev_mi)
            tr_i9 = idx_90.get_indexer(tr_mi)
            ev_i9 = idx_90.get_indexer(ev_mi)

            tr_va, ev_va = tr_ia >= 0, ev_ia >= 0
            tr_v9, ev_v9 = tr_i9 >= 0, ev_i9 >= 0

            F_tr = np.full((len(tr_cids), n_feat), np.nan, dtype=np.float32)
            F_ev = np.full((len(ev_cids), n_feat), np.nan, dtype=np.float32)

            F_tr[tr_va, PCA_ALL_DIM + PCA_90D_DIM] = all_norms[tr_ia[tr_va]]
            F_ev[ev_va, PCA_ALL_DIM + PCA_90D_DIM] = all_norms[ev_ia[ev_va]]
            F_tr[tr_v9, PCA_ALL_DIM + PCA_90D_DIM + 1] = w90_norms[tr_i9[tr_v9]]
            F_ev[ev_v9, PCA_ALL_DIM + PCA_90D_DIM + 1] = w90_norms[ev_i9[ev_v9]]

            tr_both = tr_va & tr_v9
            ev_both = ev_va & ev_v9
            F_tr[tr_both, PCA_ALL_DIM + PCA_90D_DIM + 2] = cos_w90[tr_i9[tr_both]]
            F_ev[ev_both, PCA_ALL_DIM + PCA_90D_DIM + 2] = cos_w90[ev_i9[ev_both]]

            # All-history PCA(16)
            tr_ea = np.array(all_mm[tr_ia[tr_va]], dtype=np.float32)
            sc_all = StandardScaler(copy=False)
            pca_all = PCA(n_components=PCA_ALL_DIM, svd_solver="randomized", random_state=base_seed + k)
            F_tr[tr_va, :PCA_ALL_DIM] = pca_all.fit_transform(sc_all.fit_transform(tr_ea)).astype(np.float32)
            del tr_ea
            gc.collect()
            if np.any(ev_va):
                ev_ea = np.array(all_mm[ev_ia[ev_va]], dtype=np.float32)
                F_ev[ev_va, :PCA_ALL_DIM] = pca_all.transform(sc_all.transform(ev_ea)).astype(np.float32)
                del ev_ea
            peak_rss = max(peak_rss, get_rss_mb())
            del sc_all, pca_all
            gc.collect()

            # 90d PCA(8)
            tr_e9 = np.array(w90_mm[tr_i9[tr_v9]], dtype=np.float32)
            sc_90 = StandardScaler(copy=False)
            pca_90 = PCA(n_components=PCA_90D_DIM, svd_solver="randomized", random_state=base_seed + k)
            F_tr[tr_v9, PCA_ALL_DIM:PCA_ALL_DIM + PCA_90D_DIM] = pca_90.fit_transform(sc_90.fit_transform(tr_e9)).astype(np.float32)
            del tr_e9
            gc.collect()
            if np.any(ev_v9):
                ev_e9 = np.array(w90_mm[ev_i9[ev_v9]], dtype=np.float32)
                F_ev[ev_v9, PCA_ALL_DIM:PCA_ALL_DIM + PCA_90D_DIM] = pca_90.transform(sc_90.transform(ev_e9)).astype(np.float32)
                del ev_e9
            peak_rss = max(peak_rss, get_rss_mb())
            del sc_90, pca_90
            gc.collect()

            np.save(sdir / f"f{k}_tr.npy", F_tr)
            np.save(sdir / f"f{k}_ev.npy", F_ev)
            del F_tr, F_ev
            gc.collect()
            print(f"  [PCA {stage.upper():4s} fold={k}] saved | RSS={get_rss_mb():.1f} MB (peak={peak_rss:.1f} MB)", flush=True)

    del all_mm, w90_mm, meta_df, idx_all, idx_90, all_norms, w90_norms, cos_w90
    gc.collect()
    tmp_all_path.unlink(missing_ok=True)
    tmp_w90_path.unlink(missing_ok=True)


def build_estimator(family: str, seed: int, cb_cfg: dict[str, Any]):
    if family == "logreg":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=300, random_state=int(seed))),
        ])
    if family == "catboost":
        return CatBoostClassifier(
            iterations=int(cb_cfg["iterations"]),
            random_seed=int(seed),
            thread_count=int(cb_cfg["thread_count"]),
            border_count=int(cb_cfg["border_count"]),
            used_ram_limit=str(cb_cfg["used_ram_limit"]),
            loss_function=str(cb_cfg["loss_function"]),
            verbose=bool(cb_cfg["verbose"]),
            allow_writing_files=bool(cb_cfg["allow_writing_files"]),
        )
    raise ValueError(f"Unsupported stochastic family: {family}")


def train_and_score_single_run(
    stage_name: str,
    df_tr_all: pd.DataFrame,
    df_ev_all: pd.DataFrame,
    exp_spec: dict[str, Any],
    cols: list[str],
    seed: int,
    cb_cfg: dict[str, Any],
    save_fold_models: bool = False,
    shuffle_labels: bool = False,
) -> tuple[pd.DataFrame, float, dict[str, Any]]:
    """Run 5-fold disjoint-client training for a single (stage, exp_id, seed)."""
    eid = exp_spec["exp_id"]
    tag = f"{eid}_shuffled_seed{seed}" if shuffle_labels else f"{eid}_seed{seed}"
    pred_path = PRED_DIR / stage_name / f"{tag}.parquet"
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if fold models are also requested and already present
    models_ok = True
    if save_fold_models and not shuffle_labels:
        for k in range(5):
            if not (MODEL_DIR / stage_name / f"{eid}_seed{seed}_f{k}.cbm").exists():
                models_ok = False

    if pred_path.exists() and models_ok:
        return pd.read_parquet(pred_path), 0.0, {"cached": True}

    t0 = time.perf_counter()
    fold_dfs: list[pd.DataFrame] = []
    peak_rss = get_rss_mb()
    nan_stats = {"tr_nan_cells": 0, "ev_nan_cells": 0, "zero_client_overlap": True, "peak_rss_mb": peak_rss}

    n_tab = len(cols)
    n_emb = (PCA_ALL_DIM + PCA_90D_DIM + 3) if exp_spec["uses_emb"] else 0
    total_f = n_tab + n_emb

    for k in range(5):
        tr_k = df_tr_all["fold"].to_numpy() != k
        ev_k = df_ev_all["fold"].to_numpy() == k

        tr_cids = df_tr_all.loc[tr_k, "client_id"].to_numpy()
        ev_cids = df_ev_all.loc[ev_k, "client_id"].to_numpy()
        if len(set(tr_cids).intersection(set(ev_cids))) != 0:
            raise AssertionError(f"Client overlap detected in {stage_name} fold={k}!")

        X_tr = np.empty((int(tr_k.sum()), total_f), dtype=np.float32)
        X_ev = np.empty((int(ev_k.sum()), total_f), dtype=np.float32)
        for j, cname in enumerate(cols):
            X_tr[:, j] = df_tr_all.loc[tr_k, cname].to_numpy(dtype=np.float32, copy=False)
            X_ev[:, j] = df_ev_all.loc[ev_k, cname].to_numpy(dtype=np.float32, copy=False)

        if n_emb > 0:
            F_tr_mm = np.load(PCA_CACHE_DIR / stage_name / f"f{k}_tr.npy", mmap_mode="r")
            F_ev_mm = np.load(PCA_CACHE_DIR / stage_name / f"f{k}_ev.npy", mmap_mode="r")
            X_tr[:, n_tab:] = F_tr_mm
            X_ev[:, n_tab:] = F_ev_mm
            del F_tr_mm, F_ev_mm

        if eid in ("B-raw-T", "B-raw-TGD") and exp_spec["family"] == "catboost":
            tr_nans = int(np.isnan(X_tr).sum())
            ev_nans = int(np.isnan(X_ev).sum())
            if tr_nans == 0 or ev_nans == 0:
                raise AssertionError(f"[{eid}] Expected semantic NaNs preserved in raw features, got 0!")
            nan_stats["tr_nan_cells"] += tr_nans
            nan_stats["ev_nan_cells"] += ev_nans

        y_tr = df_tr_all.loc[tr_k, "y_any"].to_numpy(dtype=np.int8)
        y_ev = df_ev_all.loc[ev_k, "y_any"].to_numpy(dtype=np.int8)

        if shuffle_labels:
            rng = np.random.default_rng(4200 + int(seed) * 10 + k)
            y_tr = rng.permutation(y_tr)

        est = build_estimator(exp_spec["family"], seed=seed, cb_cfg=cb_cfg)
        est.fit(X_tr, y_tr)
        scores = est.predict_proba(X_ev)[:, 1].astype(np.float64)
        peak_rss = max(peak_rss, get_rss_mb())

        if save_fold_models and not shuffle_labels and exp_spec["family"] == "catboost":
            mdir = MODEL_DIR / stage_name
            mdir.mkdir(parents=True, exist_ok=True)
            est.save_model(str(mdir / f"{eid}_seed{seed}_f{k}.cbm"))

        ev_meta = df_ev_all.loc[ev_k, SLICE_META_COLS].copy()
        ev_meta["y_true"] = y_ev
        ev_meta["score"] = scores
        fold_dfs.append(ev_meta)

        del X_tr, X_ev, est
        gc.collect()

    pdf = pd.concat(fold_dfs, ignore_index=True).sort_values(["month", "client_id"]).reset_index(drop=True)
    assert len(pdf) == len(df_ev_all)
    tmp = pred_path.with_suffix(".tmp.parquet")
    pdf.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(pred_path)
    sec = time.perf_counter() - t0
    nan_stats["cached"] = False
    nan_stats["peak_rss_mb"] = round(peak_rss, 1)
    return pdf, sec, nan_stats


def run_training_wave(
    spec: dict[str, Any],
    feat_map: dict[str, tuple[str, ...]],
    target_exp_ids: tuple[str, ...],
    seeds_override: list[int] | None = None,
    panel_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    seeds = seeds_override if seeds_override is not None else [int(s) for s in spec["protocol"]["seeds"]]
    cb_cfg = spec["protocol"]["catboost_default_params"]
    model_map = {m["exp_id"]: m for m in spec["models"]}

    if any(model_map[e]["uses_emb"] for e in target_exp_ids):
        ensure_stage0_pca_cached(spec)

    if panel_df is None:
        needed_feat_cols = sorted({c for eid in target_exp_ids for c in feat_map[eid]})
        read_cols = list(dict.fromkeys(SLICE_META_COLS + ["y_any"] + needed_feat_cols))
        print(f"[IO] Reading {len(read_cols)} pruned columns from panel_moscow.parquet | RSS={get_rss_mb():.1f} MB", flush=True)
        panel_df = pd.read_parquet(PANEL_PATH, columns=read_cols)
        owns_panel = True
    else:
        owns_panel = False

    timings_path = METRICS_DIR / "fe_v1_timings.csv"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, Any]] = []
    if timings_path.exists():
        existing_rows = pd.read_csv(timings_path).to_dict("records")

    new_rows: list[dict[str, Any]] = []
    for stage_name, tr_m, ev_m in (
        ("val", tuple(spec["data_contract"]["train_months_val"]), tuple(spec["data_contract"]["val_months"])),
        ("test", tuple(spec["data_contract"]["train_months_test"]), tuple(spec["data_contract"]["test_months"])),
    ):
        assert max(tr_m) < min(ev_m)
        df_tr_all = panel_df.loc[panel_df["month"].isin(tr_m)].reset_index(drop=True)
        df_ev_all = panel_df.loc[panel_df["month"].isin(ev_m)].reset_index(drop=True)

        for eid in target_exp_ids:
            mspec = model_map[eid]
            if not mspec["stochastic_across_seeds"]:
                continue
            cols = list(feat_map[eid])
            for s in seeds:
                save_cbm = (eid in ("B-raw-T", "B-raw-TGD")) and (s == int(spec["protocol"]["canonical_seed"]))
                pdf, sec, info = train_and_score_single_run(
                    stage_name=stage_name,
                    df_tr_all=df_tr_all,
                    df_ev_all=df_ev_all,
                    exp_spec=mspec,
                    cols=cols,
                    seed=s,
                    cb_cfg=cb_cfg,
                    save_fold_models=save_cbm,
                    shuffle_labels=False,
                )
                mm, _ = compute_metrics(
                    pdf["y_true"].to_numpy(),
                    pdf["score"].to_numpy(),
                    pdf["client_id"].to_numpy(),
                    budget_fractions=(0.01, 0.05, 0.10),
                    primary_budget=0.01,
                    seed=42,
                    is_probabilistic=mspec["is_probabilistic"],
                )
                status = "cached" if info.get("cached") else f"{sec:.1f}s | peakRSS={info.get('peak_rss_mb', get_rss_mb()):.0f}MB"
                print(
                    f"[{stage_name.upper():4s}] {eid:18s} seed={s:3d} ({mspec['n_features']:3d}f) | "
                    f"Pooled PR-AUC={mm['average_precision']:.4f} | AUC={mm['roc_auc']:.4f} | "
                    f"Lift@1%={mm['lift_1pct']:.2f}x | Lift@5%={mm['lift_5pct']:.2f}x ({status})",
                    flush=True,
                )
                new_rows.append({
                    "stage": stage_name,
                    "exp_id": eid,
                    "seed": s,
                    "n_features": mspec["n_features"],
                    "seconds": round(sec, 2),
                    "cached": bool(info.get("cached")),
                    "peak_rss_mb": info.get("peak_rss_mb", round(get_rss_mb(), 1)),
                    "tr_nan_cells": info.get("tr_nan_cells", 0),
                    "ev_nan_cells": info.get("ev_nan_cells", 0),
                })

        del df_tr_all, df_ev_all
        gc.collect()

    if owns_panel:
        del panel_df
        gc.collect()

    all_t = pd.DataFrame(existing_rows + new_rows).drop_duplicates(subset=["stage", "exp_id", "seed"], keep="last")
    all_t.to_csv(timings_path, index=False)
    return new_rows


# ---------------------------------------------------------------------------
# Fast Vectorized Multi-Seed Client-Clustered Bootstrap & Holm-Bonferroni
# ---------------------------------------------------------------------------
BOOT_KEYS = (
    "average_precision",
    "roc_auc",
    "lift_1pct",
    "recall_1pct",
    "precision_1pct",
    "lift_5pct",
    "recall_5pct",
    "lift_10pct",
    "recall_10pct",
    "log_loss",
    "brier",
)


def _prep_sorted(y: np.ndarray, scores: np.ndarray, codes: np.ndarray, hashes: np.ndarray, is_prob: bool) -> dict[str, Any]:
    order = np.lexsort((hashes, -scores))
    y_s = y[order].astype(np.float64)
    s_s = scores[order].astype(np.float64)
    c_s = codes[order]
    diff = np.empty(len(s_s), dtype=bool)
    diff[-1] = True
    if len(s_s) > 1:
        diff[:-1] = s_s[:-1] != s_s[1:]
    row_ll = None
    row_bs = None
    if is_prob:
        p = np.clip(s_s, 1e-15, 1.0 - 1e-15)
        row_ll = -(y_s * np.log(p) + (1.0 - y_s) * np.log(1.0 - p))
        row_bs = (s_s - y_s) ** 2
    return {"y_s": y_s, "c_s": c_s, "ge": np.flatnonzero(diff), "row_ll": row_ll, "row_bs": row_bs, "is_prob": is_prob}


def _w_metrics_multi_k(prep: dict[str, Any], cw: np.ndarray) -> dict[str, float]:
    rw = cw[prep["c_s"]].astype(np.float64)
    tot_n = float(rw.sum())
    if tot_n <= 0.0:
        return {k: float("nan") for k in BOOT_KEYS}
    y_s = prep["y_s"]
    pos_w = rw * y_s
    tot_pos = float(pos_w.sum())
    tot_neg = tot_n - tot_pos
    cum_w = np.cumsum(rw)
    cum_pos = np.cumsum(pos_w)
    prev = tot_pos / tot_n if tot_n > 0 else float("nan")

    out: dict[str, float] = {}
    for frac, tag in ((0.01, "1pct"), (0.05, "5pct"), (0.10, "10pct")):
        k_t = max(1, int(math.ceil(frac * tot_n)))
        idx = int(np.searchsorted(cum_w, k_t, side="left"))
        if idx >= len(cum_w):
            hits = tot_pos
        else:
            pw = float(cum_w[idx - 1]) if idx > 0 else 0.0
            pp = float(cum_pos[idx - 1]) if idx > 0 else 0.0
            hits = pp + (k_t - pw) * float(y_s[idx])
        prec = hits / k_t
        rec = (hits / tot_pos) if tot_pos > 0 else float("nan")
        lift = (prec / prev) if prev > 0 else float("nan")
        out[f"lift_{tag}"] = lift
        out[f"recall_{tag}"] = rec
        if tag == "1pct":
            out["precision_1pct"] = prec

    if tot_pos > 0 and tot_neg > 0:
        ge = prep["ge"]
        tp = cum_pos[ge]
        tot = cum_w[ge]
        fp = tot - tp
        dtp = np.empty_like(tp)
        dtp[0] = tp[0]
        if len(tp) > 1:
            dtp[1:] = tp[1:] - tp[:-1]
        nz = tot > 0
        pt = np.zeros_like(tp)
        pt[nz] = tp[nz] / tot[nz]
        out["average_precision"] = float(np.sum((dtp / tot_pos) * pt))
        out["roc_auc"] = float(np.trapezoid(np.concatenate(([0.0], tp / tot_pos)), np.concatenate(([0.0], fp / tot_neg))))
    else:
        out["average_precision"] = float("nan")
        out["roc_auc"] = float("nan")

    out["log_loss"] = float(np.dot(rw, prep["row_ll"]) / tot_n) if prep["is_prob"] else float("nan")
    out["brier"] = float(np.dot(rw, prep["row_bs"]) / tot_n) if prep["is_prob"] else float("nan")
    return out


def holm_bonferroni(p_values: list[float | None]) -> list[float | None]:
    """Exact step-down Holm-Bonferroni adjusted p-values preserving monotonicity."""
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None and np.isfinite(p)]
    m = len(indexed)
    adj: list[float | None] = [None] * len(p_values)
    if m == 0:
        return adj
    indexed.sort(key=lambda x: x[1])
    running_max = 0.0
    for rank_idx, (orig_i, p_raw) in enumerate(indexed):
        mult = m - rank_idx
        val = min(1.0, mult * float(p_raw))
        running_max = max(running_max, val)
        adj[orig_i] = running_max
    return adj


def load_pred_for_seed(stage: str, exp_spec: dict[str, Any], seed: int) -> pd.DataFrame:
    eid = exp_spec["exp_id"]
    actual_seed = seed if exp_spec["stochastic_across_seeds"] else 42
    p = PRED_DIR / stage / f"{eid}_seed{actual_seed}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing prediction file: {p}")
    return pd.read_parquet(p)


def run_shuffled_and_permutation_checks(spec: dict[str, Any], feat_map: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    """Step 6.2 & 6.4: Label-shuffle sanity check and Out-of-Fold Permutation Importance for B-raw."""
    cb_cfg = spec["protocol"]["catboost_default_params"]
    model_map = {m["exp_id"]: m for m in spec["models"]}
    shuf_results: list[dict[str, Any]] = []
    perm_rows: list[dict[str, Any]] = []

    read_cols = list(dict.fromkeys(SLICE_META_COLS + ["y_any"] + list(feat_map["B-raw-TGD"])))
    panel_df = pd.read_parquet(PANEL_PATH, columns=read_cols)

    for stage_name, tr_m, ev_m in (
        ("val", tuple(spec["data_contract"]["train_months_val"]), tuple(spec["data_contract"]["val_months"])),
        ("test", tuple(spec["data_contract"]["train_months_test"]), tuple(spec["data_contract"]["test_months"])),
    ):
        df_tr_all = panel_df.loc[panel_df["month"].isin(tr_m)].reset_index(drop=True)
        df_ev_all = panel_df.loc[panel_df["month"].isin(ev_m)].reset_index(drop=True)

        # 1a. Shuffled-y_train check: 10 independent shuffles for low-dim LogReg (L3), 1 shuffle for CatBoost (B-raw-T, B-raw-TGD)
        for eid in ("L3_logreg_rfm", "B-raw-T", "B-raw-TGD"):
            mspec = model_map[eid]
            cols = list(feat_map[eid])
            shuf_seeds = (42, 100, 200, 300, 400, 500, 600, 700, 800, 900) if eid == "L3_logreg_rfm" else (42,)
            s_aps, s_aucs, s_l1s, secs = [], [], [], []
            prevalence_val = float(df_ev_all["y_any"].mean())
            for sh_s in shuf_seeds:
                pdf_shuf, sec, _ = train_and_score_single_run(
                    stage_name=stage_name,
                    df_tr_all=df_tr_all,
                    df_ev_all=df_ev_all,
                    exp_spec=mspec,
                    cols=cols,
                    seed=sh_s,
                    cb_cfg=cb_cfg,
                    save_fold_models=False,
                    shuffle_labels=True,
                )
                mm, _ = compute_metrics(
                    pdf_shuf["y_true"].to_numpy(),
                    pdf_shuf["score"].to_numpy(),
                    pdf_shuf["client_id"].to_numpy(),
                    budget_fractions=(0.01, 0.05, 0.10),
                    primary_budget=0.01,
                    seed=42,
                    is_probabilistic=True,
                )
                s_aps.append(float(mm["average_precision"]))
                s_aucs.append(float(mm["roc_auc"]))
                s_l1s.append(float(mm["lift_1pct"]))
                secs.append(float(sec))
                prevalence_val = float(mm["prevalence"])

            mean_ap = float(np.mean(s_aps))
            mean_auc = float(np.mean(s_aucs))
            mean_l1 = float(np.mean(s_l1s))
            gap = abs(mean_ap - prevalence_val)
            passed = (mean_ap < prevalence_val + 0.0020) and (gap < 0.0025) and (0.42 <= mean_auc <= 0.58)
            if not passed:
                raise AssertionError(
                    f"Shuffled-y_train check FAILED for {eid} on {stage_name}: "
                    f"mean PR-AUC={mean_ap:.5f} vs base_rate={prevalence_val:.5f}, mean AUC={mean_auc:.4f}"
                )
            shuf_results.append({
                "stage": stage_name,
                "check_type": "shuffle_y_train",
                "n_shuffles": len(shuf_seeds),
                "exp_id": eid,
                "prevalence": prevalence_val,
                "shuffled_pr_auc": mean_ap,
                "abs_gap_to_prevalence": gap,
                "shuffled_roc_auc": mean_auc,
                "shuffled_lift_1pct": mean_l1,
                "seconds": round(float(np.sum(secs)), 2),
                "passed": True,
            })
            print(
                f"[SHUF-YTR {stage_name.upper():4s}] {eid:15s} ({len(shuf_seeds)}x) | PR-AUC={mean_ap:.5f} "
                f"(base={prevalence_val:.5f}) | AUC={mean_auc:.4f} | Lift@1%={mean_l1:.2f}x -> PASS",
                flush=True,
            )

        # 1b. Shuffled-y_eval check (10 repeats) on real seed=42 predictions
        for eid in ("L3_logreg_rfm", "B-raw-T", "B-raw-TGD", "L4_cb", "FE-v1"):
            real_pdf = load_pred_for_seed(stage_name, model_map[eid], 42)
            y_real = real_pdf["y_true"].to_numpy(dtype=np.int8)
            sc_real = real_pdf["score"].to_numpy(dtype=np.float64)
            c_real = real_pdf["client_id"].to_numpy()
            tie_real = deterministic_tie_hash(c_real, seed=42)
            prev_val = float(np.mean(y_real))
            e_aps, e_aucs, e_l1s = [], [], []
            for rep in range(10):
                rng_e = np.random.default_rng(7700 + rep)
                y_perm = rng_e.permutation(y_real)
                em_m, _ = compute_metrics(
                    y_perm, sc_real, c_real,
                    budget_fractions=(0.01,), primary_budget=0.01,
                    seed=42, is_probabilistic=False, tie_hashes=tie_real,
                )
                e_aps.append(float(em_m["average_precision"]))
                e_aucs.append(float(em_m["roc_auc"]))
                e_l1s.append(float(em_m["lift_1pct"]))
            mean_e_ap = float(np.mean(e_aps))
            mean_e_auc = float(np.mean(e_aucs))
            mean_e_l1 = float(np.mean(e_l1s))
            gap_e = abs(mean_e_ap - prev_val)
            passed_e = (gap_e < 0.0008) and (0.48 <= mean_e_auc <= 0.52)
            if not passed_e:
                raise AssertionError(
                    f"Shuffled-y_eval check FAILED for {eid} on {stage_name}: "
                    f"mean PR-AUC={mean_e_ap:.5f} vs base={prev_val:.5f}, mean AUC={mean_e_auc:.4f}"
                )
            shuf_results.append({
                "stage": stage_name,
                "check_type": "shuffle_y_eval_10rep",
                "n_shuffles": 10,
                "exp_id": eid,
                "prevalence": prev_val,
                "shuffled_pr_auc": mean_e_ap,
                "abs_gap_to_prevalence": gap_e,
                "shuffled_roc_auc": mean_e_auc,
                "shuffled_lift_1pct": mean_e_l1,
                "seconds": 0.0,
                "passed": True,
            })
            print(
                f"[SHUF-YEV {stage_name.upper():4s}] {eid:15s} (10x) | PR-AUC={mean_e_ap:.5f} "
                f"(base={prev_val:.5f}) | AUC={mean_e_auc:.4f} | Lift@1%={mean_e_l1:.2f}x -> PASS",
                flush=True,
            )

        # 2. Out-of-Fold Permutation Importance for B-raw-T and B-raw-TGD (seed=42, n_repeats=5)
        for eid in ("B-raw-T", "B-raw-TGD"):
            cols = list(feat_map[eid])
            base_pdf = load_pred_for_seed(stage_name, model_map[eid], 42)
            tie_h = deterministic_tie_hash(base_pdf["client_id"].to_numpy(), seed=42)
            base_m, _ = compute_metrics(
                base_pdf["y_true"].to_numpy(),
                base_pdf["score"].to_numpy(),
                base_pdf["client_id"].to_numpy(),
                budget_fractions=(0.01, 0.05),
                primary_budget=0.01,
                seed=42,
                is_probabilistic=True,
                tie_hashes=tie_h,
            )
            fold_models = []
            for k in range(5):
                cbm = CatBoostClassifier()
                cbm.load_model(str(MODEL_DIR / stage_name / f"{eid}_seed42_f{k}.cbm"))
                fold_models.append(cbm)

            # Pre-extract per-fold eval arrays aligned to sorted (month, client_id)
            df_ev_sorted = df_ev_all.sort_values(["month", "client_id"]).reset_index(drop=True)
            assert np.array_equal(df_ev_sorted["client_id"].to_numpy(), base_pdf["client_id"].to_numpy())
            fold_arr = df_ev_sorted["fold"].to_numpy()
            X_full = df_ev_sorted[cols].to_numpy(dtype=np.float32, copy=True)
            y_full = base_pdf["y_true"].to_numpy(dtype=np.int8)
            c_full = base_pdf["client_id"].to_numpy()

            n_repeats = 5
            for j, fname in enumerate(cols):
                d_aps, d_l1s, d_aucs = [], [], []
                orig_col = X_full[:, j].copy()
                for r in range(n_repeats):
                    rng = np.random.default_rng(9000 + j * 100 + r)
                    perm_scores = np.empty(len(df_ev_sorted), dtype=np.float64)
                    for k in range(5):
                        mk = fold_arr == k
                        X_k = X_full[mk].copy()
                        X_k[:, j] = rng.permutation(X_k[:, j])
                        perm_scores[mk] = fold_models[k].predict_proba(X_k)[:, 1]
                    pm, _ = compute_metrics(
                        y_full, perm_scores, c_full,
                        budget_fractions=(0.01, 0.05), primary_budget=0.01,
                        seed=42, is_probabilistic=True, tie_hashes=tie_h,
                    )
                    d_aps.append(float(base_m["average_precision"] - pm["average_precision"]))
                    d_l1s.append(float(base_m["lift_1pct"] - pm["lift_1pct"]))
                    d_aucs.append(float(base_m["roc_auc"] - pm["roc_auc"]))
                X_full[:, j] = orig_col
                perm_rows.append({
                    "stage": stage_name,
                    "exp_id": eid,
                    "feature": fname,
                    "base_pr_auc": float(base_m["average_precision"]),
                    "base_lift_1pct": float(base_m["lift_1pct"]),
                    "perm_drop_pr_auc_mean": float(np.mean(d_aps)),
                    "perm_drop_pr_auc_sd": float(np.std(d_aps, ddof=1)),
                    "perm_drop_lift_1pct_mean": float(np.mean(d_l1s)),
                    "perm_drop_lift_1pct_sd": float(np.std(d_l1s, ddof=1)),
                    "perm_drop_roc_auc_mean": float(np.mean(d_aucs)),
                    "perm_drop_roc_auc_sd": float(np.std(d_aucs, ddof=1)),
                })
            print(f"[PERM {stage_name.upper():4s}] {eid:15s} | Completed {len(cols)} features x {n_repeats} repeats.", flush=True)

        del df_tr_all, df_ev_all
        gc.collect()

    del panel_df
    gc.collect()
    perm_df = pd.DataFrame(perm_rows).sort_values(["stage", "exp_id", "perm_drop_pr_auc_mean"], ascending=[True, True, False])
    perm_df.to_csv(METRICS_DIR / "permutation_importance_b_raw.csv", index=False)
    pd.DataFrame(shuf_results).to_csv(METRICS_DIR / "shuffled_label_sanity.csv", index=False)
    return {"shuffled_label_checks": shuf_results}


def compute_all_metrics_curves_slices_and_holm(spec: dict[str, Any]) -> dict[str, Any]:
    """Compute per-seed, per-fold vs pooled, curves (1..20%), slices, multi-seed bootstrap CIs & Holm p-values."""
    seeds = [int(s) for s in spec["protocol"]["seeds"]]
    n_boot = int(spec["protocol"]["bootstrap_replicates"])
    boot_seed = int(spec["protocol"]["bootstrap_seed"])
    curve_pcts = [int(k) for k in spec["protocol"]["curve_budgets_pct"]]
    curve_fracs = tuple(k / 100.0 for k in curve_pcts)
    model_list = spec["models"]
    model_map = {m["exp_id"]: m for m in model_list}

    per_seed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    monthly_test_rows: list[dict[str, Any]] = []
    primary_holm_rows: list[dict[str, Any]] = []
    secondary_decomp_rows: list[dict[str, Any]] = []
    aux_2x2_rows: list[dict[str, Any]] = []

    for stage_name in ("val", "test"):
        # Pre-load all (exp_id, seed) dataframes & build bootstrap prep structures once!
        ref_df = load_pred_for_seed(stage_name, model_map["L0_base_rate"], 42)
        codes, uniq = pd.factorize(ref_df["client_id"].to_numpy(), sort=True)
        n_clients = len(uniq)
        tie_hashes = deterministic_tie_hash(ref_df["client_id"].to_numpy(), seed=42)
        y_ref = ref_df["y_true"].to_numpy(dtype=np.int8)

        # Generate identical 1,000 client-cluster weight vectors for all models & pairs on this stage
        rng = np.random.default_rng(boot_seed)
        boot_weights = [np.bincount(rng.integers(0, n_clients, size=n_clients), minlength=n_clients) for _ in range(n_boot)]
        ones_w = np.ones(n_clients, dtype=np.int64)

        preps: dict[tuple[str, int], dict[str, Any]] = {}
        boot_samples_by_exp_seed: dict[tuple[str, int], dict[str, np.ndarray]] = {}

        for m in model_list:
            eid = m["exp_id"]
            is_prob = bool(m["is_probabilistic"])
            if eid in ("L3_logreg_rfm", "B-raw-T", "B-raw-TGD", "B-raw-TGD_logreg", "L4_cb", "FE-v1"):
                run_seeds = seeds
                for s in run_seeds:
                    req_p = PRED_DIR / stage_name / f"{eid}_seed{s}.parquet"
                    if not req_p.exists():
                        raise FileNotFoundError(f"Primary multi-seed model missing {req_p}")
            elif m["stochastic_across_seeds"]:
                run_seeds = [s for s in seeds if (PRED_DIR / stage_name / f"{eid}_seed{s}.parquet").exists()]
                if 42 not in run_seeds:
                    raise FileNotFoundError(f"Canonical seed=42 missing for intermediate model {eid} on {stage_name}")
            else:
                run_seeds = [42]

            seed_curve_lifts = {k: [] for k in curve_pcts}
            seed_curve_gains = {k: [] for k in curve_pcts}
            seed_curve_precs = {k: [] for k in curve_pcts}

            for s in run_seeds:
                pdf = load_pred_for_seed(stage_name, m, s)
                assert np.array_equal(pdf["client_id"].to_numpy(), ref_df["client_id"].to_numpy())
                assert np.array_equal(pdf["month"].to_numpy(), ref_df["month"].to_numpy())
                scores = pdf["score"].to_numpy(dtype=np.float64)

                # 1. Pooled OOF metrics including 1%..20% curve
                pm, _ = compute_metrics(
                    y_ref, scores, ref_df["client_id"].to_numpy(),
                    budget_fractions=curve_fracs, primary_budget=0.01,
                    seed=42, is_probabilistic=is_prob, tie_hashes=tie_hashes,
                )
                for k_pct in curve_pcts:
                    seed_curve_lifts[k_pct].append(float(pm[f"lift_{k_pct}pct"]))
                    seed_curve_gains[k_pct].append(float(pm[f"recall_{k_pct}pct"]))
                    seed_curve_precs[k_pct].append(float(pm[f"precision_{k_pct}pct"]))

                # 2. Per-fold metrics (fold == 0..4 separately, then mean & SD across 5 folds)
                f_aps, f_l1s, f_l5s, f_aucs = [], [], [], []
                fold_arr = pdf["fold"].to_numpy()
                for fk in range(5):
                    fmask = fold_arr == fk
                    fm, _ = compute_metrics(
                        y_ref[fmask], scores[fmask], ref_df["client_id"].to_numpy()[fmask],
                        budget_fractions=(0.01, 0.05, 0.10), primary_budget=0.01,
                        seed=42, is_probabilistic=is_prob, tie_hashes=tie_hashes[fmask],
                    )
                    f_aps.append(float(fm["average_precision"]))
                    f_l1s.append(float(fm["lift_1pct"]))
                    f_l5s.append(float(fm["lift_5pct"]))
                    f_aucs.append(float(fm["roc_auc"]))

                per_seed_rows.append({
                    "stage": stage_name,
                    "exp_id": eid,
                    "seed": s,
                    "n_features": m["n_features"],
                    "pooled_pr_auc": pm["average_precision"],
                    "pooled_roc_auc": pm["roc_auc"],
                    "pooled_lift_1pct": pm["lift_1pct"],
                    "pooled_recall_1pct": pm["recall_1pct"],
                    "pooled_precision_1pct": pm["precision_1pct"],
                    "pooled_hits_1pct": pm["hits_1pct"],
                    "pooled_lift_5pct": pm["lift_5pct"],
                    "pooled_recall_5pct": pm["recall_5pct"],
                    "pooled_lift_10pct": pm["lift_10pct"],
                    "pooled_recall_10pct": pm["recall_10pct"],
                    "pooled_log_loss": pm["log_loss"],
                    "pooled_brier": pm["brier"],
                    "fold5_mean_pr_auc": float(np.mean(f_aps)),
                    "fold5_sd_pr_auc": float(np.std(f_aps, ddof=1)),
                    "fold5_mean_lift_1pct": float(np.mean(f_l1s)),
                    "fold5_sd_lift_1pct": float(np.std(f_l1s, ddof=1)),
                    "fold5_mean_lift_5pct": float(np.mean(f_l5s)),
                    "fold5_mean_roc_auc": float(np.mean(f_aucs)),
                })

                # Monthly breakdown on Locked Test
                if stage_name == "test":
                    for em in spec["data_contract"]["test_months"]:
                        mmask = (pdf["month"] == em).to_numpy()
                        mm_m, _ = compute_metrics(
                            y_ref[mmask], scores[mmask], ref_df["client_id"].to_numpy()[mmask],
                            budget_fractions=(0.01, 0.05, 0.10), primary_budget=0.01,
                            seed=42, is_probabilistic=is_prob, tie_hashes=tie_hashes[mmask],
                        )
                        monthly_test_rows.append({
                            "month": em, "exp_id": eid, "seed": s,
                            "pr_auc": mm_m["average_precision"],
                            "roc_auc": mm_m["roc_auc"],
                            "lift_1pct": mm_m["lift_1pct"],
                            "hits_1pct": mm_m["hits_1pct"],
                            "lift_5pct": mm_m["lift_5pct"],
                            "recall_5pct": mm_m["recall_5pct"],
                        })

            for k_pct in curve_pcts:
                curve_rows.append({
                    "stage": stage_name,
                    "exp_id": eid,
                    "budget_pct": k_pct,
                    "lift_mean": float(np.mean(seed_curve_lifts[k_pct])),
                    "lift_sd": float(np.std(seed_curve_lifts[k_pct], ddof=1)) if len(run_seeds) > 1 else 0.0,
                    "gain_recall_mean": float(np.mean(seed_curve_gains[k_pct])),
                    "gain_recall_sd": float(np.std(seed_curve_gains[k_pct], ddof=1)) if len(run_seeds) > 1 else 0.0,
                    "precision_mean": float(np.mean(seed_curve_precs[k_pct])),
                })

            # Precompute bootstrap arrays for each distinct seed
            for s in run_seeds:
                pdf = load_pred_for_seed(stage_name, m, s)
                pr = _prep_sorted(y_ref, pdf["score"].to_numpy(dtype=np.float64), codes, tie_hashes, is_prob)
                preps[(eid, s)] = pr
                arrs = {k: np.empty(n_boot, dtype=np.float64) for k in BOOT_KEYS}
                for b_idx, cw in enumerate(boot_weights):
                    wm = _w_metrics_multi_k(pr, cw)
                    for k in BOOT_KEYS:
                        arrs[k][b_idx] = wm[k]
                boot_samples_by_exp_seed[(eid, s)] = arrs

            # Multi-seed summary + 95% Client-Clustered Bootstrap CI of seed-mean estimator
            sub_ps = [r for r in per_seed_rows if r["stage"] == stage_name and r["exp_id"] == eid]
            row_sum: dict[str, Any] = {
                "stage": stage_name,
                "exp_id": eid,
                "role": m["role"],
                "in_main_table": m["in_main_table"],
                "family": m["family"],
                "uses_target_lag": m["uses_target_lag"],
                "n_features": m["n_features"],
                "n_seeds": len(run_seeds),
                "fold5_mean_pr_auc": float(np.mean([r["fold5_mean_pr_auc"] for r in sub_ps])),
                "fold5_mean_lift_1pct": float(np.mean([r["fold5_mean_lift_1pct"] for r in sub_ps])),
                "hits_1pct_mean": float(np.mean([r["pooled_hits_1pct"] for r in sub_ps])),
            }
            for k in BOOT_KEYS:
                col_ps = f"pooled_{'pr_auc' if k == 'average_precision' else k}"
                vals_seed = [r[col_ps] for r in sub_ps if r[col_ps] is not None and np.isfinite(r[col_ps])]
                if not vals_seed:
                    row_sum[f"{k}_mean"] = None
                    row_sum[f"{k}_seed_sd"] = None
                    row_sum[f"{k}_ci_low"] = None
                    row_sum[f"{k}_ci_high"] = None
                    continue
                row_sum[f"{k}_mean"] = float(np.mean(vals_seed))
                row_sum[f"{k}_seed_sd"] = (
                    float(np.std(vals_seed, ddof=1))
                    if len(run_seeds) > 1
                    else (None if m["stochastic_across_seeds"] else 0.0)
                )
                mean_boot = np.mean([boot_samples_by_exp_seed[(eid, s)][k] for s in run_seeds], axis=0)
                valid_b = mean_boot[np.isfinite(mean_boot)]
                row_sum[f"{k}_ci_low"] = float(np.percentile(valid_b, 2.5)) if len(valid_b) else None
                row_sum[f"{k}_ci_high"] = float(np.percentile(valid_b, 97.5)) if len(valid_b) else None
            summary_rows.append(row_sum)

            # Slice analysis (Step 6.5) across all requested dimensions, averaged across seeds
            slice_defs = (
                ("trx_90d_active", "trx_90d_has_event"),
                ("has_dialog_90d", "dialog_90d_has_event"),
                ("has_dialog_all", "dialog_all_has_event"),
                ("has_geo_90d", "geo_90d_has_event"),
                ("has_geo_all", "geo_all_has_event"),
                ("cold_start", "cold_start"),
            )
            for sdim, scol in slice_defs:
                for sval in (0, 1):
                    smask = (ref_df[scol].to_numpy() == sval)
                    s_aps, s_l1s, s_l5s, s_r5s, s_aucs, s_hits = [], [], [], [], [], []
                    for s in run_seeds:
                        pdf = load_pred_for_seed(stage_name, m, s)
                        sm, _ = compute_metrics(
                            y_ref[smask],
                            pdf["score"].to_numpy(dtype=np.float64)[smask],
                            ref_df["client_id"].to_numpy()[smask],
                            budget_fractions=(0.01, 0.05, 0.10),
                            primary_budget=0.01,
                            seed=42,
                            is_probabilistic=is_prob,
                            tie_hashes=tie_hashes[smask],
                        )
                        s_aps.append(sm["average_precision"])
                        s_l1s.append(sm["lift_1pct"])
                        s_l5s.append(sm["lift_5pct"])
                        s_r5s.append(sm["recall_5pct"])
                        s_aucs.append(sm["roc_auc"])
                        s_hits.append(sm["hits_1pct"])
                    slice_rows.append({
                        "stage": stage_name,
                        "exp_id": eid,
                        "slice_dim": sdim,
                        "slice_col": scol,
                        "slice_val": sval,
                        "n": int(smask.sum()),
                        "positives": int(y_ref[smask].sum()),
                        "prevalence": float(y_ref[smask].mean()),
                        "pr_auc_mean": float(np.mean(s_aps)),
                        "pr_auc_sd": float(np.std(s_aps, ddof=1)) if len(s_aps) > 1 else 0.0,
                        "roc_auc_mean": float(np.mean(s_aucs)),
                        "lift_1pct_mean": float(np.mean(s_l1s)),
                        "hits_1pct_mean": float(np.mean(s_hits)),
                        "lift_5pct_mean": float(np.mean(s_l5s)),
                        "recall_5pct_mean": float(np.mean(s_r5s)),
                    })

        # Helper to evaluate a list of paired comparisons and apply Holm-Bonferroni
        def eval_pair_family(pair_specs: list[dict[str, Any]], apply_holm: bool, force_seed42: bool = False) -> list[dict[str, Any]]:
            fam_rows: list[dict[str, Any]] = []
            for pdef in pair_specs:
                aid = pdef["model_a"]
                bid = pdef["model_b"]
                ma, mb = model_map[aid], model_map[bid]
                if force_seed42:
                    sa_list = [42]
                    sb_list = [42]
                else:
                    sa_list = [s for s in seeds if (aid, s) in preps] or [42]
                    sb_list = [s for s in seeds if (bid, s) in preps] or [42]

                # Point estimate over seeds
                pts_a_s = [_w_metrics_multi_k(preps[(aid, s)], ones_w) for s in sa_list]
                pts_b_s = [_w_metrics_multi_k(preps[(bid, s)], ones_w) for s in sb_list]
                pt_a = {k: float(np.mean([d[k] for d in pts_a_s])) for k in BOOT_KEYS}
                pt_b = {k: float(np.mean([d[k] for d in pts_b_s])) for k in BOOT_KEYS}

                row_p: dict[str, Any] = {
                    "stage": stage_name,
                    "pair_id": pdef["pair_id"],
                    "model_b": bid,
                    "model_a": aid,
                    "question": pdef["question"],
                    "pr_auc_a": pt_a["average_precision"],
                    "pr_auc_b": pt_b["average_precision"],
                    "lift_1pct_a": pt_a["lift_1pct"],
                    "lift_1pct_b": pt_b["lift_1pct"],
                }

                for k in ("average_precision", "lift_1pct", "lift_5pct", "lift_10pct", "roc_auc"):
                    boot_a = np.mean([boot_samples_by_exp_seed[(aid, s)][k] for s in sa_list], axis=0)
                    boot_b = np.mean([boot_samples_by_exp_seed[(bid, s)][k] for s in sb_list], axis=0)
                    deltas = boot_b - boot_a
                    valid = deltas[np.isfinite(deltas)]
                    d_pt = pt_b[k] - pt_a[k]
                    rel_pct = (d_pt / pt_a[k] * 100.0) if (np.isfinite(pt_a[k]) and pt_a[k] > 0) else None
                    lo = float(np.percentile(valid, 2.5)) if len(valid) else None
                    hi = float(np.percentile(valid, 97.5)) if len(valid) else None
                    p_two = min(1.0, 2.0 * min(float(np.mean(valid <= 0.0)), float(np.mean(valid >= 0.0)))) if len(valid) else None
                    tag = "pr_auc" if k == "average_precision" else k
                    row_p[f"delta_{tag}"] = float(d_pt)
                    row_p[f"rel_delta_{tag}_pct"] = float(rel_pct) if rel_pct is not None else None
                    row_p[f"delta_{tag}_ci_low"] = lo
                    row_p[f"delta_{tag}_ci_high"] = hi
                    row_p[f"delta_{tag}_p_raw"] = p_two

                fam_rows.append(row_p)

            if apply_holm:
                for tag in ("pr_auc", "lift_1pct", "lift_5pct", "lift_10pct", "roc_auc"):
                    raw_ps = [r[f"delta_{tag}_p_raw"] for r in fam_rows]
                    adj_ps = holm_bonferroni(raw_ps)
                    for r, ap in zip(fam_rows, adj_ps):
                        r[f"delta_{tag}_p_holm"] = ap

                # Apply frozen SESOI + Holm decision logic from Step 0
                sesoi = spec["practical_significance_thresholds"]
                for r in fam_rows:
                    pass_sesoi_ap = (
                        (r["delta_pr_auc"] >= float(sesoi["min_absolute_delta_pr_auc"]))
                        and (r["rel_delta_pr_auc_pct"] is not None and r["rel_delta_pr_auc_pct"] >= float(sesoi["min_relative_delta_pr_auc"]) * 100.0)
                    )
                    pass_holm_ap = (r["delta_pr_auc_p_holm"] is not None and r["delta_pr_auc_p_holm"] < float(sesoi["alpha_holm"]))
                    pass_sesoi_l1 = (r["delta_lift_1pct"] >= float(sesoi["min_absolute_delta_lift_1pct"]))
                    pass_holm_l1 = (r["delta_lift_1pct_p_holm"] is not None and r["delta_lift_1pct_p_holm"] < float(sesoi["alpha_holm"]))
                    r["pass_sesoi_pr_auc"] = bool(pass_sesoi_ap)
                    r["pass_holm_pr_auc"] = bool(pass_holm_ap)
                    r["significant_and_practical_pr_auc"] = bool(pass_sesoi_ap and pass_holm_ap)
                    r["pass_sesoi_lift_1pct"] = bool(pass_sesoi_l1)
                    r["pass_holm_lift_1pct"] = bool(pass_holm_l1)
                    r["significant_and_practical_lift_1pct"] = bool(pass_sesoi_l1 and pass_holm_l1)
            return fam_rows

        primary_holm_rows.extend(eval_pair_family(spec["primary_holm_comparisons"], apply_holm=True, force_seed42=False))
        secondary_decomp_rows.extend(eval_pair_family(spec["secondary_conditional_decomposition"]["pairs"], apply_holm=True, force_seed42=True))
        aux_2x2_rows.extend(eval_pair_family(spec["auxiliary_2x2_comparisons"], apply_holm=False, force_seed42=False))

    pd.DataFrame(per_seed_rows).to_csv(METRICS_DIR / "per_seed_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(METRICS_DIR / "model_summary_multiseed.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(METRICS_DIR / "lift_gain_curve_1_to_20pct.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(METRICS_DIR / "slice_metrics.csv", index=False)
    pd.DataFrame(monthly_test_rows).to_csv(METRICS_DIR / "monthly_metrics_test.csv", index=False)
    pd.DataFrame(primary_holm_rows).to_csv(METRICS_DIR / "paired_holm_primary.csv", index=False)
    pd.DataFrame(secondary_decomp_rows).to_csv(METRICS_DIR / "paired_holm_secondary_decomposition.csv", index=False)
    pd.DataFrame(aux_2x2_rows).to_csv(METRICS_DIR / "paired_auxiliary_2x2.csv", index=False)

    return {
        "summary_df": pd.DataFrame(summary_rows),
        "primary_holm_df": pd.DataFrame(primary_holm_rows),
        "secondary_decomp_df": pd.DataFrame(secondary_decomp_rows),
        "curve_df": pd.DataFrame(curve_rows),
        "slice_df": pd.DataFrame(slice_rows),
    }


def generate_figures(
    summary_df: pd.DataFrame,
    primary_holm_df: pd.DataFrame,
    secondary_decomp_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    slice_df: pd.DataFrame,
) -> list[str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 9.5,
    })
    created: list[str] = []

    main_order = [
        "L0_base_rate",
        "L1_rule_target_lag",
        "L2_rule_trx_rfm",
        "L3_logreg_rfm",
        "B-raw-T",
        "L4_cb",
        "B-raw-TGD",
        "M_deriv_cb",
        "L5c_cb",
        "FE-v1",
    ]
    short_labels = {
        "L0_base_rate": "L0\n(Base)",
        "L1_rule_target_lag": "L1*\n(Rule TLag)",
        "L2_rule_trx_rfm": "L2\n(Rule TRX)",
        "L3_logreg_rfm": "L3\n(LogReg 8f)",
        "B-raw-T": "B-raw-T\n(CB 8f)",
        "L4_cb": "L4_cb\n(CB TRX 63f)",
        "B-raw-TGD": "B-raw-TGD\n(CB 14f)",
        "M_deriv_cb": "M_deriv\n(CB 100f)",
        "L5c_cb": "L5c_cb\n(+Geohash 112f)",
        "FE-v1": "FE-v1\n(+Emb 139f)",
    }

    # Figure 1: Ladder Progression (Test & Val) with 95% Bootstrap CI
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.8), sharex=True)
    bar_colors = [
        "#9e9e9e", "#8c564b", "#7f7f7f", "#4e79a7",
        "#59a14f", "#76b7b2", "#f28e2b", "#edc948", "#b07aa1", "#e15759"
    ]
    x = np.arange(len(main_order))
    for row_i, stg in enumerate(("test", "val")):
        sub = summary_df[summary_df["stage"] == stg].set_index("exp_id")
        stg_title = "Locked Test (2022-11..2023-01)" if stg == "test" else "Validation (2022-09..2022-10)"
        for col_i, (mkey, mlabel) in enumerate((("average_precision", "PR-AUC (Primary Metric)"), ("lift_1pct", "Lift@1% (Budget K=1%)"))):
            ax = axes[row_i, col_i]
            vals = [float(sub.loc[e, f"{mkey}_mean"]) for e in main_order]
            lo = [max(0.0, vals[i] - float(sub.loc[e, f"{mkey}_ci_low"])) for i, e in enumerate(main_order)]
            hi = [max(0.0, float(sub.loc[e, f"{mkey}_ci_high"]) - vals[i]) for i, e in enumerate(main_order)]
            bars = ax.bar(x, vals, color=bar_colors, alpha=0.9, yerr=[lo, hi], capsize=3, error_kw={"lw": 1.1})
            for b, v in zip(bars, vals):
                fmt = f"{v:.4f}" if mkey == "average_precision" else f"{v:.2f}x"
                ax.text(b.get_x() + b.get_width() / 2, v + (0.0022 if mkey == "average_precision" else 0.65),
                        fmt, ha="center", va="bottom", fontsize=7.5, fontweight="bold")
            ax.set_title(f"{stg_title} — {mlabel}", fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([short_labels[e] for e in main_order], fontsize=8)
            if mkey == "lift_1pct":
                ax.axhline(1.0, ls="--", color="gray", lw=1)
                ax.set_ylim(0, max(vals) * 1.25)
            else:
                ax.set_ylim(0, max(vals) * 1.28)
    fig.suptitle("Progression Across Raw Baselines, Intermediate Tiers & FE-v1 (Mean of 3 Seeds + 95% Client-Clustered CI)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    p1 = FIG_DIR / "01_ladder_progression_multiseed.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    created.append(str(p1))

    # Figure 2: Holm Forest Plot (7 Primary Pairs) + 3-Part FE-v1 Decomposition on Locked Test
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    sub_p = primary_holm_df[primary_holm_df["stage"] == "test"].copy().reset_index(drop=True)
    y_pos = np.arange(len(sub_p))
    ylabels = [f"{r['pair_id']}: {r['model_b']} vs {r['model_a']}" for _, r in sub_p.iterrows()]

    for ax, tag, title, sesoi_line in (
        (axes[0], "pr_auc", "Primary Pairs: ΔPR-AUC (Test, 95% CI)", 0.0010),
        (axes[1], "lift_1pct", "Primary Pairs: ΔLift@1% (Test, 95% CI)", 0.30),
    ):
        ax.axvline(0.0, color="black", lw=1, ls="--")
        ax.axvline(sesoi_line, color="#d62728", lw=1, ls=":", label=f"SESOI (+{sesoi_line})")
        for idx, r in sub_p.iterrows():
            v = float(r[f"delta_{tag}"])
            lo = max(0.0, v - float(r[f"delta_{tag}_ci_low"]))
            hi = max(0.0, float(r[f"delta_{tag}_ci_high"]) - v)
            sig = bool(r[f"pass_holm_{tag}"])
            c = "#1f77b4" if sig else "#999999"
            ax.errorbar(v, idx, xerr=[[lo], [hi]], fmt="o", color=c, capsize=3.5, lw=1.5)
            ph = r[f"delta_{tag}_p_holm"]
            p_str = f"p_H={ph:.3f}" if ph >= 0.001 else "p_H<0.001"
            ax.text(v, idx - 0.25, p_str, ha="center", va="bottom", fontsize=7.5, color=c, fontweight="bold" if sig else "normal")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ylabels if ax is axes[0] else [""] * len(y_pos), fontsize=8.5)
        ax.invert_yaxis()
        ax.set_title(title, fontweight="bold")
        ax.legend(frameon=False, loc="lower right", fontsize=8)

    # Panel 3: Decomposition D1 + D2 + D3 of C4 (FE-v1 - B-raw-TGD) on Test & Val
    ax_d = axes[2]
    d_test = secondary_decomp_df[secondary_decomp_df["stage"] == "test"].set_index("pair_id")
    d_val = secondary_decomp_df[secondary_decomp_df["stage"] == "val"].set_index("pair_id")
    d_ids = ["D1", "D2", "D3"]
    d_names = ["D1: Derived\nTabular (+86f)", "D2: Spatial\nGeohash (+12f)", "D3: Dialog Emb\nPCA/Cos (+27f)"]
    xd = np.arange(len(d_ids))
    w = 0.35
    ax_d.axhline(0.0, color="black", lw=1, ls="--")
    ax_d.bar(xd - w / 2, [d_test.loc[i, "delta_pr_auc"] for i in d_ids], width=w, color="#e15759", label="Locked Test")
    ax_d.bar(xd + w / 2, [d_val.loc[i, "delta_pr_auc"] for i in d_ids], width=w, color="#4e79a7", label="Validation")
    for idx, i in enumerate(d_ids):
        vt = float(d_test.loc[i, "delta_pr_auc"])
        ph = float(d_test.loc[i, "delta_pr_auc_p_holm"])
        ax_d.text(idx - w / 2, vt + (0.0002 if vt >= 0 else -0.0005), f"{vt:+.4f}\n(p_H={ph:.3f})", ha="center", fontsize=7.5)
    ax_d.set_xticks(xd)
    ax_d.set_xticklabels(d_names, fontsize=8.5)
    ax_d.set_ylabel("ΔPR-AUC")
    ax_d.set_title("Decomposing FE-v1 − B-raw-TGD (D1+D2+D3)", fontweight="bold")
    ax_d.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    p2 = FIG_DIR / "02_holm_forest_and_decomposition.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    created.append(str(p2))

    # Figure 3: Lift & Gain (Recall) Curves for K = 1%..20% on Locked Test & Validation
    curve_models = [
        ("FE-v1", "#e15759", "-", 2.3),
        ("L5c_cb", "#b07aa1", "--", 1.5),
        ("B-raw-TGD", "#f28e2b", "-", 2.0),
        ("L1_rule_target_lag", "#8c564b", "-.", 1.8),
        ("L4_cb", "#76b7b2", "-", 1.6),
        ("B-raw-T", "#59a14f", "--", 1.6),
        ("L3_logreg_rfm", "#4e79a7", ":", 1.6),
        ("L2_rule_trx_rfm", "#7f7f7f", ":", 1.4),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.8), sharex=True)
    for r_i, stg in enumerate(("test", "val")):
        stg_lbl = "Locked Test" if stg == "test" else "Validation"
        sub_c = curve_df[curve_df["stage"] == stg]
        ax_l, ax_g = axes[r_i, 0], axes[r_i, 1]
        for eid, col, ls, lw in curve_models:
            mc = sub_c[sub_c["exp_id"] == eid].sort_values("budget_pct")
            ax_l.plot(mc["budget_pct"], mc["lift_mean"], color=col, ls=ls, lw=lw, marker="o", ms=3, label=eid)
            ax_g.plot(mc["budget_pct"], mc["gain_recall_mean"] * 100.0, color=col, ls=ls, lw=lw, marker="o", ms=3, label=eid)
        ax_l.axhline(1.0, ls="--", color="gray", lw=1, label="Random (1.0x)")
        ax_g.plot([1, 20], [1, 20], ls="--", color="gray", lw=1, label="Random Gain")
        ax_l.set_title(f"{stg_lbl}: Lift@K Curve (K = 1%..20%)", fontweight="bold")
        ax_l.set_ylabel("Lift@K")
        ax_l.set_xticks(range(1, 21))
        ax_g.set_title(f"{stg_lbl}: Cumulative Gain / Recall@K Curve (K = 1%..20%)", fontweight="bold")
        ax_g.set_ylabel("Cumulative Gain / Recall (%)")
        ax_g.set_xticks(range(1, 21))
        if r_i == 0:
            ax_g.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False, fontsize=8.5)
    axes[1, 0].set_xlabel("Contact Budget K (% of Population)")
    axes[1, 1].set_xlabel("Contact Budget K (% of Population)")
    fig.tight_layout()
    p3 = FIG_DIR / "03_lift_and_gain_curves_1_to_20pct.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    created.append(str(p3))

    # Figure 4: Permutation Importance for B-raw-T and B-raw-TGD (Test)
    perm_df = pd.read_csv(METRICS_DIR / "permutation_importance_b_raw.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2))
    for ax, eid, color in ((axes[0], "B-raw-T", "#59a14f"), (axes[1], "B-raw-TGD", "#f28e2b")):
        sp = perm_df[(perm_df["stage"] == "test") & (perm_df["exp_id"] == eid)].sort_values("perm_drop_pr_auc_mean", ascending=True)
        y = np.arange(len(sp))
        ax.barh(y, sp["perm_drop_pr_auc_mean"], xerr=sp["perm_drop_pr_auc_sd"], color=color, alpha=0.9, capsize=3)
        ax.axvline(0.0, color="black", lw=0.9, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(sp["feature"], fontsize=8.5)
        ax.set_xlabel("Mean Drop in PR-AUC upon OOF Feature Permutation (± SD)")
        ax.set_title(f"OOF Permutation Importance on Test — {eid}", fontweight="bold")
    fig.tight_layout()
    p4 = FIG_DIR / "04_permutation_importance_b_raw.png"
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)
    created.append(str(p4))

    # Figure 5: Slice Diagnostics on Test (trx_90d_active, has_dialog_all, has_geo_all, cold_start)
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.2))
    s_models = ["L3_logreg_rfm", "B-raw-T", "L4_cb", "B-raw-TGD", "FE-v1"]
    s_colors = ["#4e79a7", "#59a14f", "#76b7b2", "#f28e2b", "#e15759"]
    dims = [
        ("trx_90d_active", "TRX Active in 90d (0 vs 1)"),
        ("has_dialog_all", "Has Prior Dialog All (0 vs 1)"),
        ("has_geo_all", "Has Prior Geo All (0 vs 1)"),
        ("cold_start", "Cold-Start All Sources=0 (0 vs 1)"),
    ]
    sub_s = slice_df[slice_df["stage"] == "test"]
    for ax, (sdim, stitle) in zip(axes.ravel(), dims):
        x_g = np.arange(2)
        bw = 0.15
        for m_i, (eid, col) in enumerate(zip(s_models, s_colors)):
            vals = [
                float(sub_s[(sub_s["exp_id"] == eid) & (sub_s["slice_dim"] == sdim) & (sub_s["slice_val"] == v)]["pr_auc_mean"].iloc[0])
                for v in (0, 1)
            ]
            ax.bar(x_g + (m_i - 2) * bw, vals, width=bw, color=col, label=eid)
        n0 = int(sub_s[(sub_s["exp_id"] == "FE-v1") & (sub_s["slice_dim"] == sdim) & (sub_s["slice_val"] == 0)]["n"].iloc[0])
        n1 = int(sub_s[(sub_s["exp_id"] == "FE-v1") & (sub_s["slice_dim"] == sdim) & (sub_s["slice_val"] == 1)]["n"].iloc[0])
        ax.set_xticks(x_g)
        ax.set_xticklabels([f"{sdim}=0\n(n={n0:,})", f"{sdim}=1\n(n={n1:,})"])
        ax.set_ylabel("PR-AUC (Mean across seeds)")
        ax.set_title(f"Locked Test Slice: {stitle}", fontweight="bold")
        if ax is axes[0, 1]:
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p5 = FIG_DIR / "05_slice_diagnostics.png"
    fig.savefig(p5, bbox_inches="tight")
    plt.close(fig)
    created.append(str(p5))
    return created


def write_final_manifest_and_quality_report(
    spec: dict[str, Any],
    freeze_lock: dict[str, Any],
    static_qc: dict[str, Any],
    feat_map: dict[str, tuple[str, ...]],
    shuf_qc: dict[str, Any],
    summary_df: pd.DataFrame,
    figures: list[str],
) -> None:
    # Re-verify spec hashes at the very end to prove zero post-hoc spec edits!
    end_lock = verify_or_create_spec_freeze_lock()
    assert end_lock["spec_json_sha256"] == freeze_lock["spec_json_sha256"]

    # Plausibility check (Step 6.3): B-raw-T vs L3 and L4_cb on Test
    tsum = summary_df[summary_df["stage"] == "test"].set_index("exp_id")
    braw_t_ap = float(tsum.loc["B-raw-T", "average_precision_mean"])
    l3_ap = float(tsum.loc["L3_logreg_rfm", "average_precision_mean"])
    l4_ap = float(tsum.loc["L4_cb", "average_precision_mean"])
    plausible = (0.015 <= braw_t_ap <= 0.030) and (abs(braw_t_ap - l3_ap) < 0.006) and (abs(braw_t_ap - l4_ap) < 0.006)
    if not plausible:
        raise AssertionError(
            f"Plausibility check FAILED: B-raw-T AP={braw_t_ap:.4f} vs L3={l3_ap:.4f}, L4_cb={l4_ap:.4f}"
        )

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    git_short = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()

    qc_full = {
        "status": "ALL_QUALITY_GATES_PASSED",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec_freeze_verification": end_lock,
        "static_schema_and_leakage_checks": static_qc,
        "shuffled_label_sanity_checks": shuf_qc["shuffled_label_checks"],
        "b_raw_t_plausibility_check": {
            "test_B_raw_T_pr_auc": braw_t_ap,
            "test_L3_pr_auc": l3_ap,
            "test_L4_cb_pr_auc": l4_ap,
            "passed": True,
        },
    }
    (OUT_DIR / "quality_checks.json").write_text(json.dumps(qc_full, indent=2, ensure_ascii=False), encoding="utf-8")

    emb_col_names = (
        tuple(f"dialog_all_pca_{i:02d}" for i in range(PCA_ALL_DIM))
        + tuple(f"dialog_90d_pca_{i:02d}" for i in range(PCA_90D_DIM))
        + ("dialog_all_emb_l2_norm", "dialog_90d_emb_l2_norm", "dialog_90d_vs_all_cosine_sim")
    )

    manifest = {
        "manifest_id": "fe_v1_ablation_manifest_v1",
        "status": "ready",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_version": {
            "git_commit": git_commit,
            "git_short": git_short,
            "script": "src/scripts/run_fe_v1_ablation.py",
            "python": sys.version.split()[0],
            "catboost": catboost.__version__,
            "lightgbm": lightgbm.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "frozen_spec_lock": end_lock,
        "post_selection_disclosure": spec["design_decisions_locked"]["post_selection_disclosure"],
        "tier_renaming": {
            "FE-v1": "Renamed from L5d_cb_dialog_emb (seed=42 reused directly without re-training)",
            "L4_cb": "Retained intermediate tier (L4_cb_trx)",
            "L5c_cb": "Retained intermediate tier (L5c_cb_dialog_meta)",
            "next_version": "FE-v2",
        },
        "protocol": spec["protocol"],
        "models_and_features": {
            m["exp_id"]: {
                "role": m["role"],
                "legacy_exp_id": m["legacy_exp_id"],
                "family": m["family"],
                "uses_target_lag": m["uses_target_lag"],
                "uses_emb": m["uses_emb"],
                "n_features": m["n_features"],
                "seeds_evaluated": spec["protocol"]["seeds"] if m["stochastic_across_seeds"] else [42],
                "features": list(feat_map[m["exp_id"]] + (emb_col_names if m["uses_emb"] else ())),
            }
            for m in spec["models"]
        },
        "outputs": {
            "quality_checks": "outputs/fe_v1_ablation/quality_checks.json",
            "per_seed_metrics": "outputs/fe_v1_ablation/metrics/per_seed_metrics.csv",
            "model_summary_multiseed": "outputs/fe_v1_ablation/metrics/model_summary_multiseed.csv",
            "paired_holm_primary": "outputs/fe_v1_ablation/metrics/paired_holm_primary.csv",
            "paired_holm_secondary_decomposition": "outputs/fe_v1_ablation/metrics/paired_holm_secondary_decomposition.csv",
            "paired_auxiliary_2x2": "outputs/fe_v1_ablation/metrics/paired_auxiliary_2x2.csv",
            "lift_gain_curve_1_to_20pct": "outputs/fe_v1_ablation/metrics/lift_gain_curve_1_to_20pct.csv",
            "permutation_importance_b_raw": "outputs/fe_v1_ablation/metrics/permutation_importance_b_raw.csv",
            "slice_metrics": "outputs/fe_v1_ablation/metrics/slice_metrics.csv",
            "monthly_metrics_test": "outputs/fe_v1_ablation/metrics/monthly_metrics_test.csv",
            "figures": [str(Path(p).relative_to(REPO_ROOT).as_posix()) for p in figures],
        },
    }
    (OUT_DIR / "fe_v1_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[DONE] Manifest & quality report written to {OUT_DIR}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FE-v1 & B-raw multi-seed ablation protocol.")
    parser.add_argument(
        "--phase",
        choices=("prep_and_fast", "pca_only", "run_model", "diagnostics_and_stats", "all"),
        default="all",
    )
    parser.add_argument("--exp-id", type=str, default=None, help="Target exp_id for --phase run_model")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds for --phase run_model")
    args = parser.parse_args()

    # Always verify Step 0 spec freeze lock FIRST!
    freeze_lock = verify_or_create_spec_freeze_lock()
    spec = load_spec()
    panel_cols = pq.read_schema(PANEL_PATH).names
    feat_map = resolve_model_features(spec, panel_cols)
    static_qc = run_static_quality_gates(spec, panel_cols, feat_map)
    print(
        f"[STEP 0 LOCK OK] spec_json_sha256={freeze_lock['spec_json_sha256'][:16]}... | "
        f"Static quality assertions PASSED across {len(spec['models'])} models | start RSS={get_rss_mb():.1f} MB",
        flush=True,
    )

    if args.phase == "pca_only":
        print("\n=== STAGE 0: PRE-COMPUTING FROZEN DIALOG PCA (BOUNDED RAM) ===", flush=True)
        ensure_stage0_pca_cached(spec)
        return

    if args.phase == "run_model":
        if not args.exp_id:
            raise ValueError("--exp-id is required when --phase run_model")
        seeds_override = [int(x.strip()) for x in args.seeds.split(",") if x.strip()] if args.seeds else None
        print(f"\n=== RUN MODEL: {args.exp_id} (seeds={seeds_override or spec['protocol']['seeds']}) ===", flush=True)
        run_training_wave(spec, feat_map, (args.exp_id,), seeds_override=seeds_override)
        return

    if args.phase in ("prep_and_fast", "all"):
        print("\n=== PHASE 1: STEP 2 RENAMING + FAST MULTI-SEED MODELS (L3, B-raw-TGD_logreg, B-raw-T, B-raw-TGD) ===", flush=True)
        panel_meta = pd.read_parquet(PANEL_PATH, columns=SLICE_META_COLS)
        alias_info = apply_step2_renaming_and_seed42_aliases(spec, panel_meta)
        del panel_meta
        gc.collect()
        print(f"[STEP 2 OK] Aliased {len(alias_info['seed42_aliased_files'])} canonical seed=42 prediction files.", flush=True)
        run_training_wave(spec, feat_map, ("L3_logreg_rfm", "B-raw-TGD_logreg", "B-raw-T", "B-raw-TGD"))
        if args.phase == "prep_and_fast":
            return

    if args.phase == "all":
        print("\n=== PHASE 2: PRIMARY TABLE MULTI-SEED MODELS (L4_cb, FE-v1) ===", flush=True)
        run_training_wave(spec, feat_map, ("L4_cb", "FE-v1"))
        print("\n=== PHASE 3: SECONDARY DECOMPOSITION BRIDGE (M_deriv_cb seed=42) ===", flush=True)
        run_training_wave(spec, feat_map, ("M_deriv_cb",), seeds_override=[42])

    if args.phase in ("diagnostics_and_stats", "all"):
        print("\n=== PHASE 4: QUALITY DIAGNOSTICS, PERMUTATION IMPORTANCE, BOOTSTRAP & HOLM ===", flush=True)
        shuf_qc = run_shuffled_and_permutation_checks(spec, feat_map)
        shutil.rmtree(PCA_CACHE_DIR, ignore_errors=True)

        res = compute_all_metrics_curves_slices_and_holm(spec)
        figs = generate_figures(
            res["summary_df"],
            res["primary_holm_df"],
            res["secondary_decomp_df"],
            res["curve_df"],
            res["slice_df"],
        )
        write_final_manifest_and_quality_report(
            spec=spec,
            freeze_lock=freeze_lock,
            static_qc=static_qc,
            feat_map=feat_map,
            shuf_qc=shuf_qc,
            summary_df=res["summary_df"],
            figures=figs,
        )


if __name__ == "__main__":
    main()
