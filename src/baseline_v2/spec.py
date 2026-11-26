"""Frozen spec loading, SHA-256 integrity verification, and leakage guards."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_JSON = REPO_ROOT / "configs" / "baseline_v2_evaluation_spec.json"
DEFAULT_SPEC_MD = REPO_ROOT / "docs" / "BASELINE_V2_FROZEN_SPEC.md"

DERIVED_COLUMN_TRANSFORMS: dict[str, tuple[str, float]] = {
    "trx_all_days_since_first_event__cap365": ("trx_all_days_since_first_event", 365.0),
}


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_spec(spec_path: Path | None = None) -> dict[str, Any]:
    """Load and validate the frozen Baseline v2 JSON specification."""
    p = Path(spec_path) if spec_path is not None else DEFAULT_SPEC_JSON
    spec = json.loads(p.read_text(encoding="utf-8"))
    if not spec.get("frozen_before_execution", False):
        raise ValueError(f"Spec at {p} must have frozen_before_execution=true")

    # Validate every declared feature set and model
    feat_sets: dict[str, list[str]] = spec["feature_sets"]
    for m in spec["models"]:
        cols = feat_sets[m["feature_set"]]
        validate_feature_set(
            exp_id=m["exp_id"],
            feature_cols=cols,
            uses_target_lag=bool(m["uses_target_lag"]),
            spec=spec,
        )
    return spec


def verify_or_create_freeze_lock(
    out_dir: Path,
    spec_json_path: Path | None = None,
    spec_md_path: Path | None = None,
) -> dict[str, Any]:
    """Create or verify SHA-256 freeze lock so the protocol cannot drift after execution."""
    jpath = Path(spec_json_path) if spec_json_path is not None else DEFAULT_SPEC_JSON
    mpath = Path(spec_md_path) if spec_md_path is not None else DEFAULT_SPEC_MD
    if not jpath.exists() or not mpath.exists():
        raise FileNotFoundError(f"Frozen spec files missing: {jpath}, {mpath}")

    cur_json_hash = sha256_file(jpath)
    cur_md_hash = sha256_file(mpath)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / "spec_freeze_lock.json"

    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("spec_json_sha256") != cur_json_hash:
            raise AssertionError(
                f"VIOLATION: {jpath} modified after freeze lock! "
                f"Expected {lock.get('spec_json_sha256')}, got {cur_json_hash}"
            )
        if lock.get("spec_md_sha256") != cur_md_hash:
            raise AssertionError(
                f"VIOLATION: {mpath} modified after freeze lock! "
                f"Expected {lock.get('spec_md_sha256')}, got {cur_md_hash}"
            )
        return lock

    spec = load_spec(jpath)
    lock = {
        "spec_id": spec["spec_id"],
        "spec_version": spec["spec_version"],
        "spec_json_sha256": cur_json_hash,
        "spec_md_sha256": cur_md_hash,
    }
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return lock


def resolve_raw_panel_columns(feature_cols: list[str] | tuple[str, ...]) -> list[str]:
    """Map virtual/capped feature names back to physical columns required in panel_moscow.parquet."""
    raw_cols: list[str] = []
    for c in feature_cols:
        src = DERIVED_COLUMN_TRANSFORMS[c][0] if c in DERIVED_COLUMN_TRANSFORMS else c
        if src not in raw_cols:
            raw_cols.append(src)
    return raw_cols


def validate_feature_set(
    exp_id: str,
    feature_cols: list[str] | tuple[str, ...],
    uses_target_lag: bool,
    spec: dict[str, Any],
) -> None:
    """Enforce zero forbidden columns, zero duplicates, and target-lag isolation."""
    if len(feature_cols) != len(set(feature_cols)):
        raise ValueError(f"[{exp_id}] Duplicate columns in feature set")

    forbidden = set(spec["data_contract"]["forbidden_cols"])
    lag_prefix = str(spec["data_contract"]["target_lag_prefix"])
    for c in feature_cols:
        base_c = DERIVED_COLUMN_TRANSFORMS[c][0] if c in DERIVED_COLUMN_TRANSFORMS else c
        if base_c in forbidden or c in forbidden:
            raise ValueError(f"[{exp_id}] Forbidden column in predictors: {c}")
        if not uses_target_lag and (base_c.startswith(lag_prefix) or c.startswith(lag_prefix)):
            raise ValueError(f"[{exp_id}] Target-lag column '{c}' not allowed in behavioral model")


def extract_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str] | tuple[str, ...],
) -> np.ndarray:
    """Build float32 predictor matrix preserving semantic NaNs and applying any capped transforms."""
    n = len(df)
    p = len(feature_cols)
    if p == 0:
        return np.empty((n, 0), dtype=np.float32)

    X = np.empty((n, p), dtype=np.float32)
    for j, col in enumerate(feature_cols):
        if col in DERIVED_COLUMN_TRANSFORMS:
            src_col, cap_val = DERIVED_COLUMN_TRANSFORMS[col]
            arr = df[src_col].to_numpy(dtype=np.float32, copy=False)
            # np.minimum preserves NaNs cleanly
            X[:, j] = np.minimum(arr, np.float32(cap_val))
        else:
            X[:, j] = df[col].to_numpy(dtype=np.float32, copy=False)
    return X
