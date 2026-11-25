"""Contract validation, feature-set whitelisting, and memory-safe fold loading."""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .config import assert_temporal_embargo
from .features import (
    DERIVED_BEHAVIORAL_FEATURES,
    PRIOR_PURCHASE_FEATURES,
    TOP50_FEATURES,
    TOP50_NO_PRIOR_FEATURES,
    TOP50_TRX_FEATURES,
)

FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    {
        "client_id",
        "cutoff",
        "month",
        "y_any",
        "y_h1",
        "y_h3",
        "target_1",
        "target_2",
        "target_3",
        "target_4",
        "supervised_eligible",
        "coverage_slice",
        "fold",
        "trans_count",
        "diff_trans_date",
    }
)


@dataclass
class DatasetContract:
    processed_dir: Path
    parquet_path: Path
    manifest: dict[str, Any]
    schema: dict[str, Any]
    protocol: dict[str, Any]


@dataclass
class FoldData:
    X_train: np.ndarray
    y_train: np.ndarray
    train_cold_start: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray
    eval_meta: pd.DataFrame
    feature_names: tuple[str, ...]


def load_dataset_contract(processed_dir: Path) -> DatasetContract:
    root = Path(processed_dir).resolve()
    manifest_path = root / "manifest.json"
    schema_path = root / "feature_schema.json"
    protocol_path = root / "protocol.json"
    parquet_path = root / "propensity.parquet"
    for p in (manifest_path, schema_path, protocol_path, parquet_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required processed artifact: {p}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ready":
        raise ValueError(f"Processed dataset manifest is not ready: {manifest.get('status')}")
    if manifest.get("schema_version") != 2:
        raise ValueError(f"Expected schema_version == 2, got {manifest.get('schema_version')}")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    return DatasetContract(
        processed_dir=root,
        parquet_path=parquet_path,
        manifest=manifest,
        schema=schema,
        protocol=protocol,
    )


def resolve_feature_sets(schema: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """Build whitelisted feature sets for both Top-50 and Full-252 suites."""
    ablation = schema["ablation_groups"]
    trx = tuple(ablation["trx"])
    trx_84d = tuple(c for c in trx if c.startswith("trx_84d_"))
    source_meta = tuple(ablation["source_metadata"])
    geo_content = tuple(ablation["geo_content"])
    cohort_flags = tuple(ablation["cohort_flags"])

    all_221 = tuple(schema["predictors"])
    all_prior = all_221 + PRIOR_PURCHASE_FEATURES
    all_prior_derived = all_prior + DERIVED_BEHAVIORAL_FEATURES

    feature_sets: dict[str, tuple[str, ...]] = {
        "none": (),
        "heuristic_rf": ("trx_84d_event_count", "trx_84d_days_since_last_event"),
        "trx_84d": trx_84d,
        "trx": trx,
        "trx_meta": trx + source_meta,
        "all_221": all_221,
        "all_prior": all_prior,
        "all_prior_derived": all_prior_derived,
        "top50_trx_only": TOP50_TRX_FEATURES,
        "top50_no_prior": TOP50_NO_PRIOR_FEATURES,
        "top50": TOP50_FEATURES,
    }

    all_valid = set(all_prior_derived)
    for name, cols in feature_sets.items():
        if len(cols) != len(set(cols)):
            raise ValueError(f"Duplicate columns in feature set {name}")
        forbidden = FORBIDDEN_COLUMNS.intersection(cols)
        if forbidden:
            raise ValueError(f"Feature set {name} contains forbidden columns: {sorted(forbidden)}")
        unknown = set(cols) - all_valid
        if unknown:
            raise ValueError(f"Feature set {name} contains unknown columns: {sorted(unknown)}")
    return feature_sets


def _sql_lit(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def load_fold_arrays(
    enriched_parquet: Path,
    train_months: tuple[str, ...] | list[str],
    eval_month: str,
    horizon: int,
    feature_cols: tuple[str, ...] | list[str],
    burn_in_months: tuple[str, ...] | list[str],
    memory_limit: str = "2GB",
    threads: int = 2,
) -> FoldData:
    """Load strictly validated train/eval float32 arrays via DuckDB projection."""
    assert_temporal_embargo(train_months, eval_month, horizon, burn_in_months)
    cols = tuple(feature_cols)
    forbidden = FORBIDDEN_COLUMNS.intersection(cols)
    if forbidden:
        raise ValueError(f"Refusing to load forbidden columns as predictors: {sorted(forbidden)}")

    target_col = "y_h1" if horizon == 1 else "y_h3"
    pq_lit = _sql_lit(enriched_parquet.as_posix())
    train_sql = ", ".join(_sql_lit(m) for m in train_months)
    eval_sql = _sql_lit(eval_month)

    feat_select = ", ".join(f'CAST("{c}" AS FLOAT) AS "{c}"' for c in cols)
    feat_prefix = f"{feat_select}, " if feat_select else ""

    with duckdb.connect() as con:
        con.execute(f"SET memory_limit={_sql_lit(memory_limit)}")
        con.execute(f"SET threads={int(threads)}")

        train_df = con.execute(
            f"""
            SELECT {feat_prefix}
                   CAST({target_col} AS TINYINT) AS _y,
                   CAST(cold_start AS TINYINT) AS _cold
            FROM read_parquet({pq_lit}, hive_partitioning=false)
            WHERE month IN ({train_sql})
              AND supervised_eligible = TRUE
              AND {target_col} IS NOT NULL
            ORDER BY cutoff, client_id
            """
        ).fetchdf()

        eval_df = con.execute(
            f"""
            SELECT {feat_prefix}
                   CAST({target_col} AS TINYINT) AS _y,
                   client_id,
                   month,
                   CAST(cold_start AS TINYINT) AS cold_start,
                   CAST(prior_all_bought_any AS TINYINT) AS prior_all_bought_any,
                   coverage_slice
            FROM read_parquet({pq_lit}, hive_partitioning=false)
            WHERE month = {eval_sql}
              AND supervised_eligible = TRUE
              AND {target_col} IS NOT NULL
            ORDER BY client_id
            """
        ).fetchdf()

    if len(train_df) == 0:
        raise ValueError(f"No labelled training rows for {train_months} (H={horizon})")
    if len(eval_df) == 0:
        raise ValueError(f"No labelled evaluation rows for {eval_month} (H={horizon})")

    if cols:
        X_train = np.ascontiguousarray(train_df[list(cols)].to_numpy(dtype=np.float32, copy=False))
        X_eval = np.ascontiguousarray(eval_df[list(cols)].to_numpy(dtype=np.float32, copy=False))
    else:
        X_train = np.empty((len(train_df), 0), dtype=np.float32)
        X_eval = np.empty((len(eval_df), 0), dtype=np.float32)

    y_train = train_df["_y"].to_numpy(dtype=np.int8, copy=True)
    train_cold = train_df["_cold"].to_numpy(dtype=np.int8, copy=True)
    y_eval = eval_df["_y"].to_numpy(dtype=np.int8, copy=True)
    eval_meta = eval_df[["client_id", "month", "cold_start", "prior_all_bought_any", "coverage_slice"]].copy()

    return FoldData(
        X_train=X_train,
        y_train=y_train,
        train_cold_start=train_cold,
        X_eval=X_eval,
        y_eval=y_eval,
        eval_meta=eval_meta,
        feature_names=cols,
    )
