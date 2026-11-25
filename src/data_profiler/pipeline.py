"""Dataset-agnostic 7-section descriptive profiling pipeline (single-table and multi-table)."""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
import platform
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from ._compat import load_dataframe
from .engine import SQLiteProfileEngine
from .spec import (
    DatasetSpec,
    ProfileResult,
    ProfilerConfig,
    SectionResult,
    TableSpec,
    auto_infer_table_spec,
)
from .streaming import grouped_target_rates, numeric_summary


def _column_role(spec: TableSpec, col: str) -> str:
    if col == spec.entity_key:
        return "entity_key"
    if col in spec.target_columns:
        return "target"
    if col == spec.time_column:
        return "datetime"
    if col in spec.numeric_columns:
        return "numeric"
    if col in spec.categorical_columns:
        return "categorical"
    if col in spec.vector_columns:
        return "vector"
    if col in spec.group_columns:
        return "group"
    return "unassigned"


def profile_dataset(
    dataset: DatasetSpec,
    config: ProfilerConfig | None = None,
    *,
    progress=None,
) -> ProfileResult:
    """Run single-pass descriptive profiling over any single-table or multi-table DatasetSpec."""
    started = perf_counter()
    cfg = config or ProfilerConfig()
    tables_map = dataset.table_map()

    # Load reference tables (entity & target) if declared
    ref_frames: dict[str, pd.DataFrame] = {}
    entity_ids: pd.Index | None = None
    entity_col: str | None = None

    if dataset.entity_table is not None:
        ent_spec = tables_map[dataset.entity_table]
        ent_df = load_dataframe(ent_spec.source, **ent_spec.source_options)
        entity_col = ent_spec.entity_key or (ent_spec.unique_key[0] if ent_spec.unique_key else ent_df.columns[0])
        ent_df[entity_col] = ent_df[entity_col].astype("string")
        ref_frames[ent_spec.name] = ent_df
        entity_ids = pd.Index(ent_df[entity_col].dropna().unique())

    if dataset.target_table is not None and dataset.target_table != dataset.entity_table:
        tgt_spec = tables_map[dataset.target_table]
        tgt_df = load_dataframe(tgt_spec.source, **tgt_spec.source_options)
        if tgt_spec.entity_key and tgt_spec.entity_key in tgt_df.columns:
            tgt_df[tgt_spec.entity_key] = tgt_df[tgt_spec.entity_key].astype("string")
        ref_frames[tgt_spec.name] = tgt_df

    scope = "prefix" if cfg.max_batches_per_source is not None else "full"
    tz_label = cfg.timezone or dataset.timezone or "naive_as_stored"
    sections: dict[str, SectionResult] = {}
    erd_edges: list[dict[str, str]] = []

    with SQLiteProfileEngine(dataset, cfg, entity_ids=entity_ids, progress=progress) as eng:
        eng.scan()

        # 1. Structure
        inv_rows, role_rows = [], []
        for spec in dataset.tables:
            if spec.name in ref_frames:
                frame = ref_frames[spec.name]
                row_count = len(frame)
                cols_info = {c: str(frame[c].dtype) for c in frame.columns}
            else:
                st = eng.stats[spec.name]
                row_count = st.rows
                cols_info = {c: ",".join(sorted(dt)) for c, dt in st.dtypes.items()}
            inv_rows.append(dict(source=spec.name, role=spec.role, row_unit=spec.row_unit,
                                 rows=row_count, columns=len(cols_info)))
            for col, dtype in cols_info.items():
                role_rows.append(dict(source=spec.name, column=col, dtype=dtype, role=_column_role(spec, col)))
            if dataset.entity_table and spec.name != dataset.entity_table and spec.entity_key:
                erd_edges.append({"left": dataset.entity_table, "right": spec.name,
                                  "key": spec.entity_key, "cardinality": "one_to_many"})

        sections["structure"] = SectionResult(
            section="structure",
            tables={
                "table_inventory": pd.DataFrame(inv_rows),
                "column_roles": pd.DataFrame(role_rows),
                "relationships": pd.DataFrame(erd_edges, columns=["left", "right", "key", "cardinality"]),
            },
            metadata={"scope": scope, "timezone": tz_label},
        )

        # 2. Quality
        missing_rows, invalid_rows, dup_rows, cov_rows = [], [], [], []
        for name, frame in ref_frames.items():
            spec = tables_map[name]
            for col in frame.columns:
                n = int(frame[col].isna().sum())
                missing_rows.append(dict(source=name, column=col, rows=len(frame), missing_n=n,
                                         missing_rate=n / len(frame) if len(frame) else np.nan))
            if spec.unique_key:
                dup_rows.append(dict(source=name, kind="unique_key",
                                     duplicate_n=int(frame.duplicated(list(spec.unique_key)).sum()),
                                     analyzed_rows=len(frame), scope="full"))

        aggregates = eng.entity_aggregates()
        for spec in dataset.event_tables():
            st = eng.stats[spec.name]
            for col, n in st.missing.items():
                missing_rows.append(dict(source=spec.name, column=col, rows=st.rows, missing_n=n,
                                         missing_rate=n / st.rows if st.rows else np.nan))
            for chk, n in st.invalid.items():
                invalid_rows.append(dict(source=spec.name, check=chk, n=int(n)))
            dup_rows.append(dict(source=spec.name, kind="full_row", duplicate_n=st.duplicate_n,
                                 analyzed_rows=st.duplicate_rows, scope=st.duplicate_scope))
            if entity_ids is not None and spec.entity_key:
                seen_ids = aggregates.loc[aggregates["source"].eq(spec.name), "entity_id"]
                matched = int(entity_ids.isin(seen_ids).sum())
                cov_rows.append(dict(
                    source=spec.name,
                    reference_entities=len(entity_ids),
                    matched_entities=matched,
                    distinct_seen_entities=len(seen_ids),
                    unmatched_entities=len(entity_ids) - matched,
                    coverage=matched / len(entity_ids) if len(entity_ids) else np.nan,
                    unknown_entity_rows=int(st.invalid.get("unknown_entity_rows", 0)),
                    missing_entity_rows=int(st.missing.get(spec.entity_key, 0)),
                ))

        sections["quality"] = SectionResult(
            section="quality",
            tables={
                "missing_values": pd.DataFrame(missing_rows),
                "invalid_values": pd.DataFrame(invalid_rows, columns=["source", "check", "n"]),
                "duplicates": pd.DataFrame(dup_rows),
                "key_coverage": pd.DataFrame(cov_rows),
            },
            metadata={"scope": scope, "duplicate_mode": cfg.duplicate_mode},
        )

        # 3. Temporal
        range_rows, hist_rows = [], []
        for spec in dataset.event_tables():
            if not spec.time_column:
                continue
            st = eng.stats[spec.name]
            valid_time_rows = st.rows - st.missing.get(spec.time_column, 0) - st.invalid.get("invalid_time", 0)
            range_rows.append(dict(source=spec.name, time_column=spec.time_column,
                                   first_event=st.start, last_event=st.end,
                                   scanned_rows=st.rows, valid_time_rows=valid_time_rows))
        monthly = eng.query(
            """SELECT source, substr(day, 1, 7) AS month, SUM(event_n) AS events,
                      COUNT(DISTINCT entity_id) AS entities, COUNT(DISTINCT day) AS observed_days
               FROM daily WHERE day <> '__NO_TIME__'
               GROUP BY source, substr(day, 1, 7) ORDER BY source, month"""
        )
        if not monthly.empty:
            monthly["events_per_active_entity"] = monthly["events"] / monthly["entities"]
        if not aggregates.empty:
            timed = aggregates[aggregates["first_ns"].notna() & aggregates["last_ns"].notna()]
            for src, grp in timed.groupby("source"):
                span = (grp["last_ns"] - grp["first_ns"]) / 86400e9
                hist_rows.append(numeric_summary(span, source=src, metric="span_days"))
                hist_rows.append(numeric_summary(grp["active_days"], source=src, metric="active_days"))
        sections["temporal"] = SectionResult(
            section="temporal",
            tables={
                "time_ranges": pd.DataFrame(range_rows),
                "monthly_activity": monthly,
                "history_length_summary": pd.DataFrame(hist_rows),
            },
            metadata={"scope": scope, "timezone": tz_label},
        )

        # 4. Numeric distributions (exact moments + sampled quantiles/IQR/histograms)
        num_rows, outlier_rows, hist_bins = [], [], []
        for spec in dataset.event_tables():
            st = eng.stats[spec.name]
            sample = st.sample.frame
            q_scope = "full_scanned_rows" if len(sample) == st.rows else "uniform_row_sample"
            for col in spec.numeric_columns:
                if col not in st.numeric_moments:
                    continue
                vals = pd.to_numeric(sample.get(col, pd.Series(dtype=float)), errors="coerce")
                arr = vals.to_numpy(dtype=float, na_value=np.nan)
                clean = pd.Series(arr[np.isfinite(arr)], dtype=float)
                if len(clean):
                    q = clean.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
                    iqr = float(q.loc[0.75] - q.loc[0.25])
                    lo, hi = float(q.loc[0.25] - 1.5 * iqr), float(q.loc[0.75] + 1.5 * iqr)
                    inside = clean[clean.between(lo, hi)]
                    whisker_lo = float(inside.min()) if len(inside) else np.nan
                    whisker_hi = float(inside.max()) if len(inside) else np.nan
                    out_n = int(((clean < lo) | (clean > hi)).sum())
                    counts, edges = np.histogram(clean, bins=min(30, max(1, int(np.sqrt(len(clean))))))
                    for left, right, cnt in zip(edges[:-1], edges[1:], counts):
                        hist_bins.append(dict(source=spec.name, column=col, left=float(left),
                                              right=float(right), n=int(cnt), scope=q_scope))
                    zeros_n = int((clean == 0).sum())
                    neg_n = int((clean < 0).sum())
                    q_dict = {
                        "zeros_n": zeros_n,
                        "zeros_pct": float(zeros_n / len(clean) * 100.0),
                        "negative_n": neg_n,
                        "skewness": float(clean.skew()) if len(clean) > 2 else np.nan,
                        "kurtosis": float(clean.kurt()) if len(clean) > 3 else np.nan,
                        "p01": float(q.loc[0.01]),
                        "p05": float(q.loc[0.05]),
                        "p25": float(q.loc[0.25]),
                        "median": float(q.loc[0.5]),
                        "p75": float(q.loc[0.75]),
                        "iqr": iqr,
                        "p95": float(q.loc[0.95]),
                        "p99": float(q.loc[0.99]),
                        "whisker_low": whisker_lo,
                        "whisker_high": whisker_hi,
                    }
                else:
                    lo = hi = np.nan
                    out_n = 0
                    q_dict = {
                        "zeros_n": 0, "zeros_pct": np.nan, "negative_n": 0,
                        **{k: np.nan for k in ("skewness", "kurtosis", "p01", "p05", "p25", "median",
                                               "p75", "iqr", "p95", "p99", "whisker_low", "whisker_high")},
                    }
                num_rows.append({
                    "source": spec.name,
                    "column": col,
                    **st.numeric_moments[col].summary(),
                    "scanned_rows": st.rows,
                    "sample_rows": len(sample),
                    "sample_finite_n": len(clean),
                    "missing_n": int(st.missing.get(col, 0)),
                    "infinite_n": int(st.invalid.get(f"infinite_{col}", 0)),
                    **q_dict,
                    "moments_scope": "all_scanned_finite",
                    "quantile_scope": q_scope,
                })
                outlier_rows.append(dict(source=spec.name, column=col, iqr_lower=lo, iqr_upper=hi,
                                         outlier_n=out_n, analyzed_finite_n=len(clean), scope=q_scope))

        sections["numeric"] = SectionResult(
            section="numeric",
            tables={
                "numeric_summary": pd.DataFrame(num_rows),
                "iqr_outliers": pd.DataFrame(outlier_rows),
                "histograms": pd.DataFrame(hist_bins),
            },
            metadata={"scope": scope},
        )

        # 5. Categories & Vectors
        cat_frames = [eng.categories(spec.name) for spec in dataset.event_tables() if spec.categorical_columns]
        cat_df = pd.concat(cat_frames, ignore_index=True) if cat_frames else pd.DataFrame(
            columns=["source", "column", "kind", "value", "n", "distinct_n", "rows", "rate"]
        )
        vec_rows = []
        for spec in dataset.event_tables():
            st = eng.stats[spec.name]
            for col in spec.vector_columns:
                for dim, cnt in sorted(st.vector_lengths[col].items()):
                    vec_rows.append(dict(source=spec.name, column=col, check=f"dimension_{dim}", n=int(cnt), value=np.nan))
                for k, v in st.invalid.items():
                    if col in k:
                        vec_rows.append(dict(source=spec.name, column=col, check=k, n=int(v), value=np.nan))
                for k, v in st.vector_norms[col].summary().items():
                    vec_rows.append(dict(source=spec.name, column=col, check=f"valid_norm_{k}", n=np.nan, value=v))
        sections["categories_and_vectors"] = SectionResult(
            section="categories_and_vectors",
            tables={
                "category_frequencies": cat_df,
                "vector_quality": pd.DataFrame(vec_rows, columns=["source", "column", "check", "n", "value"]),
            },
            metadata={"scope": scope, "max_categories": cfg.max_categories},
        )

        # 6. Entity-level summary & source presence combinations (2^M)
        if entity_ids is not None and len(dataset.event_tables()) > 0:
            ent_summary = pd.DataFrame({"entity_id": entity_ids.astype(str)})
            event_names = [s.name for s in dataset.event_tables() if s.entity_key]
            for name in event_names:
                part = aggregates[aggregates["source"].eq(name)].drop(columns="source").copy()
                part = part.rename(columns={c: f"{name}_{c}" for c in part.columns if c != "entity_id"})
                ent_summary = ent_summary.merge(part, on="entity_id", how="left", validate="one_to_one")
                for c in (f"{name}_event_count", f"{name}_active_days"):
                    if c in ent_summary:
                        ent_summary[c] = ent_summary[c].fillna(0).astype("int64")
                ent_summary[f"has_{name}"] = ent_summary[f"{name}_event_count"].gt(0)
            ent_summary = ent_summary.merge(eng.any_active_days(), on="entity_id", how="left", validate="one_to_one")
            ent_summary["any_active_days"] = ent_summary["any_active_days"].fillna(0).astype("int64")

            metric_cols = [c for c in ent_summary.columns if c.endswith(("_event_count", "_active_days")) or c == "any_active_days"]
            act_summary = pd.DataFrame([numeric_summary(ent_summary[c], feature=c) for c in metric_cols])
            if event_names:
                combo = ent_summary[[f"has_{n}" for n in event_names]].astype(int).astype(str).agg("".join, axis=1)
                combo_col = "_".join(event_names)
                coverage_df = combo.value_counts().rename_axis(combo_col).reset_index(name="entities")
                coverage_df["population"] = len(ent_summary)
                coverage_df["rate"] = coverage_df["entities"] / len(ent_summary) if len(ent_summary) else 0.0
            else:
                coverage_df = pd.DataFrame()
            checks_df = pd.DataFrame([dict(
                reference_entities=len(entity_ids),
                output_entities=len(ent_summary),
                duplicate_keys=int(ent_summary["entity_id"].duplicated().sum()),
                entities_without_timed_events=int(ent_summary["any_active_days"].eq(0).sum()),
            )])
            sections["entity_summary"] = SectionResult(
                section="entity_summary",
                tables={
                    "entity_activity": act_summary,
                    "source_combinations": coverage_df,
                    "aggregation_checks": checks_df,
                },
                metadata={"population": len(ent_summary), "sources": event_names},
            )

        # 7. Targets (from dedicated target table OR target_columns on single table)
        target_spec = tables_map[dataset.target_table] if dataset.target_table else next(
            (t for t in dataset.tables if t.target_columns), None
        )
        if target_spec is not None and target_spec.target_columns:
            tdf = ref_frames.get(target_spec.name)
            if tdf is None:
                tdf = load_dataframe(target_spec.source, **target_spec.source_options)
            tcols = list(target_spec.target_columns)
            prev_rows = []
            for t in tcols:
                n = int(tdf[t].notna().sum())
                pos = int(tdf[t].eq(1).sum())
                prev_rows.append(dict(target=t, rows=len(tdf), labelled_n=n, positive_n=pos,
                                      negative_n=n - pos, missing_n=len(tdf) - n,
                                      rate=pos / n if n else np.nan))
            group_cols = list(target_spec.group_columns)
            if target_spec.time_column and target_spec.time_column in tdf.columns and target_spec.time_column not in group_cols:
                group_cols = [target_spec.time_column, *group_cols]
            grouped_df = grouped_target_rates(tdf, group_cols, tcols, entity_col=target_spec.entity_key) if group_cols else pd.DataFrame()

            pairs = []
            for left, right in combinations(tcols, 2):
                both_valid = tdf[[left, right]].notna().all(axis=1)
                pairs.append(dict(left=left, right=right, observed_pairs=int(both_valid.sum()),
                                  both_positive=int((tdf[left].eq(1) & tdf[right].eq(1)).sum())))

            hist_t = []
            if target_spec.entity_key and target_spec.entity_key in tdf.columns:
                ent_grp = tdf.groupby(target_spec.entity_key)[tcols].agg(["sum", "count"])
                for t in tcols:
                    obs = ent_grp[(t, "count")] > 0
                    pos = ent_grp[(t, "sum")] > 0
                    hist_t.append(dict(target=t, entities=len(ent_grp), observed_entities=int(obs.sum()),
                                       ever_positive_entities=int(pos.sum()),
                                       rate=int(pos.sum()) / int(obs.sum()) if obs.any() else np.nan,
                                       mean_positive_periods=float(ent_grp.loc[obs, (t, "sum")].mean()) if obs.any() else np.nan))

            pos_counts = tdf[tcols].eq(1).sum(axis=1).value_counts().sort_index()
            sections["targets"] = SectionResult(
                section="targets",
                tables={
                    "target_prevalence": pd.DataFrame(prev_rows),
                    "target_by_group": grouped_df,
                    "target_cooccurrence": pd.DataFrame(pairs, columns=["left", "right", "observed_pairs", "both_positive"]),
                    "entity_positive_history": pd.DataFrame(hist_t),
                    "positive_labels_per_row": pos_counts.rename_axis("positive_labels").reset_index(name="rows"),
                },
                metadata={"scope": "full_target_rows"},
            )

        run_meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset.name,
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "duration_seconds": round(perf_counter() - started, 3),
            "scope": scope,
            "timezone": tz_label,
            "config": cfg.to_dict(),
            "sources": {
                name: {
                    "scanned_rows": st.rows,
                    "sample_rows": len(st.sample.frame) if st.sample else 0,
                    "batches": st.batches,
                    "batch_size": st.batch_size,
                    "duration_seconds": st.duration_seconds,
                }
                for name, st in eng.stats.items()
            },
        }
        return ProfileResult(
            dataset_name=dataset.name,
            config=cfg,
            sections=sections,
            metadata=run_meta,
            erd_edges=erd_edges,
        )


def profile_table(
    source: Any,
    *,
    name: str = "table",
    entity_key: str | None = None,
    time_column: str | None = None,
    target_columns: tuple[str, ...] = (),
    config: ProfilerConfig | None = None,
    source_options: dict[str, Any] | None = None,
    progress=None,
) -> ProfileResult:
    """One-liner descriptive profiler for any single tabular source (CSV, Parquet, JSONL, SQLite, DataFrame...)."""
    spec = auto_infer_table_spec(
        source,
        name=name,
        role="tabular",
        entity_key=entity_key,
        time_column=time_column,
        target_columns=tuple(target_columns),
        source_options=source_options,
    )
    dataset = DatasetSpec(name=name, tables=(spec,), timezone=config.timezone if config else None)
    return profile_dataset(dataset, config=config, progress=progress)


def describe_table(
    source: Any,
    *,
    max_rows: int | None = None,
    source_options: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Return a rich 1-row-per-column descriptive summary table for any readable dataset."""
    df = load_dataframe(source, **(source_options or {}))
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)

    n = len(df)
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        miss = int(s.isna().sum())
        non_null = n - miss
        is_bool = pd.api.types.is_bool_dtype(s)
        is_dt = pd.api.types.is_datetime64_any_dtype(s)
        is_num = pd.api.types.is_numeric_dtype(s) and not is_bool

        # Detect array/list columns safely before nunique
        first_valid = s.dropna().iloc[0] if non_null else None
        is_vec = isinstance(first_valid, (list, tuple, np.ndarray, dict))

        if is_vec:
            inferred = "vector"
            nunique = None
            top_val = None
            top_freq = None
        else:
            inferred = "boolean" if is_bool else "datetime" if is_dt else "numeric" if is_num else "categorical"
            nunique = int(s.nunique(dropna=True))
            vc = s.dropna().value_counts()
            top_val = vc.index[0] if len(vc) else None
            top_freq = int(vc.iloc[0]) if len(vc) else 0

        rec: dict[str, Any] = {
            "column": col,
            "dtype": str(s.dtype),
            "inferred_type": inferred,
            "rows": n,
            "non_null_n": non_null,
            "missing_n": miss,
            "missing_pct": (miss / n * 100.0) if n else 0.0,
            "nunique": nunique,
            "unique_pct": (nunique / non_null * 100.0) if (nunique is not None and non_null) else np.nan,
            "top_value": top_val,
            "top_freq": top_freq,
            "top_pct": (top_freq / non_null * 100.0) if (top_freq is not None and non_null) else np.nan,
        }
        if is_num:
            ns = numeric_summary(s)
            for k in ("mean", "std", "min", "p25", "median", "p75", "max", "zeros_pct", "negative_n", "skewness", "kurtosis"):
                rec[k] = ns[k]
        elif is_dt and non_null:
            rec.update(min=str(s.min()), max=str(s.max()))
        rows.append(rec)

    return pd.DataFrame(rows)
