"""Compute profiles without writes. Approximate mode keeps a uniform row sample."""
from datetime import datetime, timezone
from contextlib import closing, nullcontext
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
import platform
import numpy as np
import pandas as pd
from .config import DatasetSpec, FileSource, safe_name
from .loader import iter_chunks, source_manifest
from .schema import resolve_schema
from .result import ProfileResult, DatasetResult
from .descriptive import dataset, numerical, categorical, temporal, quality
from .analysis import correlation, bivariate, target, events, drift
from .analysis.relationships import audit
from .utils.stats import missing_mask


def package_version():
    try:
        return version("new-propensity")
    except PackageNotFoundError:
        return "0.1.0+source"


def profile(data, spec=None, *, reference=None):
    """Profile a DataFrame or FileSource. Exact mode materializes all rows in RAM.

    Approximate mode scans all rows for missingness, anomalies and binary-label
    validation. Other summaries use a bounded uniform sample without replacement.
    ``reference`` is an optional DataFrame for descriptive drift comparisons.
    No raw record preview is published by default.
    """
    spec = spec or DatasetSpec()
    if not isinstance(data, (pd.DataFrame, FileSource)):
        raise TypeError("data must be a pandas DataFrame or FileSource")
    chunks = [data] if isinstance(data, pd.DataFrame) else iter_chunks(data, spec.chunk_size)
    total, parts, sampled, priorities = 0, [], None, np.empty(0)
    random = np.random.default_rng(spec.random_seed)
    schema = None
    full_missing = None
    anomaly_totals = None
    dtypes = None
    observed = set()
    declared = set(spec.id_columns + spec.exclude_columns + (spec.numeric_columns or []) +
                   (spec.categorical_columns or []) + list(spec.targets) + list(spec.datetime_columns))
    with closing(chunks) if isinstance(data, FileSource) else nullcontext(chunks):
        for chunk in chunks:
            roles = resolve_schema(chunk, spec)
            missing_counts = chunk.apply(lambda s: int(missing_mask(s).sum()))
            if schema is None:
                schema = roles.copy()
                dtypes = {c: set() for c in chunk}
            elif list(roles) != list(schema):
                raise ValueError("Schema changed across chunks")
            for c in chunk:
                dtypes[c].add(str(chunk[c].dtype))
                # All-missing CSV chunks do not establish a semantic type.
                if c not in declared and missing_counts[c] < len(chunk):
                    if c not in observed:
                        schema[c] = roles[c]
                    elif roles[c] != schema[c]:
                        schema[c] = "categorical"
                    observed.add(c)
            total += len(chunk)
            full_missing = missing_counts if full_missing is None else full_missing.add(missing_counts, fill_value=0)
            numeric_cols = [c for c, role in roles.items() if role == "numeric"]
            anomaly = quality.anomalies(chunk, numeric_cols, spec.valid_ranges)
            if len(anomaly):
                values = anomaly.set_index("column")[["infinite_n", "outside_range_n"]]
                anomaly_totals = values if anomaly_totals is None else anomaly_totals.add(values, fill_value=0)
            # Validate all labels, including rows that will not enter the sample.
            for c, kind in spec.targets.items():
                if kind == "binary" and not chunk[c].dropna().isin([0, 1]).all():
                    raise ValueError(f"{c}: binary labels must be 0/1")
                if kind == "regression" and not pd.api.types.is_numeric_dtype(chunk[c]):
                    raise ValueError(f"{c}: regression target must be numeric")
            if spec.mode == "exact":
                parts.append(chunk)
            else:
                weights = random.random(len(chunk))
                combined = chunk.reset_index(drop=True) if sampled is None else pd.concat([sampled, chunk], ignore_index=True)
                weights = np.concatenate([priorities, weights])
                keep = np.argsort(weights, kind="stable")[:spec.sample_size]
                sampled = combined.iloc[keep].reset_index(drop=True)
                priorities = weights[keep]
    if schema is None:
        raise ValueError("No readable table schema found")
    frame = pd.concat(parts, ignore_index=True) if spec.mode == "exact" else sampled
    numeric_cols = [c for c, role in schema.items() if role == "numeric"]
    cat_cols = [c for c, role in schema.items() if role == "categorical"]
    scope = "full" if len(frame) == total else "sample"
    tables = {}
    scopes = {}

    def add(name, table, calculation_scope=scope):
        tables[name] = table
        scopes[name] = calculation_scope

    info = dataset.overview(frame, spec.row_unit)
    info["rows"] = total
    info["analyzed_rows"] = len(frame)
    info = info.rename(columns={"memory_bytes": "analyzed_memory_bytes"})
    add("overview", info, "full counts; memory refers to analyzed rows")
    columns = dataset.column_profile(frame, schema, spec.column_metadata)
    columns["missing_n"] = columns.column.map(full_missing).astype("int64")
    columns["missing_pct"] = columns.missing_n / total * 100 if total else np.nan
    columns["dtype"] = columns.column.map({c: ", ".join(sorted(types)) for c, types in dtypes.items()})
    add("column_profile", columns, f"missing counts/dtype inventory: full; distinct/constant: {scope}")
    missing = pd.DataFrame(dict(column=list(full_missing.index), missing_n=full_missing.to_numpy(dtype="int64")))
    missing["missing_pct"] = missing.missing_n / total * 100 if total else np.nan
    add("missing", missing, "full")
    add("duplicates", quality.duplicates(frame))
    if spec.id_columns:
        add("duplicate_keys", quality.duplicates(frame, spec.id_columns))
    if numeric_cols:
        add("numeric_summary", numerical.summarize(frame, numeric_cols))
        add("outliers", quality.outliers(frame, numeric_cols))
    if anomaly_totals is not None:
        selected = anomaly_totals.index.intersection(numeric_cols + list(spec.valid_ranges), sort=False)
        add("anomalies", anomaly_totals.loc[selected].reset_index(), "full")
    if cat_cols:
        add("categorical_summary", categorical.frequencies(frame, cat_cols, spec.max_categories))
        add("cardinality", categorical.cardinality(frame, cat_cols, spec.rare_threshold))
    corr_cols = numeric_cols[:spec.max_correlation_columns]
    if len(corr_cols) >= 2:
        add("correlations", correlation.spearman(frame, corr_cols))
    if spec.group_columns:
        add("group_numeric", bivariate.categorical_numeric(frame, spec.group_columns, numeric_cols))
        for i, left in enumerate(spec.group_columns):
            for j, right in enumerate(spec.group_columns[i + 1:], i + 1):
                # Bound cross-tab dimensions, retaining counts in a remainder group.
                from .utils.stats import tokens
                reduced = frame[[left, right]].apply(tokens)
                for c in (left, right):
                    top = reduced[c].value_counts().head(spec.max_categories).index
                    reduced[c] = reduced[c].where(reduced[c].isin(top), "<REMAINING>")
                add(f"crosstab_{i}_{j}", pd.crosstab(reduced[left], reduced[right]).reset_index())
    if spec.datetime_columns:
        summary, counts = temporal.summarize(frame, spec.datetime_columns, spec.time_frequency)
        add("temporal_summary", summary)
        add("temporal_counts", counts)
    if spec.targets:
        add("target_summary", target.summarize(frame, spec.targets, spec.independent_observations))
        regression = [c for c, kind in spec.targets.items() if kind == "regression"]
        if regression:
            add("target_numeric", numerical.summarize(frame, regression))
        add("target_rates", target.grouped_rates(frame, spec.group_columns, spec.targets, spec.independent_observations))
        candidates = [c for c in frame if schema[c] not in {"id", "excluded", "target"}]
        add("leakage_flags", target.leakage_flags(frame, spec.targets, candidates))
    if spec.event_user:
        user_summary = events.summarize(frame, spec.event_user, spec.event_time, spec.datetime_columns[spec.event_time])
        # Keep per-user summaries in memory only; publication is aggregate by default.
        add("event_summary", pd.DataFrame([dict(observed_users=len(user_summary),
                                                events=int(user_summary.event_n.sum()),
                                                median_events_per_user=user_summary.event_n.median(),
                                                median_active_days=user_summary.active_days.median())]))
        if spec.event_type:
            add("event_types", categorical.frequencies(frame, [spec.event_type], spec.max_categories))
    if reference is not None:
        add("drift", drift.compare(reference, frame, numeric_cols, cat_cols), f"reference: supplied frame; current: {scope}")
    manifest = source_manifest(data, spec.hash_sources) if isinstance(data, FileSource) else pd.DataFrame()
    add("manifest", manifest, "source files")
    findings = []
    if scope == "sample":
        findings.append(dict(code="sampled_statistics", message=f"Analyzed {len(frame):,} uniformly sampled rows of {total:,}; see table scopes."))
    if any(role == "unassigned" for role in schema.values()):
        findings.append(dict(code="unassigned_roles", message="Some columns lack a semantic role; declare datetime/numeric/categorical roles."))
    if len(numeric_cols) > len(corr_cols):
        findings.append(dict(code="correlation_limit", message=f"Correlation limited to first {len(corr_cols)} numeric columns."))
    if numeric_cols:
        findings.append(dict(code="finite_statistics", message="Numeric moments and quantiles use finite values; infinities are counted separately."))
    if spec.independent_observations:
        findings.append(dict(code="wilson_assumption", message="Wilson intervals assume independent observations, as declared by the caller."))
    metadata = dict(dataset=spec.dataset, created_at=datetime.now(timezone.utc).isoformat(),
                    package_version=package_version(), python=platform.python_version(),
                    pandas=pd.__version__, numpy=np.__version__, spec=spec.to_dict(),
                    source=asdict(data) if isinstance(data, FileSource) else {"type": "in_memory_dataframe"},
                    total_rows=total, analyzed_rows=len(frame), table_scopes=scopes,
                    sampling="uniform bottom-k independent random priorities" if scope == "sample" else "none",
                    preview_exported=False)
    return ProfileResult(spec, tables, metadata, findings, frame)


def profile_dataset(sources, specs, *, dataset="dataset", relationships=()):
    """Profile independent tables; relationship audits are opt-in and never merge."""
    safe_name(dataset)
    if not sources or set(sources) != set(specs):
        raise ValueError("sources and specs must have identical, nonempty table names")
    results = {}
    for name, data in sources.items():
        safe_name(name)
        results[name] = profile(data, specs[name])
    checks = []
    for relation in relationships:
        if relation.left not in results or relation.right not in results:
            raise ValueError(f"Unknown table in relationship {relation.name}")
        left, right = results[relation.left], results[relation.right]
        table = audit(left._frame, right._frame, relation)
        table["scope"] = "full" if all(r.metadata["total_rows"] == r.metadata["analyzed_rows"] for r in (left, right)) else "sample; not population coverage"
        checks.append(table)
    return DatasetResult(dataset, results, pd.concat(checks, ignore_index=True) if checks else pd.DataFrame())
