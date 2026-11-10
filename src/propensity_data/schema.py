"""Source layout and anonymized MBD-mini columns."""
SOURCES = {"trx": "detail/trx", "geo": "detail/geo", "dialog": "detail/dialog"}
LABELS = tuple(f"target_{i}" for i in range(1, 5))
CATEGORIES = ("event_type", "event_subtype", "src_type11", "src_type12", "dst_type11",
              "dst_type12", "src_type21", "src_type22", "src_type31", "src_type32")


def feature_schema(config, *, window_tags=None) -> dict:
    """Approved names, order and DuckDB types; never infer predictors from input."""
    excluded = ["client_id", "cutoff", "y_any", *LABELS, "supervised_eligible"]
    dtypes = {"client_id": "VARCHAR", "cutoff": "TIMESTAMP", "y_any": "TINYINT",
              **{c: "DOUBLE" for c in LABELS}, "supervised_eligible": "BOOLEAN"}
    for source in SOURCES:
        tags = window_tags if window_tags is not None else [*(str(d) + 'd' for d in config.windows_days), 'all']
        for tag in tags:
            prefix = f"{source}_{tag}_"
            window = {
                "source_window_overlap_days": "DOUBLE", "history_truncated": "TINYINT",
                "has_event": "TINYINT", "event_count": "BIGINT", "active_days": "BIGINT",
                "days_since_last_event": "DOUBLE", "days_since_first_recorded_event": "DOUBLE",
            }
            if source == "trx":
                window.update({name: "BIGINT" for name in (
                    "main_count", "foreign_count", "known_currency_count", "currency_missing_count",
                    "currency_invalid_count", "amount_missing_count", "amount_invalid_count",
                    "main_amount_count", "main_negative_amount_count")})
                window.update({name: "DOUBLE" for name in (
                    "main_amount_sum", "main_amount_mean", "main_amount_median", "main_amount_max",
                    "main_amount_abs_sum", "main_amount_signed_log_mean", "foreign_fraction",
                    "foreign_fraction_known_currency")})
                for cat in CATEGORIES:
                    window.update({cat + "_nunique": "BIGINT", cat + "_missing_count": "BIGINT"})
            elif source == "geo":
                for i in (4, 5, 6):
                    window.update({f"geohash_{i}_nunique": "BIGINT", f"geohash_{i}_missing_count": "BIGINT"})
            window["log1p_event_count"] = "DOUBLE"
            if source == "trx":
                window.update({"log1p_main_amount_abs_sum": "DOUBLE", "has_foreign": "TINYINT",
                               "event_type_top_share": "DOUBLE", "event_subtype_top_share": "DOUBLE"})
            elif source == "geo":
                window["top_geohash_6_share"] = "DOUBLE"
            elif config.include_dialog_embeddings and tag == str(config.embedding_window_days) + 'd':
                window["valid_embedding_count"] = "BIGINT"
                for pooling in ("mean", "last"):
                    window.update({f"embedding_{pooling}_{i}": "DOUBLE"
                                   for i in range(1, config.embedding_size + 1)})
            dtypes.update({prefix + name: dtype for name, dtype in window.items()})
        dtypes[source + "_ever_seen_source"] = "TINYINT"
        if source == "dialog":
            dtypes["dialog_has_dialog"] = "TINYINT"
    dtypes["cold_start"] = "TINYINT"
    predictors = [c for c in dtypes if c not in excluded]
    return {
        "schema_version": 2 if window_tags is None else 3, "columns": list(dtypes), "dtypes": dtypes,
        "keys": ["client_id", "cutoff"], "target": "y_any", "excluded_from_predictors": excluded,
        "predictors": predictors,
        "groups": {s: [c for c in predictors if c.startswith(s + '_')] for s in SOURCES},
        "ablation_groups": {
            "trx": [c for c in predictors if c.startswith('trx_')],
            "source_metadata": [c for c in predictors if c.startswith(('geo_', 'dialog_'))
                                and 'geohash_' not in c and 'embedding_' not in c],
            "geo_content": [c for c in predictors if c.startswith('geo_') and 'geohash_' in c],
            "dialog_content": [c for c in predictors if c.startswith('dialog_') and 'embedding_' in c],
            "cohort_flags": ["cold_start"],
        },
        "missing_numeric": "Keep NaN; counts are zero only within calendar overlap with the source",
        "currency_sidecar": "Separate units; do not sum amounts across currency values",
        "feature_definitions": {
            "*_has_foreign": "1 if known foreign observed; 0 if no logs in covered window or known main and no known foreign; null if only unknown currency or no source overlap",
            "*_foreign_fraction": "Observed foreign logs / all logs; 0 does not prove all currencies were main",
            "*_foreign_fraction_known_currency": "Known foreign logs / logs with valid currency; null when denominator is zero",
        },
        "scope_note": "Calendar extract overlap is not continuous observation of each customer",
    }


def validate_feature_columns(description, schema):
    """Reject unapproved/missing/reordered columns and dtype drift before reading X."""
    found = [(row[0], row[1]) for row in description]
    expected = [(name, schema["dtypes"][name]) for name in schema["columns"]]
    if found != expected:
        extra = sorted(set(name for name, _ in found) - set(schema["columns"]))
        missing = sorted(set(schema["columns"]) - set(name for name, _ in found))
        raise ValueError(f"Feature schema mismatch (columns/order/dtypes); extra={extra}, missing={missing}")
