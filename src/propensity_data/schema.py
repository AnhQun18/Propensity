"""Source layout and anonymized MBD-mini columns."""
SOURCES = {"trx": "detail/trx", "geo": "detail/geo", "dialog": "detail/dialog"}
LABELS = tuple(f"target_{i}" for i in range(1, 5))
CATEGORIES = ("event_type", "event_subtype", "src_type11", "src_type12", "dst_type11",
              "dst_type12", "src_type21", "src_type22", "src_type31", "src_type32")


def feature_schema(columns: list[str]) -> dict:
    """Explicit whitelist; never expose source targets or keys as predictors."""
    excluded = ["client_id", "cutoff", "y_any", *LABELS, "supervised_eligible"]
    predictors = [c for c in columns if c not in excluded]
    return {
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
        "scope_note": "Calendar extract overlap is not continuous observation of each customer",
    }
