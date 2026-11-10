"""Shared data gate for CLI and both notebooks, before manifest ready."""
from .schema import LABELS, SOURCES, validate_feature_columns
from .storage import assert_empty, literal


def validate_month(con, month, config, out, schema):
    """Validate persisted features, retaining unknown labels and missing sources."""
    file = out / f"features/month={month}/part.parquet"
    con.execute(f"CREATE OR REPLACE TEMP VIEW month_data AS SELECT * FROM "
                f"read_parquet({literal(file.as_posix())}, hive_partitioning=false)")
    validate_feature_columns(con.execute("DESCRIBE month_data").fetchall(), schema)
    assert_empty(con, "SELECT client_id, cutoff FROM month_data GROUP BY ALL HAVING count(*) <> 1",
                 "Duplicate customer-month keys")
    assert_empty(con, "SELECT 1 FROM month_data d ANTI JOIN clients c USING(client_id)",
                 "Feature ID absent from client_split")
    expected_y = ("CASE WHEN target_1=1 OR target_2=1 OR target_3=1 OR target_4=1 THEN 1 "
                  "WHEN target_1=0 AND target_2=0 AND target_3=0 AND target_4=0 THEN 0 ELSE NULL END")
    checks = {
        "Invalid month/key": f"client_id IS NULL OR cutoff IS DISTINCT FROM TIMESTAMP {literal(month + '-01')}",
        "Label eligibility mismatch": "y_any NOT IN (0,1) OR supervised_eligible IS DISTINCT FROM (y_any IS NOT NULL)",
        "Invalid source labels": " OR ".join(f"{c} NOT IN (0,1)" for c in LABELS),
        "y_any differs from source labels": f"y_any IS DISTINCT FROM ({expected_y})",
        "Non-finite/invalid predictor": " OR ".join(
            f'("{name}" IS NOT NULL AND NOT isfinite("{name}"::DOUBLE))' for name in schema["predictors"]),
        "Invalid cold_start": "cold_start IS DISTINCT FROM cast(trx_ever_seen_source=0 AND "
                              "geo_ever_seen_source=0 AND dialog_ever_seen_source=0 AS TINYINT)",
    }
    for source in SOURCES:
        for days in (*config.windows_days, None):
            prefix = f"{source}_{'all' if days is None else str(days) + 'd'}_"
            p = lambda key: prefix + key
            conditions = [
                f"{p('event_count')} < 0", f"{p('active_days')} > {p('event_count')}",
                f"{p('has_event')} IS DISTINCT FROM cast(coalesce({p('event_count')},0)>0 AS TINYINT)",
                f"{p('days_since_last_event')} < 0",
                f"{p('days_since_first_recorded_event')} < {p('days_since_last_event')}",
                f"{p('source_window_overlap_days')} < 0",
                f"({p('source_window_overlap_days')}>0 AND {p('event_count')} IS NULL)",
                f"({p('source_window_overlap_days')}=0 AND {p('event_count')} IS NOT NULL)",
            ]
            if days is not None:
                conditions.append(f"{p('days_since_first_recorded_event')} > {days}")
            if source == "trx":
                conditions += [
                    f"{p('main_count')}+{p('foreign_count')}+{p('currency_missing_count')}+{p('currency_invalid_count')} <> {p('event_count')}",
                    f"{p('known_currency_count')} <> {p('main_count')}+{p('foreign_count')}",
                    f"{p('main_amount_count')} > {p('main_count')}",
                    f"{p('has_foreign')} IS DISTINCT FROM (CASE WHEN {p('source_window_overlap_days')}<=0 THEN NULL "
                    f"WHEN coalesce({p('event_count')},0)=0 THEN 0 WHEN {p('foreign_count')}>0 THEN 1 "
                    f"WHEN {p('main_count')}>0 THEN 0 ELSE NULL END)",
                ]
            checks[prefix + "invariants"] = " OR ".join(conditions)
        tags = [str(days) + "d" for days in sorted(config.windows_days)] + ["all"]
        checks[source + " nested counts"] = " OR ".join(
            f"{source}_{a}_event_count > {source}_{b}_event_count" for a, b in zip(tags, tags[1:]))
        checks[source + " ever seen"] = f"{source}_ever_seen_source IS DISTINCT FROM {source}_all_has_event"
    checks["Invalid dialog flag"] = "dialog_has_dialog IS DISTINCT FROM dialog_ever_seen_source"
    fractions = [c for c in schema["predictors"] if "fraction" in c or "share" in c]
    checks["Invalid fraction"] = " OR ".join(f'("{c}"<0 OR "{c}">1)' for c in fractions)
    expressions = ", ".join(f"count(*) FILTER(WHERE {condition})" for condition in checks.values())
    violations = con.execute("SELECT " + expressions + " FROM month_data").fetchone()
    failed = {name: count for name, count in zip(checks, violations) if count}
    if failed:
        raise ValueError(f"Feature validation failed for {month}: {failed}")
    rows, supervised, unknown, all_missing = con.execute("""
        SELECT count(*), count(*) FILTER(WHERE supervised_eligible),
               count(*) FILTER(WHERE NOT supervised_eligible),
               count(*) FILTER(WHERE target_1 IS NULL AND target_2 IS NULL
                                  AND target_3 IS NULL AND target_4 IS NULL)
        FROM month_data""").fetchone()
    expected = con.execute("SELECT count(*) FROM clients").fetchone()[0]
    if rows != expected or rows != supervised + unknown:
        raise ValueError("Dataset counts do not reconcile")
    # Compare labels to references as well as checking their internal OR rule.
    differences = " OR ".join(f"d.{c} IS DISTINCT FROM t.{c}" for c in (*LABELS, "y_any"))
    assert_empty(con, "SELECT 1 FROM month_data d LEFT JOIN selected_targets t "
                 f"ON d.client_id=t.client_id AND d.cutoff=t.cutoff WHERE {differences}",
                 "Feature labels differ from target references")
    return dict(month=month, scoring_rows=rows, supervised_rows=supervised,
                unlabelled_rows=unknown, all_four_targets_missing=all_missing)
