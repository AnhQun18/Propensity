"""Reference labels and event quality policies; retain missing modalities."""
from .protocol import evaluation_protocol
from .schema import CATEGORIES, LABELS
from .storage import assert_empty, export, literal, normalized_id


def prepare_references(con, scans, config, out):
    con.execute(f"CREATE TABLE clients AS SELECT {normalized_id()} client_id FROM {scans['clients']}")
    assert_empty(con, "SELECT client_id FROM clients GROUP BY client_id HAVING client_id IS NULL OR count(*) > 1",
                 "client_split has missing or duplicate IDs")
    label_select = ", ".join(f"try_cast({c} AS DOUBLE) AS {c}" for c in LABELS)
    # Invalid source strings must not be turned silently into missing labels.
    invalid = " OR ".join(f"({c} IS NOT NULL AND (try_cast({c} AS DOUBLE) IS NULL "
                             f"OR try_cast({c} AS DOUBLE) NOT IN (0,1)))" for c in LABELS)
    assert_empty(con, f"SELECT 1 FROM {scans['targets']} WHERE {invalid}", "Targets contain values outside 0/1/null")
    con.execute(f"""CREATE TABLE targets AS
        WITH parsed AS (
            SELECT {normalized_id()} client_id, date_trunc('month', try_cast(mon AS TIMESTAMP)) cutoff,
                   cast(mon AS VARCHAR) raw_mon, {label_select},
                   filename source_file, file_row_number source_row
            FROM {scans['targets']}
        ) SELECT *, CASE
            WHEN {' OR '.join(c + ' = 1' for c in LABELS)} THEN 1
            WHEN {' AND '.join(c + ' = 0' for c in LABELS)} THEN 0
            ELSE NULL END::TINYINT y_any FROM parsed""")
    assert_empty(con, "SELECT 1 FROM targets WHERE client_id IS NULL OR cutoff IS NULL OR NOT isfinite(cutoff)",
                 "Invalid target ID or reporting month")
    assert_empty(con, "SELECT client_id, cutoff FROM targets GROUP BY ALL HAVING count(*) > 1", "Duplicate customer-month target keys")
    assert_empty(con, "SELECT 1 FROM targets t ANTI JOIN clients c USING(client_id)", "Target ID absent from client_split")
    # Targets can be incomplete: do not fabricate negative labels for absent rows.
    # All published client IDs form the offline scoring universe for every month.
    con.execute(f"CREATE VIEW selected_targets AS SELECT * FROM targets WHERE "
                f"cutoff >= TIMESTAMP {literal(config.first_month + '-01')} AND "
                f"cutoff <= TIMESTAMP {literal(config.last_month + '-01')}")
    months = [r[0] for r in con.execute("SELECT DISTINCT strftime(cutoff, '%Y-%m') FROM selected_targets ORDER BY 1").fetchall()]
    protocol = evaluation_protocol(config, months)
    export(con, "SELECT * FROM targets WHERE y_any IS NULL", out / "audit/unknown_labels.parquet")
    export(con, "SELECT cutoff, count(*) target_rows, count(y_any) labelled_rows, sum(y_any) positives "
           "FROM selected_targets GROUP BY cutoff ORDER BY cutoff", out / "audit/labels_by_month.parquet")
    return months, protocol


def normalize_events(con, name, scan, config, out):
    """Keep good events even when an optional attribute is absent/invalid."""
    fields = [f"{normalized_id()} AS client_id", "try_cast(event_time AS TIMESTAMP) event_time",
              "cast(event_time AS VARCHAR) raw_event_time", "filename source_file", "file_row_number source_row"]
    if name == "trx":
        fields += ["try_cast(amount AS DOUBLE) raw_amount", "cast(amount AS VARCHAR) amount_source",
                   "cast(currency AS VARCHAR) currency_source", "try_cast(currency AS DOUBLE) currency_number"]
        fields += [f"coalesce(cast({c} AS VARCHAR), '__MISSING__') AS {c}" for c in CATEGORIES]
    elif name == "geo":
        fields += [f"cast(geohash_{i} AS VARCHAR) geohash_{i}" for i in (4, 5, 6)]
    elif config.include_dialog_embeddings:
        fields += ["try_cast(embedding AS DOUBLE[]) raw_embedding"]
    con.execute(f"CREATE VIEW raw_{name} AS SELECT {', '.join(fields)} FROM {scan}")
    con.execute(f"""CREATE VIEW classified_{name} AS SELECT *, CASE
        WHEN client_id IS NULL THEN 'missing_client_id'
        WHEN event_time IS NULL OR NOT isfinite(event_time) THEN 'invalid_event_time'
        WHEN NOT EXISTS (SELECT 1 FROM clients c WHERE c.client_id = r.client_id) THEN 'unknown_client_id'
        ELSE NULL END AS rejection_reason FROM raw_{name} r""")
    export(con, f"SELECT * FROM classified_{name} WHERE rejection_reason IS NOT NULL",
           out / f"quarantine/{name}.parquet")
    additions = ""
    if name == "trx":
        known_currency = ("currency_number IS NOT NULL AND isfinite(currency_number) AND "
                          "currency_number = trunc(currency_number) AND try_cast(currency_number AS BIGINT) IS NOT NULL")
        additions = f""",
            CASE WHEN isfinite(raw_amount) THEN raw_amount END AS amount,
            amount_source IS NULL AS amount_missing,
            amount_source IS NOT NULL AND (raw_amount IS NULL OR NOT isfinite(raw_amount)) AS amount_invalid,
            CASE WHEN {known_currency} THEN try_cast(currency_number AS BIGINT) END AS currency,
            currency_source IS NULL AS currency_missing,
            currency_source IS NOT NULL AND NOT coalesce(({known_currency}), false) AS currency_invalid"""
    elif name == "dialog" and config.include_dialog_embeddings:
        valid = (f"array_length(raw_embedding) = {config.embedding_size} AND "
                 "array_length(list_filter(raw_embedding, lambda x: x IS NULL OR NOT isfinite(x))) = 0")
        additions = f", CASE WHEN {valid} THEN raw_embedding END AS embedding"
    con.execute(f"CREATE TABLE {name} AS SELECT * EXCLUDE(rejection_reason){additions} "
                f"FROM classified_{name} WHERE rejection_reason IS NULL")
    con.execute(f"ANALYZE {name}")
    counts = con.execute(f"SELECT count(*), count(*) FILTER(WHERE rejection_reason IS NOT NULL) FROM classified_{name}").fetchone()
    # A timestamp collision is reported, never treated as proof of duplication.
    export(con, f"SELECT client_id, event_time, count(*) n_events FROM {name} "
           "GROUP BY client_id, event_time HAVING count(*) > 1", out / f"audit/{name}_timestamp_collisions.parquet")
    start = con.execute(f"SELECT min(event_time) FROM {name}").fetchone()[0]
    return {"raw_rows": counts[0], "quarantined_rows": counts[1], "accepted_rows": counts[0] - counts[1],
            "first_valid_event_utc": start.isoformat() if start else None,
            "automatic_deduplication": False}
