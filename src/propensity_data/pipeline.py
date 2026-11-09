"""Disk-backed preparation. Raw Parquet is read-only; output is a new directory.

No fitted preprocessing or model training is performed here. SQL identifiers are
internal constants; user paths and timestamps are escaped as SQL literals.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .config import PreparationConfig
from .protocol import evaluation_protocol


SOURCES = {"trx": "detail/trx", "geo": "detail/geo", "dialog": "detail/dialog"}
LABELS = tuple(f"target_{i}" for i in range(1, 5))
CATEGORIES = ("event_type", "event_subtype", "src_type11", "src_type12", "dst_type11",
              "dst_type12", "src_type21", "src_type22", "src_type31", "src_type32")


def literal(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def files_at(root: Path) -> list[Path]:
    files = sorted(root.rglob("*.parquet")) if root.is_dir() else [root] if root.is_file() else []
    if not files:
        raise FileNotFoundError(f"No Parquet files under {root}")
    return files


def parquet_scan(files) -> str:
    paths = ",".join(literal(p.as_posix()) for p in files)
    return (f"read_parquet([{paths}], hive_partitioning=false, union_by_name=false, "
            "filename=true, file_row_number=true)")


def export(con, query: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({query}) TO {literal(path.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)")


def assert_empty(con, query: str, message: str):
    if con.execute(f"SELECT EXISTS ({query})").fetchone()[0]:
        raise ValueError(message)


def normalized_id(column="client_id"):
    return f"nullif(trim(cast({column} AS VARCHAR)), '')"


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
                 "array_length(list_filter(raw_embedding, x -> x IS NULL OR NOT isfinite(x))) = 0")
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


def window_features(con, source, cutoff, days, config, out):
    """Aggregate one source/window before joining to the customer universe."""
    tag = "all" if days is None else f"{days}d"
    prefix = f"{source}_{tag}_"
    end = f"TIMESTAMP {literal(cutoff)}"
    lower = "" if days is None else f"AND event_time >= {end} - INTERVAL {days} DAY"
    con.execute(f"CREATE OR REPLACE TEMP VIEW events AS SELECT * FROM {source} WHERE event_time < {end} {lower}")
    scope_start = con.execute("SELECT first_valid_event FROM source_bounds WHERE source = ?", [source]).fetchone()[0]
    start = f"TIMESTAMP {literal(scope_start)}" if scope_start else "NULL::TIMESTAMP"
    # Use source start only: a future last-event timestamp must not define coverage.
    overlap_start = start if days is None else f"greatest({start}, {end} - INTERVAL {days} DAY)"
    overlap = f"CASE WHEN {start} IS NULL THEN 0 ELSE greatest(0, epoch({end} - {overlap_start}) / 86400.0) END"
    truncated = (f"({start} IS NULL OR {start} >= {end})" if days is None else
                 f"({start} IS NULL OR {start} > {end} - INTERVAL {days} DAY)")
    numeric = {
        "event_count": "count(*)", "active_days": "count(DISTINCT cast(event_time AS DATE))",
        "days_since_last_event": f"epoch({end} - max(event_time)) / 86400.0",
        "days_since_first_recorded_event": f"epoch({end} - min(event_time)) / 86400.0",
    }
    zero_when_covered = {"event_count", "active_days"}
    if source == "trx":
        numeric.update({
            "main_count": "count(*) FILTER(WHERE currency = 11)",
            "foreign_count": "count(*) FILTER(WHERE currency <> 11)",
            "currency_missing_count": "count(*) FILTER(WHERE currency_missing)",
            "currency_invalid_count": "count(*) FILTER(WHERE currency_invalid)",
            "amount_missing_count": "count(*) FILTER(WHERE amount_missing)",
            "amount_invalid_count": "count(*) FILTER(WHERE amount_invalid)",
            "main_amount_count": "count(amount) FILTER(WHERE currency = 11)",
            "main_negative_amount_count": "count(*) FILTER(WHERE currency = 11 AND amount < 0)",
            "main_amount_sum": "sum(amount) FILTER(WHERE currency = 11)",
            "main_amount_mean": "avg(amount) FILTER(WHERE currency = 11)",
            "main_amount_median": "median(amount) FILTER(WHERE currency = 11)",
            "main_amount_max": "max(amount) FILTER(WHERE currency = 11)",
            "main_amount_abs_sum": "sum(abs(amount)) FILTER(WHERE currency = 11)",
            "main_amount_signed_log_mean": "avg(sign(amount) * ln(1 + abs(amount))) FILTER(WHERE currency = 11)",
            "foreign_fraction": "count(*) FILTER(WHERE currency <> 11)::DOUBLE / count(*)",
        })
        zero_when_covered.update(k for k in numeric if k.endswith("_count"))
        for cat in CATEGORIES:
            numeric[cat + "_nunique"] = f"count(DISTINCT {cat}) FILTER(WHERE {cat} <> '__MISSING__')"
            numeric[cat + "_missing_count"] = f"count(*) FILTER(WHERE {cat} = '__MISSING__')"
            zero_when_covered.update((cat + "_nunique", cat + "_missing_count"))
        # Sidecar keeps each valid currency as its own monetary unit.
        export(con, "SELECT client_id, currency, count(*) event_count, count(amount) amount_count, "
               "sum(amount) amount_sum, median(amount) amount_median, max(amount) amount_max "
               "FROM events WHERE currency IS NOT NULL GROUP BY client_id, currency",
               out / f"currency/month={cutoff[:7]}/{tag}.parquet")
        # Long-form frequencies avoid learning a category vocabulary from future months.
        category_queries = [
            f"SELECT client_id, {literal(cat)} attribute, {cat} category, count(*) event_count "
            f"FROM events GROUP BY client_id, {cat}" for cat in ("event_type", "event_subtype")
        ]
        export(con, " UNION ALL ".join(category_queries),
               out / f"categories/month={cutoff[:7]}/{tag}.parquet")
    elif source == "geo":
        for i in (4, 5, 6):
            numeric[f"geohash_{i}_nunique"] = f"count(DISTINCT geohash_{i})"
            numeric[f"geohash_{i}_missing_count"] = f"count(*) FILTER(WHERE geohash_{i} IS NULL)"
            zero_when_covered.update((f"geohash_{i}_nunique", f"geohash_{i}_missing_count"))
    expressions = ", ".join(f"{expression} AS {name}" for name, expression in numeric.items())
    con.execute(f"CREATE OR REPLACE TEMP TABLE aggregate_window AS SELECT client_id, {expressions} FROM events GROUP BY client_id")
    select = ["c.client_id", f"({overlap}) AS {prefix}source_window_overlap_days",
              f"cast({truncated} AS TINYINT) AS {prefix}history_truncated",
              f"cast(a.event_count IS NOT NULL AS TINYINT) AS {prefix}has_event"]
    for key in numeric:
        expression = (f"CASE WHEN ({overlap}) > 0 THEN coalesce(a.{key}, 0) END"
                      if key in zero_when_covered else f"a.{key}")
        select.append(f"{expression} AS {prefix}{key}")
    select.append(f"CASE WHEN ({overlap}) > 0 THEN ln(1 + coalesce(a.event_count, 0)) END "
                  f"AS {prefix}log1p_event_count")
    if source == "trx":
        select += [f"ln(1 + a.main_amount_abs_sum) AS {prefix}log1p_main_amount_abs_sum",
                   f"CASE WHEN ({overlap}) > 0 THEN cast(coalesce(a.foreign_count, 0) > 0 AS TINYINT) END "
                   f"AS {prefix}has_foreign"]
    query = f"SELECT {', '.join(select)} FROM clients c LEFT JOIN aggregate_window a USING(client_id)"
    con.execute(f"CREATE OR REPLACE TEMP TABLE window_result AS {query}")
    if source == "trx":
        for cat in ("event_type", "event_subtype"):
            con.execute(f"CREATE OR REPLACE TEMP TABLE category_frequency AS SELECT client_id, {cat}, count(*) n "
                        f"FROM events WHERE {cat} <> '__MISSING__' GROUP BY client_id, {cat}")
            con.execute(f"ALTER TABLE window_result ADD COLUMN {prefix}{cat}_top_share DOUBLE")
            con.execute(f"UPDATE window_result w SET {prefix}{cat}_top_share = g.share FROM "
                        "(SELECT client_id, max(n)::DOUBLE/sum(n) share FROM category_frequency GROUP BY client_id) g "
                        "WHERE w.client_id = g.client_id")
    if source == "geo":
        con.execute("CREATE OR REPLACE TEMP TABLE geo_frequency AS SELECT client_id, geohash_6, count(*) n "
                    "FROM events WHERE geohash_6 IS NOT NULL GROUP BY client_id, geohash_6")
        con.execute(f"ALTER TABLE window_result ADD COLUMN {prefix}top_geohash_6_share DOUBLE")
        con.execute(f"UPDATE window_result w SET {prefix}top_geohash_6_share = g.share FROM "
                    "(SELECT client_id, max(n)::DOUBLE/sum(n) share FROM geo_frequency GROUP BY client_id) g "
                    "WHERE w.client_id = g.client_id")
    if source == "dialog" and config.include_dialog_embeddings and days == config.embedding_window_days:
        expr = ["count(embedding) AS valid_embedding_count"]
        expr += [f"avg(embedding[{i}]) AS mean_{i}" for i in range(1, config.embedding_size + 1)]
        expr += ["first(embedding ORDER BY event_time DESC, source_file, source_row) "
                 "FILTER(WHERE embedding IS NOT NULL) AS latest"]
        con.execute(f"CREATE OR REPLACE TEMP TABLE dialog_vectors AS SELECT client_id, {', '.join(expr)} "
                    "FROM events GROUP BY client_id")
        cols = ["w.*", f"CASE WHEN w.{prefix}source_window_overlap_days > 0 "
                f"THEN coalesce(v.valid_embedding_count, 0) END AS {prefix}valid_embedding_count"]
        cols += [f"v.mean_{i} AS {prefix}embedding_mean_{i}" for i in range(1, config.embedding_size + 1)]
        cols += [f"v.latest[{i}] AS {prefix}embedding_last_{i}" for i in range(1, config.embedding_size + 1)]
        con.execute(f"CREATE OR REPLACE TEMP TABLE with_vectors AS SELECT {', '.join(cols)} "
                    "FROM window_result w LEFT JOIN dialog_vectors v USING(client_id)")
        con.execute("DROP TABLE window_result")
        con.execute("ALTER TABLE with_vectors RENAME TO window_result")
    return prefix


def build_month(con, month, config, out):
    cutoff = month + "-01 00:00:00"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE features AS
        SELECT c.client_id, TIMESTAMP {literal(cutoff)} cutoff,
               t.y_any, {', '.join('t.' + c for c in LABELS)},
               cast(t.y_any IS NOT NULL AS BOOLEAN) supervised_eligible
        FROM clients c LEFT JOIN selected_targets t
          ON c.client_id = t.client_id AND t.cutoff = TIMESTAMP {literal(cutoff)}""")
    for source in SOURCES:
        for days in (*config.windows_days, None):
            window_features(con, source, cutoff, days, config, out)
            con.execute("CREATE OR REPLACE TEMP TABLE joined AS SELECT f.*, w.* EXCLUDE(client_id) "
                        "FROM features f LEFT JOIN window_result w USING(client_id)")
            con.execute("DROP TABLE features")
            con.execute("ALTER TABLE joined RENAME TO features")
        con.execute(f"ALTER TABLE features ADD COLUMN {source}_ever_seen_source TINYINT")
        con.execute(f"UPDATE features SET {source}_ever_seen_source = {source}_all_has_event")
        if source == "dialog":
            con.execute("ALTER TABLE features ADD COLUMN dialog_has_dialog TINYINT")
            con.execute("UPDATE features SET dialog_has_dialog = dialog_ever_seen_source")
    con.execute("ALTER TABLE features ADD COLUMN cold_start TINYINT")
    con.execute("UPDATE features SET cold_start = cast(trx_ever_seen_source = 0 AND "
                "geo_ever_seen_source = 0 AND dialog_ever_seen_source = 0 AS TINYINT)")
    assert_empty(con, "SELECT client_id FROM features GROUP BY client_id HAVING count(*) > 1", "Join multiplied customer rows")
    expected = con.execute("SELECT count(*) FROM clients").fetchone()[0]
    actual, known, positive, cold = con.execute("SELECT count(*), count(y_any), coalesce(sum(y_any),0), "
                                               "sum(cold_start) FROM features").fetchone()
    if actual != expected:
        raise ValueError("Feature preparation lost customers")
    export(con, "SELECT * FROM features ORDER BY client_id", out / f"features/month={month}/part.parquet")
    columns = [row[0] for row in con.execute("DESCRIBE features").fetchall()]
    return {"month": month, "rows": actual, "labelled_rows": known, "positives": positive,
            "unknown_labels": actual - known, "cold_start_rows": cold}, columns


def prepare(config: PreparationConfig) -> Path:
    import duckdb

    sources = {"clients": "client_split", "targets": "targets", **SOURCES}
    files = {key: files_at(config.data_root / rel) for key, rel in sources.items()}
    inventory = [{"source": key, "path": p.relative_to(config.data_root).as_posix(),
                  "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                 for key, paths in files.items() for p in paths]
    fingerprint = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
    out = config.output_dir
    # No overwrite/reuse of half-complete runs or previous results.
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                "config": config.to_dict(), "source_inventory_sha256": fingerprint,
                "fingerprint_note": "Paths, sizes and mtimes; not a full byte checksum",
                "duckdb_version": duckdb.__version__, "models_fitted": False,
                "raw_modified": False, "amount_capping": False,
                "negative_sampling": False, "source_audits": {}, "months": []}
    write_json(out / "manifest.json", manifest)
    write_json(out / "source_inventory.json", inventory)
    con = None
    try:
        con = duckdb.connect(str(out / "work.duckdb"))
        con.execute("SET TimeZone = 'UTC'")
        con.execute(f"SET memory_limit = {literal(config.memory_limit)}")
        con.execute(f"SET threads = {config.threads}")
        con.execute("SET preserve_insertion_order = false")
        con.execute(f"SET temp_directory = {literal((out / 'spill').as_posix())}")
        scans = {key: parquet_scan(paths) for key, paths in files.items()}
        months, protocol = prepare_references(con, scans, config, out)
        write_json(out / "protocol.json", protocol)
        con.execute("CREATE TABLE source_bounds (source VARCHAR PRIMARY KEY, first_valid_event TIMESTAMP)")
        for source in SOURCES:
            print(f"Normalize {source}", flush=True)
            manifest["source_audits"][source] = normalize_events(con, source, scans[source], config, out)
            con.execute("INSERT INTO source_bounds VALUES (?, ?)",
                        [source, manifest["source_audits"][source]["first_valid_event_utc"]])
            write_json(out / "manifest.json", manifest)
        feature_columns = None
        for month in months:
            print(f"Features {month}", flush=True)
            summary, columns = build_month(con, month, config, out)
            if feature_columns is not None and columns != feature_columns:
                raise ValueError("Feature schema changed across months")
            feature_columns = columns
            manifest["months"].append(summary)
            write_json(out / "manifest.json", manifest)
        excluded = ["client_id", "cutoff", "y_any", *LABELS, "supervised_eligible"]
        predictors = [c for c in feature_columns if c not in excluded]
        write_json(out / "feature_schema.json", {
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
        })
        # Catch edits/replacements during the run before declaring outputs ready.
        final_inventory = [{"source": key, "path": p.relative_to(config.data_root).as_posix(),
                            "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                           for key, rel in sources.items() for p in files_at(config.data_root / rel)]
        if final_inventory != inventory:
            raise ValueError("Raw source inventory changed during preparation")
        manifest["status"] = "ready"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if con is not None:
            con.close()
        write_json(out / "manifest.json", manifest)
    return out / "manifest.json"
