"""Point-in-time source/window features and customer-month assembly."""
from .schema import CATEGORIES, LABELS, SOURCES
from .storage import assert_empty, export, literal


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
            "known_currency_count": "count(currency)",
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
            "foreign_fraction_known_currency": "count(*) FILTER(WHERE currency <> 11)::DOUBLE / nullif(count(currency), 0)",
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
                   f"CASE WHEN ({overlap}) <= 0 THEN NULL "
                   f"WHEN coalesce(a.event_count, 0) = 0 THEN 0 "
                   f"WHEN a.foreign_count > 0 THEN 1 "
                   f"WHEN a.main_count > 0 THEN 0 ELSE NULL END::TINYINT AS {prefix}has_foreign"]
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
