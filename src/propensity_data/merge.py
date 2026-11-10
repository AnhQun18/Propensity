"""Stream prepared customer-month features into one verified local Parquet."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .config import PreparationConfig
from .protocol import evaluation_protocol
from .schema import feature_schema, validate_feature_columns
from .storage import literal, write_json
from .validation import validate_month


def merge_prepared(source_dir, output_dir, batch_size=10_000):
    """Concatenate already joined months; never join raw one-to-many event logs."""
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    source, out = Path(source_dir).resolve(), Path(output_dir).resolve()
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    metadata_paths = [source / name for name in
                      ("manifest.json", "feature_schema.json", "protocol.json", "work.duckdb")]
    prepared = read(metadata_paths[0])
    if prepared.get("status") != "ready":
        raise ValueError("Source manifest must be ready")
    config = PreparationConfig(**prepared["config"])
    for protected in (source, config.data_root):
        if out == protected or out in protected.parents or protected in out.parents:
            raise ValueError("Merged output must not overlap prepared or raw inputs")
    schema = feature_schema(config)
    if read(metadata_paths[1]) != schema:
        raise ValueError("Source schema must match the approved schema v2")
    months = [row["month"] for row in prepared["months"]]
    protocol = evaluation_protocol(config, months)
    if read(metadata_paths[2]) != protocol:
        raise ValueError("Source protocol differs from the evaluation calendar")
    paths = [source / f"features/month={month}/part.parquet" for month in months]

    def inventory():
        return [{"path": p.relative_to(source).as_posix(), "bytes": p.stat().st_size,
                 "mtime_ns": p.stat().st_mtime_ns} for p in (*metadata_paths, *paths)]

    before = inventory()  # Fail on absent inputs before creating output.
    out.mkdir(parents=True, exist_ok=False)
    partial = out / "propensity.incomplete.parquet"
    destination = out / "propensity.parquet"
    manifest = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                "source_dir": str(source), "source_inventory_sha256": prepared["source_inventory_sha256"],
                "schema_version": schema["schema_version"], "operation": "concatenate_customer_month_partitions",
                "input_files": before, "months": [], "batch_size": batch_size,
                "raw_modified": False, "models_fitted": False}
    write_json(out / "manifest.json", manifest)
    try:
        with duckdb.connect(str(source / "work.duckdb"), read_only=True) as con:
            con.execute("SET memory_limit='2GB'")
            con.execute("SET threads=2")
            con.execute(f"SET temp_directory={literal((out / 'spill').as_posix())}")
            writer = None
            arrow_types = {"VARCHAR": pa.string(), "TIMESTAMP": pa.timestamp("us"),
                           "TINYINT": pa.int8(), "BIGINT": pa.int64(),
                           "DOUBLE": pa.float64(), "BOOLEAN": pa.bool_()}
            arrow_schema = pa.schema([(name, arrow_types[schema["dtypes"][name]])
                                      for name in schema["columns"]])
            try:
                for summary, path in zip(prepared["months"], paths):
                    month = summary["month"]
                    check = validate_month(con, month, config, source, schema)
                    positive = con.execute("SELECT coalesce(sum(y_any),0) FROM month_data").fetchone()[0]
                    if (check["scoring_rows"] != summary["rows"] or
                            check["supervised_rows"] != summary["labelled_rows"] or
                            check["unlabelled_rows"] != summary["unknown_labels"] or
                            positive != summary["positives"]):
                        raise ValueError(f"Source manifest counts differ for {month}")
                    parquet = pq.ParquetFile(path)
                    if writer is None:
                        writer = pq.ParquetWriter(partial, arrow_schema, compression="zstd")
                    written = 0
                    for batch in parquet.iter_batches(batch_size=batch_size):
                        # Normalize equivalent Arrow encodings/nullability only after
                        # SQL names/types passed the fixed contract; casts are safe.
                        writer.write_batch(batch.cast(arrow_schema, safe=True))
                        written += batch.num_rows
                    if written != summary["rows"]:
                        raise ValueError(f"Incomplete streamed month {month}")
                    manifest["months"].append(dict(summary))
                    write_json(out / "manifest.json", manifest)
                    print(f"Merged {month}: {written:,} rows", flush=True)
            finally:
                if writer is not None:
                    writer.close()
            con.execute(f"CREATE OR REPLACE TEMP VIEW merged_data AS SELECT * FROM "
                        f"read_parquet({literal(partial.as_posix())}, hive_partitioning=false)")
            validate_feature_columns(con.execute("DESCRIBE merged_data").fetchall(), schema)
            rows, unique, invalid_keys = con.execute("""SELECT count(*),
                count(DISTINCT (client_id,cutoff)),
                count(*) FILTER(WHERE client_id IS NULL OR cutoff IS NULL) FROM merged_data""").fetchone()
            actual = con.execute("""SELECT strftime(cutoff,'%Y-%m'), count(*), count(y_any),
                coalesce(sum(y_any),0), count(*) FILTER(WHERE y_any IS NULL), sum(cold_start)
                FROM merged_data GROUP BY cutoff ORDER BY cutoff""").fetchall()
            expected = [(r["month"], r["rows"], r["labelled_rows"], r["positives"],
                         r["unknown_labels"], r["cold_start_rows"]) for r in prepared["months"]]
            if rows != unique or invalid_keys or rows != sum(r["rows"] for r in prepared["months"]) or actual != expected:
                raise ValueError("Merged keys/monthly counts do not reconcile")
        if inventory() != before:
            raise ValueError("Prepared inputs changed during merge")
        write_json(out / "feature_schema.json", schema)
        write_json(out / "protocol.json", protocol)
        digest = hashlib.sha256()
        with partial.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        partial.rename(destination)
        manifest.update(status="ready", rows=rows, columns=len(schema["columns"]),
                        predictors=len(schema["predictors"]), unique_customer_months=unique,
                        positives=sum(r["positives"] for r in prepared["months"]),
                        unknown_labels=sum(r["unknown_labels"] for r in prepared["months"]),
                        dataset="propensity.parquet", bytes=destination.stat().st_size,
                        dataset_sha256=digest.hexdigest(), finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(out / "manifest.json", manifest)
    return out / "manifest.json"
