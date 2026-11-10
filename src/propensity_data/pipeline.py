"""Disk-backed preparation. Raw Parquet is read-only; output is a new directory.

No fitted preprocessing or model training is performed here. SQL identifiers are
internal constants; user paths and timestamps are escaped as SQL literals.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .config import PreparationConfig
from .features import build_month
from .quality import normalize_events, prepare_references
from .schema import SOURCES, feature_schema
from .storage import files_at, literal, parquet_scan, write_json
from .validation import validate_month


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
                "negative_sampling": False, "source_audits": {}, "months": [], "dataset_checks": []}
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
        schema = feature_schema(config)
        for month in months:
            print(f"Features {month}", flush=True)
            summary, _ = build_month(con, month, config, out)
            manifest["dataset_checks"].append(validate_month(con, month, config, out, schema))
            manifest["months"].append(summary)
            write_json(out / "manifest.json", manifest)
        write_json(out / "feature_schema.json", schema)
        manifest["feature_schema_version"] = schema["schema_version"]
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
