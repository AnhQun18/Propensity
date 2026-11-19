"""Read-only audit of zeros and negatives in every numeric detail/trx field."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def inventory(files):
    return [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/MBD-mini")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/trx_zero_negative_20261116")
    args = parser.parse_args()
    source = args.data_root.resolve() / "detail/trx"
    files = sorted(source.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(source)
    before = inventory(files)
    schema = pq.ParquetFile(files[0]).schema_arrow
    numeric = [f.name for f in schema if pa.types.is_integer(f.type) or pa.types.is_floating(f.type)]
    out = args.output_dir.resolve()
    if out == args.data_root.resolve() or args.data_root.resolve() in out.parents or out in args.data_root.resolve().parents:
        raise ValueError("Audit output must not overlap raw-data tree")
    out.mkdir(parents=True, exist_ok=False)
    print(f"Scanning {len(files)} raw trx shards, {len(numeric)} numeric columns", flush=True)
    with duckdb.connect() as con:
        con.execute("SET threads=2")
        con.execute("SET memory_limit='2GB'")
        con.read_parquet([p.as_posix() for p in files], hive_partitioning=False).create_view("trx")
        expressions = ["count(*) AS rows"]
        for index, name in enumerate(numeric):
            expressions += [
                f'count(*) FILTER(WHERE "{name}"=0) AS z{index}',
                f'count(*) FILTER(WHERE "{name}"<0) AS n{index}',
                f'count(*) FILTER(WHERE "{name}" IS NULL) AS m{index}',
                f'count(*) FILTER(WHERE "{name}" IS NOT NULL AND NOT isfinite("{name}")) AS f{index}',
                f'min("{name}") FILTER(WHERE isfinite("{name}")) AS lo{index}',
                f'max("{name}") FILTER(WHERE isfinite("{name}")) AS hi{index}',
            ]
        values = con.execute("SELECT " + ", ".join(expressions) + " FROM trx").fetchall()[0]
        rows = values[0]
        columns = []
        for index, name in enumerate(numeric):
            zero, negative, missing, nonfinite, minimum, maximum = values[1 + index * 6: 7 + index * 6]
            columns.append(dict(column=name, rows=rows, zero=zero, negative=negative,
                                missing=missing, nonfinite=nonfinite, minimum=minimum, maximum=maximum))
        with (out / "numeric_columns.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns[0]))
            writer.writeheader()
            writer.writerows(columns)
        print(json.dumps(next(item for item in columns if item["column"] == "amount")), flush=True)
        currency = con.execute("""SELECT currency, count(*) rows,
            count(*) FILTER(WHERE amount=0) zero_amount,
            count(*) FILTER(WHERE amount<0) negative_amount,
            count(*) FILTER(WHERE amount IS NULL) missing_amount,
            min(amount)::DOUBLE minimum_amount, max(amount)::DOUBLE maximum_amount
            FROM trx GROUP BY currency ORDER BY currency NULLS LAST""").df()
        currency.to_csv(out / "amount_by_currency.csv", index=False)
        event = con.execute("""SELECT event_type, event_subtype, count(*) rows,
            count(*) FILTER(WHERE amount=0) zero_amount,
            count(*) FILTER(WHERE amount<0) negative_amount,
            min(amount)::DOUBLE minimum_amount
            FROM trx GROUP BY event_type,event_subtype ORDER BY event_type,event_subtype""").df()
        event.to_csv(out / "amount_by_event_type_subtype.csv", index=False)
        display_effect = con.execute("""SELECT
            count(*) FILTER(WHERE amount>0 AND amount<1) positive_below_one,
            count(*) FILTER(WHERE amount>0 AND round(amount::DOUBLE,2)=0) positive_rounds_to_zero_2dp,
            count(*) FILTER(WHERE amount>0 AND round(amount::DOUBLE,0)=0) positive_rounds_to_zero_0dp
            FROM trx""").fetchall()[0]
    if before != inventory(sorted(source.rglob("*.parquet"))):
        raise RuntimeError("Raw inventory changed during audit")
    report = dict(status="ready", checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  scope="All raw detail/trx rows; no complete-case filters, sampling or preprocessing",
                  files=len(files), rows=rows, numeric_columns=columns,
                  amount=next(item for item in columns if item["column"] == "amount"),
                  display_effect=dict(zip(("positive_below_one", "positive_rounds_to_zero_2dp", "positive_rounds_to_zero_0dp"), display_effect)),
                  raw_modified=False,
                  inventory_note="Paths, sizes and mtimes checked before/after; not a byte checksum",
                  categorical_note="src/dst/event/currency codes are categories; zero is not automatically an error")
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
