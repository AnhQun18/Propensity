#!/usr/bin/env python3
"""Bổ sung raw, Python >=3.11; giữ cạnh repair_mbd_export.py.

Cài: python -m pip install pandas numpy pyarrow "matplotlib>=3.10" "duckdb>=2.0,<3"
Chạy từ thư mục repo:
  python complete_mbd_statistics.py --export docs/mbd_mini_eda --config configs/mbd_mini_eda.local.json --output-dir docs/mbd_mini_eda_completed
Hoặc truyền --data-root "D:/Datasets/MBD-mini" thay cho --config.

Mặc định: tần suất mọi mã, phân vị amount chính xác và histogram trên toàn bộ
giá trị hữu hạn, amount theo currency, phân phối tháng đầu/cuối của khách,
trùng nguyên dòng xuyên mọi shard/batch. --skip-duplicates bỏ riêng phép đếm
trùng tốn thời gian/đĩa; report vẫn ghi phạm vi trùng cũ, không báo đã làm đủ.

Mặc định memory-limit=4GB, threads<=4; đây là cấu hình DuckDB, không phải giới
hạn cứng của cả Python. --work-dir nên là thư mục trên SSD ngoài raw; dữ liệu
tạm có thể lớn khi kiểm tra trùng. Không nạp toàn bộ sự kiện vào pandas.
Raw và export gốc được giữ nguyên. Output-dir phải mới/rỗng.

Tham khảo API: https://duckdb.org/docs/stable/sql/functions/aggregates.html
và https://duckdb.org/docs/stable/guides/performance/how_to_tune_workloads.html
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from time import perf_counter
from urllib.parse import unquote

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import repair_mbd_export as fix


SOURCES = {"clients": "client_split", "targets": "targets", "transactions": "detail/trx",
           "geo": "detail/geo", "dialog": "detail/dialog"}
EVENTS = ("transactions", "geo", "dialog")
FULL = "all_scanned_finite_amounts"


def qi(name):
    return '"'+str(name).replace('"', '""')+'"'


def sql_literal(text):
    return "'"+str(text).replace("'", "''")+"'"


def log(text):
    print(text, flush=True)


def list_files(root):
    files = [root] if root.is_file() else sorted(root.rglob("*.parquet"))
    if not files or not all(p.exists() for p in files):
        raise ValueError(f"Không có Parquet ở {root}")
    return files


def hive_parts(path):
    result = {}
    for part in path.parent.parts:
        if "=" in part:
            k, v = part.split("=", 1)
            result[k] = unquote(v)
    return result


def open_sources(db, data_root, inventory):
    """Kiểm tra schema và số dòng trước khi tính; không dùng filename làm khóa."""
    descriptions = {}
    for source, relative in SOURCES.items():
        files = list_files(data_root / relative)
        schema, keys, rows = None, None, 0
        for path in files:
            reader = pq.ParquetFile(path)
            current = reader.schema_arrow.remove_metadata()
            parts = hive_parts(path)
            if schema is not None and (not schema.equals(current) or set(parts) != keys):
                raise ValueError(f"Schema/partition thay đổi: {path}")
            schema, keys = current, set(parts)
            rows += reader.metadata.num_rows
            # Nếu fold vừa có trong file vừa có trong đường dẫn, phải khớp.
            for name in set(current.names) & set(parts):
                for batch in reader.iter_batches(columns=[name], batch_size=100000):
                    values = batch.to_pandas()[name]
                    expected = int(parts[name]) if pa.types.is_integer(current.field(name).type) else parts[name]
                    if not values.eq(expected).fillna(False).all():
                        raise ValueError(f"Partition {name} xung đột: {path}")
        expected_rows = int(inventory.loc[inventory.source.eq(source), "rows"].iloc[0])
        if rows != expected_rows:
            raise ValueError(f"Raw khác export: {source} có {rows:,} dòng, export có {expected_rows:,}")
        columns = schema.names + sorted(keys - set(schema.names))
        relation = db.read_parquet([p.as_posix() for p in files], hive_partitioning=bool(keys), union_by_name=False)
        relation.project(",".join(qi(c) for c in columns)).create_view("src_"+source)
        descriptions[source] = dict(schema=schema, columns=columns, rows=rows)
        log(f"Nguồn {source}: {rows:,} dòng, {len(files):,} file; schema và quy mô khớp export")
    return descriptions


def check_references(db, root):
    for source, columns in (("clients", ["client_id"]), ("targets", ["client_id", "mon"])):
        cols = ",".join(qi(c) for c in columns)
        invalid = db.execute(f"SELECT count(*) FROM src_{source} WHERE "+" OR ".join(qi(c)+" IS NULL" for c in columns)).fetchone()[0]
        repeated = db.execute(f"SELECT count(*) FROM (SELECT {cols},count(*) n FROM src_{source} GROUP BY ALL HAVING n>1)").fetchone()[0]
        if invalid or repeated:
            raise ValueError(f"Khóa {source} null hoặc trùng")
    unknown = db.execute("""SELECT count(*) FROM src_targets t LEFT JOIN src_clients c USING(client_id)
        WHERE c.client_id IS NULL OR t.fold IS DISTINCT FROM c.fold""").fetchone()[0]
    if unknown:
        raise ValueError("targets có ID/fold ngoài clients")
    exported = fix.read_table(root, "descriptive.targets", "target_prevalence").set_index("target")
    for target in exported.index:
        n, positive = db.execute(f"SELECT count({qi(target)}),count(*) FILTER(WHERE {qi(target)}=1) FROM src_targets").fetchone()
        if int(n) != int(exported.loc[target, "labelled_n"]) or int(positive) != int(exported.loc[target, "positive_n"]):
            raise ValueError(f"Raw không khớp thống kê nhãn {target} trong export")


def full_categories(db, root, descriptions, top_n):
    roles = fix.read_table(root, "descriptive.structure", "column_roles")
    results = {}
    for source, section, name in (("transactions", "descriptive.transactions", "transaction_categories"),
                                  ("geo", "descriptive.modalities", "location_categories")):
        columns = roles.loc[roles.source.eq(source) & roles.role.eq("categorical"), "column"].tolist()
        total, records = descriptions[source]["rows"], []
        for column in columns:
            log(f"Tần suất đầy đủ: {source}.{column}")
            dtype = roles.loc[roles.source.eq(source) & roles.column.eq(column), "dtype"].iloc[0]
            # CAST từ Arrow integer trực tiếp trong SQL: null không làm mã int thành float.
            rows = db.execute(f"SELECT CAST({qi(column)} AS VARCHAR) value,count(*) n FROM src_{source} GROUP BY {qi(column)}").fetchall()
            nonnull = [(v, int(n)) for v, n in rows if v is not None]
            missing = sum(int(n) for v, n in rows if v is None)
            distinct = len(nonnull)
            for value, n in nonnull:
                encoded = (fix.code_key(value, dtype) if re.fullmatch(r"u?int\d+", dtype) or dtype == "partition"
                           else json.dumps(value) if "string" in dtype else json.dumps(float(value)))
                records.append(dict(source=source, column=column, kind="value", value=encoded, n=n,
                                    distinct_n=distinct, rows=total, rate=n/total if total else np.nan,
                                    frequency_scope="all_rows"))
            records.append(dict(source=source, column=column, kind="missing", value=None, n=missing,
                                distinct_n=distinct, rows=total, rate=missing/total if total else np.nan,
                                frequency_scope="all_rows"))
        cols = ["source", "column", "kind", "value", "n", "distinct_n", "rows", "rate", "frequency_scope"]
        full = pd.DataFrame(records, columns=cols)
        results[section, name+"_full"] = full
        top = []
        for _, part in full.groupby("column", sort=True):
            values = part[part.kind.eq("value")].sort_values(["n", "value"], ascending=[False, True])
            shown = values.head(top_n)
            top.extend(shown.to_dict("records"))
            top.extend(part[part.kind.eq("missing")].to_dict("records"))
            extra = part.iloc[0].to_dict()
            extra.update(kind="remaining", value=None, n=int(values.n.sum()-shown.n.sum()))
            extra["rate"] = extra["n"]/total if total else np.nan
            top.append(extra)
            if int(part.n.sum()) != total:
                raise ValueError("Tần suất không cộng đủ tổng dòng")
        results[section, name] = pd.DataFrame(top, columns=cols)
    return results


def make_histogram(db, minimum, maximum, transform="v", bins=40):
    columns = ["left", "right", "n", "scope"]
    if minimum is None or maximum is None:
        return pd.DataFrame(columns=columns)
    low, high = float(minimum), float(maximum)
    if low == high:
        width = max(abs(low)*.01, .5)
        low, high = low-width, high+width
    if not np.isfinite(high-low):
        raise ValueError("Biên amount quá lớn cho histogram; kiểm tra raw")
    span = high-low
    query = f"""WITH transformed AS (SELECT {transform} x FROM finite_amounts), buckets AS (
        SELECT least({bins-1},greatest(0,CAST(floor((x-?)/?*{bins}) AS INTEGER))) bucket
        FROM transformed)
        SELECT bucket,count(*) n FROM buckets GROUP BY bucket"""
    counts = dict(db.execute(query, [low, span]).fetchall())
    edges = np.linspace(low, high, bins+1)
    return pd.DataFrame([dict(left=edges[i], right=edges[i+1], n=int(counts.get(i, 0)), scope=FULL) for i in range(bins)])


def full_amounts(db, descriptions):
    log("Amount: moment và phân vị chính xác trên toàn bộ giá trị hữu hạn")
    db.execute("CREATE TEMP VIEW amounts AS SELECT TRY_CAST(amount AS DOUBLE) v,currency FROM src_transactions")
    db.execute("CREATE TEMP VIEW finite_amounts AS SELECT v,currency FROM amounts WHERE isfinite(v)")
    n, mean, std, low, high, q = db.execute("""SELECT count(*),avg(v),stddev_samp(v),min(v),max(v),
        quantile_cont(v,[0.25,0.5,0.75,0.95,0.99]) FROM finite_amounts""").fetchone()
    missing, infinite, invalid = db.execute("""SELECT count(*) FILTER(WHERE amount IS NULL),
        count(*) FILTER(WHERE isinf(TRY_CAST(amount AS DOUBLE))),
        count(*) FILTER(WHERE amount IS NOT NULL AND TRY_CAST(amount AS DOUBLE) IS NULL)
        FROM src_transactions""").fetchone()
    q = list(q) if q is not None else [None]*5
    lower = q[0]-1.5*(q[2]-q[0]) if q[0] is not None else None
    upper = q[2]+1.5*(q[2]-q[0]) if q[0] is not None else None
    outliers, whisker_low, whisker_high = db.execute("""SELECT
        count(*) FILTER(WHERE v<? OR v>?),min(v) FILTER(WHERE v>=? AND v<=?),
        max(v) FILTER(WHERE v>=? AND v<=?) FROM finite_amounts""", [lower, upper, lower, upper, lower, upper]).fetchone()
    summary = dict(finite_n=int(n), mean=mean, std=std, min=low, max=high,
                   scanned_rows=descriptions["transactions"]["rows"], sample_rows=0, sample_finite_n=0,
                   missing_n=int(missing), infinite_n=int(infinite), invalid_n=int(invalid),
                   p25=q[0], median=q[1], p75=q[2], p95=q[3], p99=q[4],
                   moments_scope=FULL, quantile_scope=FULL, whisker_low=whisker_low, whisker_high=whisker_high,
                   quantile_method="exact_continuous_type7_linear")
    hist = make_histogram(db, low, high)
    # Equal-width raw-value zoom retains a readable central distribution.
    # The full histogram still accounts for every finite observation.
    zoom_hist = hist.copy()
    if q[-1] is not None:
        zoom_hist = make_histogram_range(db, low, q[-1])
    log("Amount theo mã currency, gồm cả nhóm currency bị thiếu")
    grouped = db.execute("""SELECT CAST(currency AS VARCHAR) currency,count(*) rows,
        count(*) FILTER(WHERE isfinite(v)) finite_n,
        avg(v) FILTER(WHERE isfinite(v)) mean,stddev_samp(v) FILTER(WHERE isfinite(v)) std,
        min(v) FILTER(WHERE isfinite(v)) min,max(v) FILTER(WHERE isfinite(v)) max,
        quantile_cont(v,[0.25,0.5,0.75,0.95,0.99]) FILTER(WHERE isfinite(v)) quantiles
        FROM amounts GROUP BY currency ORDER BY currency""").fetchall()
    rows = []
    for currency, total, finite, avg, stdev, minimum, maximum, quantiles in grouped:
        quantiles = list(quantiles) if quantiles is not None else [None]*5
        rows.append(dict(currency=currency, currency_missing=currency is None, rows=int(total), finite_n=int(finite),
                         mean=avg, std=stdev, min=minimum, max=maximum,
                         **dict(zip(("p25", "median", "p75", "p95", "p99"), quantiles)), scope=FULL))
    if int(hist.n.sum()) != n:
        raise ValueError("Histogram không cộng đủ số amount hữu hạn")
    return {("descriptive.transactions", "amount_summary"): pd.DataFrame([summary]),
            ("descriptive.transactions", "transaction_outliers"): pd.DataFrame([dict(lower=lower, upper=upper,
                outlier_n=int(outliers), analyzed_finite_n=int(n), scope=FULL)]),
            ("descriptive.transactions", "amount_histogram"): hist,
            ("descriptive.transactions", "amount_zoom_histogram"): zoom_hist,
            ("descriptive.transactions", "amount_by_currency"): pd.DataFrame(rows, columns=[
                "currency", "currency_missing", "rows", "finite_n", "mean", "std", "min", "max",
                "p25", "median", "p75", "p95", "p99", "scope"])}


def make_histogram_range(db, minimum, maximum, bins=40):
    """Linear raw-value histogram on an explicit subset, without tail folding."""
    low,high=float(minimum),float(maximum)
    if low==high:
        high=low+max(abs(low)*.01,.5)
    edges=np.linspace(low,high,bins+1)
    counts=dict(db.execute(f"""SELECT least({bins-1},floor((v-?)/?*{bins})::INTEGER) bucket,count(*) n
        FROM finite_amounts WHERE v>=? AND v<=? GROUP BY bucket""",[low,high-low,minimum,maximum]).fetchall())
    return pd.DataFrame([dict(left=edges[i],right=edges[i+1],n=int(counts.get(i,0)),
                              scope='all_finite_amounts_within_min_to_p99') for i in range(bins)])


def customer_dates(db, descriptions):
    records, coverage = [], []
    population = descriptions["clients"]["rows"]
    for source in EVENTS:
        log("Phân phối tháng đầu/cuối theo khách: "+source)
        # GROUP BY chỉ giữ khoảng 100k dòng khách; không tạo bảng khách hàng-tháng.
        query = f"""SELECT c.client_id,min(s.event_time) first_time,max(s.event_time) last_time
            FROM src_clients c LEFT JOIN src_{source} s
            ON c.client_id=s.client_id AND s.event_time IS NOT NULL GROUP BY c.client_id"""
        db.execute("CREATE TEMP TABLE customer_dates_"+source+" AS "+query)
        logged = int(db.execute(f"SELECT count(*) FROM customer_dates_{source} WHERE first_time IS NOT NULL").fetchone()[0])
        coverage.append(dict(source=source, reference_customers=population, with_log=logged,
                             without_log=population-logged, scope="full_reference_customers"))
        for milestone, field in (("first", "first_time"), ("last", "last_time")):
            values = db.execute(f"SELECT strftime({qi(field)},'%Y-%m') AS \"month\",count(*) AS customers FROM customer_dates_{source} WHERE {qi(field)} IS NOT NULL GROUP BY \"month\" ORDER BY \"month\"").fetchall()
            if sum(n for _, n in values) != logged:
                raise ValueError("Phân phối đầu/cuối không khớp số khách có log")
            records += [dict(source=source, milestone=milestone, month=month, customers=int(n),
                             logged_customers=logged, reference_customers=population,
                             rate_among_logged=n/logged if logged else np.nan, scope="full_reference_customers")
                        for month, n in values]
    cols = ["source", "milestone", "month", "customers", "logged_customers", "reference_customers", "rate_among_logged", "scope"]
    return {("descriptive.customers", "first_last_event_month"): pd.DataFrame(records, columns=cols),
            ("descriptive.customers", "first_last_event_coverage"): pd.DataFrame(coverage)}


def normalized_expression(expression, dtype, depth=0):
    """NaN và null là cùng dạng thiếu, kể cả phần tử embedding; không dùng hash thay nội dung."""
    if pa.types.is_floating(dtype):
        return f"CASE WHEN isnan({expression}) THEN NULL ELSE {expression} END"
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype):
        variable = "_element"+str(depth)
        inside = normalized_expression(variable, dtype.value_type, depth+1)
        return f"list_transform({expression},lambda {variable}: {inside})"
    return expression


def full_duplicates(db, root, descriptions):
    rows = []
    old = fix.read_table(root, "descriptive.quality", "duplicates")
    rows += old[old.kind.eq("unique_key")].to_dict("records")
    for source, detail in descriptions.items():
        log(f"Trùng nguyên dòng chính xác: {source} ({detail['rows']:,} dòng); có thể dùng nhiều đĩa tạm")
        types = {f.name: f.type for f in detail["schema"]}
        expressions = [normalized_expression(qi(c), types[c])+" AS "+qi(c) if c in types else qi(c)
                       for c in detail["columns"]]
        query = f"SELECT COALESCE(SUM(n-1),0) FROM (SELECT {','.join(expressions)},count(*) n FROM src_{source} GROUP BY ALL)"
        duplicates = int(db.execute(query).fetchone()[0])
        rows.append(dict(source=source, kind="full_row", duplicate_n=duplicates,
                         analyzed_rows=detail["rows"], scope="all_scanned_rows"))
    return {("descriptive.quality", "duplicates"): pd.DataFrame(rows)}


def raw_options(args, metadata):
    options, parent = {}, Path.cwd()
    if args.config:
        path = args.config.resolve()
        options = json.loads(path.read_text(encoding="utf-8-sig"))
        parent = path.parent
    candidate = args.data_root or options.get("data_root") or metadata.get("config", {}).get("data_root")
    if candidate is None:
        raise ValueError("Cần --data-root hoặc --config có data_root")
    data_root = Path(candidate)
    data_root = data_root.resolve() if data_root.is_absolute() else (parent/data_root).resolve()
    if not data_root.is_dir():
        raise ValueError(f"Không tìm thấy data_root: {data_root}")
    return data_root, options.get("timezone", metadata.get("config", {}).get("timezone"))


def run(args):
    try:
        import duckdb
    except ImportError as error:
        raise ValueError('Cần cài: python -m pip install "duckdb>=2.0,<3"') from error
    started = perf_counter()
    # Kiểm tra raw/đầu ra trước khi ghi bất kỳ file nào.
    original_metadata = fix.peek_metadata(args.export)
    data_root, tz = raw_options(args, original_metadata)
    destination = args.output_dir.resolve()
    if destination.is_relative_to(data_root) or data_root.is_relative_to(destination):
        raise ValueError("Output-dir phải tách khỏi cây thư mục raw")
    if args.work_dir and args.work_dir.resolve().is_relative_to(data_root):
        raise ValueError("work-dir không được nằm trong raw")
    if original_metadata.get("event_scope") != "full":
        raise ValueError("Cần export đầy đủ (event_scope=full) để bổ sung raw và đối chiếu quy mô")
    root, metadata = fix.prepare_export(args.export, args.output_dir)
    try:
        if metadata.get("event_scope") != "full":
            raise ValueError("Cần export đầy đủ (event_scope=full) để bổ sung raw và đối chiếu quy mô")
        if args.work_dir:
            args.work_dir.mkdir(parents=True, exist_ok=True)
        metadata["raw_supplements"] = dict(status="running", data_root=str(data_root))
        fix.write_json(root/"metadata.json", metadata)
        inventory = fix.read_table(root, "descriptive.structure", "table_inventory")
        with tempfile.TemporaryDirectory(prefix="mbd-supp-", dir=args.work_dir) as scratch:
            config = {"memory_limit": args.memory_limit, "threads": args.threads,
                      "preserve_insertion_order": "false", "temp_directory": scratch,
                      "max_temp_directory_size": args.temp_limit}
            with duckdb.connect(config=config) as db:
                db.execute("SET enable_progress_bar = true")
                if tz:
                    db.execute("SET TimeZone = "+sql_literal(tz))
                descriptions = open_sources(db, data_root, inventory)
                check_references(db, root)
                computed = full_categories(db, root, descriptions, args.top_categories)
                computed.update(full_amounts(db, descriptions))
                computed.update(customer_dates(db, descriptions))
                if not args.skip_duplicates:
                    computed.update(full_duplicates(db, root, descriptions))
        # Chỉ ghi các bảng mới sau khi mọi phép tính raw đã thành công.
        for (section, name), frame in computed.items():
            fix.write_table(root, section, name, frame)
        fix.supplement_from_tables(root)
        metadata["raw_supplements"] = dict(status="completed_except_duplicates" if args.skip_duplicates else "completed",
            script_version=fix.VERSION, created_at=datetime.now(timezone.utc).isoformat(),
            data_root=str(data_root), duckdb_version=duckdb.__version__,
            duration_seconds=round(perf_counter()-started, 3), memory_limit=args.memory_limit,
            threads=args.threads, exact_duplicates=not args.skip_duplicates,
            full_amount_quantiles=True, full_categorical_frequencies=True,
            first_last_event_month=True, raw_validation="schema_row_counts_reference_keys_label_totals")
        metadata["sections"]["descriptive.transactions"]["quantile_scope"] = FULL
        metadata["sections"]["descriptive.quality"]["duplicate_mode"] = "retained_from_base_export" if args.skip_duplicates else "exact"
        fix.finalize(root, metadata)
        return root
    except Exception as error:
        metadata["raw_supplements"] = dict(status="failed", error=str(error), duration_seconds=round(perf_counter()-started, 3))
        fix.write_json(root/"metadata.json", metadata)
        # Report phải thể hiện rằng bổ sung chưa hoàn tất, không để trạng thái cũ gây hiểu nhầm.
        original = (root/"report.md").read_text(encoding="utf-8-sig")
        (root/"report.md").write_text("# Bổ sung raw chưa hoàn tất\n\n"+str(error)+"\n\n"+original, encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--temp-limit", default="40GB", help="Giới hạn đĩa tạm DuckDB")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--top-categories", type=int, default=20)
    parser.add_argument("--skip-duplicates", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.top_categories < 1:
        parser.error("threads và top-categories phải dương")
    for setting in (args.memory_limit, args.temp_limit):
        if not re.fullmatch(r"[1-9]\d*(?:\.\d+)?(?:MB|GB|MiB|GiB)", setting):
            parser.error("memory-limit/temp-limit cần dạng 512MB, 4GB, ...")
    try:
        root = run(args)
    except Exception as error:
        parser.exit(1, f"Lỗi: {error}\n")
    print(f"Hoàn thành: {root / 'report.md'}")


if __name__ == "__main__":
    main()
