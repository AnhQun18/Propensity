#!/usr/bin/env python3
"""Sửa export MBD-mini mà không đọc raw; Python >=3.10.

Cài: python -m pip install pandas numpy "matplotlib>=3.10"
Chạy: python repair_mbd_export.py --export docs/mbd_mini_eda --output-dir docs/mbd_mini_eda_fixed
Hoặc: python repair_mbd_export.py --export docs.zip --output-dir docs/mbd_mini_eda_fixed

Đặt cùng thư mục với complete_mbd_statistics.py. File này cũng cung cấp các
hàm xuất báo cáo cho script bổ sung. Export gốc được giữ nguyên. Thư mục đầu
ra phải mới hoặc rỗng. Không tự suy ra phân phối log từ histogram tuyến tính,
không biến thống kê trên mẫu thành kết quả toàn bộ.
"""

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np
import pandas as pd


VERSION = "1.0.0"
TITLES = {
    "descriptive.structure": "1. Cấu trúc và quy mô",
    "descriptive.quality": "2. Chất lượng dữ liệu",
    "descriptive.temporal": "3. Phạm vi thời gian",
    "descriptive.transactions": "4. Giao dịch",
    "descriptive.modalities": "5. Vị trí và hội thoại",
    "descriptive.customers": "6. Khách hàng",
    "descriptive.targets": "7. Nhãn",
}


def safe_path(root, relative):
    rel = PurePosixPath(str(relative).replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts or any(":" in p for p in rel.parts):
        raise ValueError(f"Đường dẫn trong export không hợp lệ: {relative}")
    path = root.joinpath(*rel.parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Đường dẫn ngoài export: {relative}")
    return path


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if value is pd.NA or value is pd.NaT or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def peek_metadata(source):
    source = Path(source).resolve()
    if source.is_dir():
        return json.loads((source/"metadata.json").read_text(encoding="utf-8-sig"))
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        candidates = [n for n in names if n.endswith("metadata.json")
                      and n[:-len("metadata.json")]+"manifest.json" in names]
        if len(candidates) != 1:
            raise ValueError("Cần đúng một bộ metadata/manifest trong ZIP")
        return json.loads(archive.read(candidates[0]).decode("utf-8-sig"))


def prepare_export(source, destination):
    """Chỉ chép các file manifest chỉ định; không giải nén tài liệu khác."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source.is_dir() and (destination.is_relative_to(source) or source.is_relative_to(destination)):
        raise ValueError("Thư mục đầu ra phải tách khỏi cây thư mục export gốc")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"Chọn output-dir mới hoặc rỗng: {destination}")
    archive = None
    try:
        if source.is_file() and zipfile.is_zipfile(source):
            archive = zipfile.ZipFile(source)
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("ZIP có tên file lặp")
            candidates = [n[:-len('manifest.json')] for n in names if n.endswith('manifest.json')
                          and n[:-len('manifest.json')]+'metadata.json' in names]
            if len(candidates) != 1:
                raise ValueError("ZIP cần đúng một bộ manifest.json/metadata.json của export")
            prefix = candidates[0]
            reader = lambda name: archive.read(prefix + name)
        elif source.is_dir():
            reader = lambda name: safe_path(source, name).read_bytes()
        else:
            raise ValueError(f"Không có export hoặc ZIP: {source}")
        manifest = json.loads(reader("manifest.json").decode("utf-8-sig"))
        metadata = json.loads(reader("metadata.json").decode("utf-8-sig"))
        raw_root = metadata.get("config", {}).get("data_root")
        if raw_root and Path(raw_root).is_absolute() and destination.is_relative_to(Path(raw_root).resolve()):
            raise ValueError("Không xuất kết quả bên trong dữ liệu gốc")
        paths = ["metadata.json", "manifest.json", "report.md"]
        paths += [t["path"] for t in manifest["tables"]] + list(manifest["figures"])
        for name in paths:
            safe_path(destination, name)
        # Kiểm tra đủ file trước khi bắt đầu chép.
        for name in paths:
            reader(name)
        destination.mkdir(parents=True, exist_ok=True)
        for name in paths:
            target = safe_path(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(reader(name))
        for entry in manifest["tables"]:
            frame = pd.read_csv(safe_path(destination, entry["path"]))
            if len(frame) != entry["rows"] or list(frame) != entry["columns"]:
                raise ValueError(f"CSV không khớp manifest: {entry['path']}")
        return destination, metadata
    finally:
        if archive is not None:
            archive.close()


def read_table(root, section, name):
    path = root / "tables" / section / (name + ".csv")
    if not path.exists():
        return pd.DataFrame()
    header = pd.read_csv(path, nrows=0)
    types = {c: "string" for c in ("value", "trx_geo_dialog", "currency") if c in header}
    return pd.read_csv(path, dtype=types)


def write_table(root, section, name, frame):
    if "client_id" in frame:
        raise ValueError("Chỉ xuất tổng hợp; không xuất client_id")
    path = root / "tables" / section / (name + ".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def code_key(value, dtype):
    """Chuẩn hóa theo schema; giữ mã string có số 0 đầu và giữ chính xác int64."""
    if value is None or value is pd.NA:
        return None
    text = str(value)
    if re.fullmatch(r"u?int\d+", str(dtype)) or dtype == "partition":
        try:
            number = Decimal(text)
        except Exception as error:
            raise ValueError(f"Mã {text!r} không đúng kiểu {dtype}") from error
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError(f"Mã {text!r} không phải số nguyên")
        return str(int(number))
    # Giá trị trong export gốc đã được encode JSON, không parse string thành int.
    parsed = json.loads(text)
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def repair_categories(root, metadata):
    roles = read_table(root, "descriptive.structure", "column_roles")
    types = {(r.source, r.column): r.dtype for r in roles.itertuples(index=False)}
    repairs = []
    for section, table in (("descriptive.transactions", "transaction_categories"),
                           ("descriptive.modalities", "location_categories")):
        original = read_table(root, section, table)
        if original.empty:
            continue
        rows = []
        for (source, column), group in original.groupby(["source", "column"], sort=True):
            if group.get("frequency_scope", pd.Series(dtype=str)).eq("all_rows").all() and "frequency_scope" in group:
                rows.extend(group.to_dict("records"))
                continue
            total = int(group.rows.iloc[0])
            values = group[group.kind.eq("value")].copy()
            values["value"] = values.value.map(lambda v: code_key(v, types.get((source, column), "string")))
            counts = values.groupby("value", dropna=False).n.sum().sort_values(ascending=False)
            remainder = int(group.loc[group.kind.eq("remaining"), "n"].sum())
            missing = int(group.loc[group.kind.eq("missing"), "n"].sum())
            complete = remainder == 0
            scope = "all_rows" if complete else "partial_export_lower_bounds"
            common = dict(source=source, column=column, rows=total,
                          distinct_n=len(counts) if complete else None, frequency_scope=scope)
            rows += [{**common, "kind": "value", "value": str(k), "n": int(n)} for k, n in counts.items()]
            rows += [{**common, "kind": kind, "value": None, "n": n}
                     for kind, n in (("remaining", remainder), ("missing", missing))]
            if int(counts.sum()) + remainder + missing != total:
                raise ValueError(f"Tần suất không cộng đủ tổng dòng: {source}.{column}")
            repairs.append(dict(source=source, column=column,
                                merged_representations=len(values)-len(counts), complete=complete))
        result = pd.DataFrame(rows)
        result["rate"] = result.n / result.rows.replace(0, np.nan)
        write_table(root, section, table, result)
    metadata["export_repairs"] = dict(version=VERSION, corrected_at=datetime.now(timezone.utc).isoformat(),
                                       category_normalization=repairs,
                                       categories_note="partial_export_lower_bounds cần raw để khôi phục đầy đủ")


def supplement_from_tables(root):
    coverage = read_table(root, "descriptive.customers", "modality_coverage")
    if not coverage.empty:
        codes = coverage.trx_geo_dialog.str.zfill(3)
        if not codes.str.fullmatch("[01]{3}").all():
            raise ValueError("Mã tổ hợp nguồn phải gồm ba bit")
        counts = coverage.groupby(codes.str.count("1")).customers.sum().reindex(range(4), fill_value=0)
        total = int(coverage.customers.sum())
        frame = pd.DataFrame({"source_count": range(4), "customers": counts.to_numpy(), "population": total})
        frame["rate"] = frame.customers / total if total else np.nan
        write_table(root, "descriptive.customers", "source_count_distribution", frame)
    labels = read_table(root, "descriptive.targets", "target_by_month_fold")
    if not labels.empty:
        rows = []
        for (mon, fold), part in labels.groupby(["mon", "fold"], sort=True):
            if part.rows.nunique() != 1 or part.customers.nunique() != 1:
                raise ValueError("Mẫu số giữa các target không khớp")
            rows.append(dict(mon=mon, fold=fold, customer_month_rows=int(part.rows.iloc[0]),
                             customers=int(part.customers.iloc[0]), targets=len(part),
                             observed_label_cells=int(part.labelled_n.sum()),
                             missing_label_cells=int(part.missing_n.sum())))
        write_table(root, "descriptive.targets", "label_coverage", pd.DataFrame(rows))


def figure(title, xlabel="", ylabel="", height=4.8):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(9, height), layout="constrained")
    ax = fig.subplots()
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return fig, ax


def save_figure(root, name, fig):
    path = root / "figures" / (name + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=150)
    finally:
        fig.clear()


def refresh_figures(root):
    from matplotlib import dates
    monthly = read_table(root, "descriptive.temporal", "monthly_activity")
    if not monthly.empty:
        for metric, filename in (("events", "monthly_activity"), ("customers", "monthly_customers")):
            fig, ax = figure(f"{metric.title()} by month", "Month", metric.title())
            for source, part in monthly.groupby("source", sort=True):
                part = part.sort_values("month")
                ax.plot(pd.to_datetime(part.month), part[metric], label=source)
            ax.xaxis.set_major_locator(dates.AutoDateLocator(minticks=4, maxticks=12))
            ax.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m"))
            ax.tick_params(axis="x", rotation=45)
            ax.legend()
            save_figure(root, filename, fig)
    ranges = read_table(root, "descriptive.temporal", "source_time_ranges")
    targets = read_table(root, "descriptive.targets", "target_by_month_fold")
    spans = [(r.source, r.first_event, r.last_event) for r in ranges.itertuples(index=False)]
    if not targets.empty:
        spans.append(("targets (mon markers)", targets.mon.min(), targets.mon.max()))
    valid = [(s, pd.Timestamp(a), pd.Timestamp(b)) for s, a, b in spans if pd.notna(a) and pd.notna(b)]
    if valid:
        fig, ax = figure("Observed event ranges and label markers", "Date", "Source")
        for source, first, last in valid:
            a, b = dates.date2num(first.to_pydatetime()), dates.date2num(last.to_pydatetime())
            ax.barh(source, max(b-a, .1), left=a)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=45)
        save_figure(root, "source_timeline", fig)
    for section, table, column, filename in (
        ("descriptive.transactions", "transaction_categories", "event_type", "transaction_event_types"),
        ("descriptive.modalities", "location_categories", "geohash_6", "location_codes")):
        frame = read_table(root, section, table)
        if not frame.empty:
            part = frame[frame.kind.eq("value") & frame.column.eq(column)].nlargest(20, "n")
            if not part.empty:
                fig, ax = figure(column+" (see frequency_scope in CSV)", "Encoded value", "Events")
                ax.bar(part.value.astype(str), part.n)
                ax.tick_params(axis="x", rotation=60)
                save_figure(root, filename, fig)
    amount = read_table(root, "descriptive.transactions", "amount_summary")
    if not amount.empty:
        r = amount.iloc[0]
        if pd.notna(r.get("median")):
            fig, ax = figure("Amount quantiles: "+str(r.quantile_scope), "Stored amount", "Quantile")
            names = ["p25", "median", "p75", "p95", "p99"]
            ax.scatter([r[n] for n in names], names)
            ax.set_xscale("symlog", linthresh=1)
            save_figure(root, "amount_quantiles", fig)
            if pd.notna(r.get("whisker_low")) and pd.notna(r.get("whisker_high")):
                fig, ax = figure("Amount boxplot: "+str(r.quantile_scope)+" (outlier points omitted)", "Stored amount")
                ax.bxp([dict(med=r["median"], q1=r.p25, q3=r.p75, whislo=r.whisker_low,
                             whishi=r.whisker_high, fliers=[])], orientation="horizontal", showfliers=False)
                ax.set_yticks([])
                save_figure(root, "amount_boxplot", fig)
    for name in ("amount_histogram", "amount_log_histogram"):
        frame = read_table(root, "descriptive.transactions", name)
        if not frame.empty:
            xlabel = "signed log10(1+abs(amount))" if "log" in name else "Stored amount"
            fig, ax = figure(name+": "+str(frame.scope.iloc[0]), xlabel, "Events")
            ax.bar(frame.left, frame.n, width=frame.right-frame.left, align="edge")
            save_figure(root, name, fig)
    counts = read_table(root, "descriptive.customers", "source_count_distribution")
    if not counts.empty:
        fig, ax = figure("Number of sources with valid ID/time logs", "Sources", "Customers")
        ax.bar(counts.source_count.astype(str), counts.customers)
        save_figure(root, "source_count_distribution", fig)
    first_last = read_table(root, "descriptive.customers", "first_last_event_month")
    if not first_last.empty:
        for kind in ("first", "last"):
            fig, ax = figure(kind.title()+" event month per customer", "Month", "Customers with logs")
            for source, part in first_last[first_last.milestone.eq(kind)].groupby("source"):
                part = part.sort_values("month")
                ax.plot(pd.to_datetime(part.month), part.customers, label=source)
            ax.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m"))
            ax.tick_params(axis="x", rotation=45)
            ax.legend()
            save_figure(root, kind+"_event_month", fig)


def findings(root, metadata):
    notes = {key: [] for key in TITLES}
    inventory = read_table(root, "descriptive.structure", "table_inventory")
    if not inventory.empty:
        notes["descriptive.structure"].append("Quy mô: "+"; ".join(f"{r.source}: {r.rows:,} dòng" for r in inventory.itertuples(index=False))+".")
    missing = read_table(root, "descriptive.quality", "missing_values")
    if not missing.empty:
        part = missing[missing.missing_n.gt(0)].sort_values("missing_rate", ascending=False).head(5)
        notes["descriptive.quality"] += [f"{r.source}.{r.column}: thiếu {r.missing_n:,}/{r.rows:,} ({r.missing_rate:.3%})." for r in part.itertuples(index=False)]
    duplicates = read_table(root, "descriptive.quality", "duplicates")
    for r in duplicates.itertuples(index=False):
        notes["descriptive.quality"].append(f"Trùng {r.source}/{r.kind}: {int(r.duplicate_n):,} trên {int(r.analyzed_rows):,} dòng; phạm vi **{r.scope}**.")
    ranges = read_table(root, "descriptive.temporal", "source_time_ranges")
    notes["descriptive.temporal"] += [f"{r.source}: {r.first_event} đến {r.last_event}." for r in ranges.itertuples(index=False)]
    labels = read_table(root, "descriptive.targets", "target_by_month_fold")
    if not labels.empty:
        notes["descriptive.temporal"].append(f"Mốc mon của nhãn: {labels.mon.min()} đến {labels.mon.max()}; đây là marker gốc, không suy diễn khoảng nhãn/cutoff.")
    amount = read_table(root, "descriptive.transactions", "amount_summary")
    if not amount.empty:
        a = amount.iloc[0]
        notes["descriptive.transactions"].append(f"Amount: mean={a['mean']:,.6g}, std={a['std']:,.6g}, median={a['median']:,.6g}, min={a['min']:,.6g}, max={a['max']:,.6g}.")
        notes["descriptive.transactions"].append(f"Moment: **{a.moments_scope}**; phân vị: **{a.quantile_scope}**.")
        if a.quantile_scope != "all_scanned_finite_amounts":
            notes["descriptive.transactions"].append("Phân vị trên mẫu không phải số chính xác của quần thể.")
        notes["descriptive.transactions"].append("Các mã currency chưa được diễn giải hoặc quy đổi; xem amount_by_currency nếu đã chạy bổ sung raw.")
    hist = read_table(root, "descriptive.transactions", "amount_histogram")
    if not hist.empty and hist.n.sum():
        notes["descriptive.transactions"].append(f"Cột đầu histogram tuyến tính chứa {int(hist.n.iloc[0]):,}/{int(hist.n.sum()):,} quan sát ({hist.n.iloc[0]/hist.n.sum():.6%}); xem thêm histogram log nếu có.")
    for section, name in (("descriptive.transactions", "transaction_categories"), ("descriptive.modalities", "location_categories")):
        frame = read_table(root, section, name)
        if not frame.empty and "frequency_scope" in frame:
            partial = sorted(frame.loc[frame.frequency_scope.eq("partial_export_lower_bounds"), "column"].unique())
            if partial:
                notes[section].append("Tần suất chưa khôi phục đầy đủ từ top 20: "+", ".join(partial)+". distinct_n để trống; cần script đọc raw.")
    coverage = read_table(root, "descriptive.quality", "key_coverage")
    notes["descriptive.modalities"] += [f"{r.source}: {r.customers_with_valid_time_log:,}/{r.reference_customers:,} khách có log ({r.coverage:.2%})." for r in coverage.itertuples(index=False) if r.source in ("geo", "dialog")]
    counts = read_table(root, "descriptive.customers", "source_count_distribution")
    notes["descriptive.customers"] += [f"{r.source_count} nguồn: {r.customers:,} khách ({r.rate:.2%})." for r in counts.itertuples(index=False)]
    if read_table(root, "descriptive.customers", "first_last_event_month").empty and not metadata.get("raw_supplements", {}).get("first_last_event_month"):
        notes["descriptive.customers"].append("Chưa có phân phối thời điểm đầu/cuối theo khách; cần chạy complete_mbd_statistics.py với raw.")
    prevalence = read_table(root, "descriptive.targets", "target_prevalence")
    notes["descriptive.targets"] += [f"{r.target}: {r.positive_n:,} dương, {r.negative_n:,} âm, thiếu {r.missing_n:,}; tỷ lệ dương {r.rate:.4%}." for r in prevalence.itertuples(index=False)]
    return notes


def finalize(root, metadata):
    """Viết report và manifest khớp đúng các file hiện có trong bản sao."""
    refresh_figures(root)
    manifest = {"tables": [], "figures": [], "metadata": "metadata.json", "report": "report.md"}
    for path in sorted((root / "tables").glob("*/*.csv")):
        frame = pd.read_csv(path)
        if "client_id" in frame:
            raise ValueError("Export chỉ chứa bảng tổng hợp")
        manifest["tables"].append(dict(section=path.parent.name, name=path.stem,
                                       path=path.relative_to(root).as_posix(), rows=len(frame), columns=list(frame)))
    manifest["figures"] = [p.relative_to(root).as_posix() for p in sorted((root / "figures").glob("*.png"))]
    notes = findings(root, metadata)
    lines = ["# Bảy bước thống kê mô tả MBD-mini — bản chỉnh sửa", "",
             "Phạm vi dữ liệu gốc: **"+str(metadata.get("event_scope", "unknown"))+"**.",
             "Timezone: **"+str(metadata.get("config", {}).get("timezone") or "naive_as_stored")+"**; giữ quy ước thời gian trong dữ liệu.",
             "Tổng hợp toàn lịch sử; không tạo feature khách hàng–tháng hoặc huấn luyện mô hình.",
             "Mỗi bảng ghi phạm vi riêng. Không suy rộng kết quả kiểm tra trùng trên mẫu.", ""]
    raw = metadata.get("raw_supplements", {})
    category_note = ("Tần suất mọi mã đã tính lại từ raw; bảng top 20 giữ phần remaining, bảng *_full.csv chứa đầy đủ mã."
                     if raw.get("full_categorical_frequencies") else
                     "Script sửa export chỉ khôi phục đầy đủ cột categorical khi remaining=0.")
    lines += ["Bổ sung raw: **"+raw.get("status", "chưa chạy")+"**.", category_note, "",
              "```mermaid", "erDiagram", "    clients ||--o{ targets : client_id",
              "    clients ||--o{ transactions : client_id", "    clients ||--o{ geo : client_id",
              "    clients ||--o{ dialog : client_id", "```", ""]
    raw_time = str(raw["duration_seconds"])+" giây" if "duration_seconds" in raw else "chưa chạy"
    lines += [f"Thời gian phân tích export gốc: {metadata.get('duration_seconds', 'unknown')} giây.",
              f"Thời gian bổ sung raw: {raw_time}.",
              "Config và thời gian các section/source giữ thông tin lần chạy gốc; raw_supplements ghi lần bổ sung. Phạm vi cuối xem ở từng bảng.", ""]
    for section, title in TITLES.items():
        lines += ["## "+title, ""]
        lines += ["- "+note for note in notes[section]]
        lines += [""]
        for item in manifest["tables"]:
            if item["section"] == section:
                lines.append(f"- [{item['name']}]({item['path']}): {item['rows']:,} dòng tổng hợp.")
        lines += [""]
    lines += ["## Biểu đồ", ""]
    for path in manifest["figures"]:
        lines += [f"![{Path(path).stem}]({path})", ""]
    metadata["export_revision"] = dict(version=VERSION, revised_at=datetime.now(timezone.utc).isoformat(),
                                       tables=len(manifest["tables"]), figures=len(manifest["figures"]))
    write_json(root / "metadata.json", metadata)
    write_json(root / "manifest.json", manifest)
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def repair(source, destination):
    root, metadata = prepare_export(source, destination)
    repair_categories(root, metadata)
    supplement_from_tables(root)
    finalize(root, metadata)
    return root, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", required=True, type=Path, help="Thư mục có manifest.json hoặc docs.zip")
    parser.add_argument("--output-dir", required=True, type=Path, help="Thư mục kết quả mới/rỗng")
    args = parser.parse_args()
    try:
        root, _ = repair(args.export, args.output_dir)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Lỗi: {error}\n")
    print(f"Đã sửa export: {root / 'report.md'}")
    print("Các mục còn cần raw được ghi rõ trong report; chạy file complete_mbd_statistics.py để bổ sung.")


if __name__ == "__main__":
    main()
