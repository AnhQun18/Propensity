"""Stream detail/trx+geo+dialog into a customer-by-month matrix and raster views.

Run: python -X utf8 src/plot_behavior_coverage.py --help
Month boundaries use the naive timestamps as stored. No source is assumed
continuously observed; a zero means no retained log, not missing collection.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("trx", "geo", "dialog")


def month_range(start: str, end: str) -> np.ndarray:
    """[start, end), with explicit whole-month boundaries."""
    for value in (start, end):
        if len(value) != 7 or value[4] != "-":
            raise ValueError("Month must use YYYY-MM; end is exclusive")
        pd.Timestamp(value + "-01")
    months = np.arange(np.datetime64(start, "M"), np.datetime64(end, "M"))
    if not len(months):
        raise ValueError("end must be after start")
    return months


def update_matrix(counts, source_mask, positions, timestamps, months, source_bit, daily=None):
    """Accumulate one batch; repeated pairs across batches/sources remain exact."""
    stamps = np.asarray(timestamps, dtype="datetime64[ns]")
    known = positions >= 0
    valid_time = ~np.isnat(stamps)
    offsets = stamps.astype("datetime64[M]").astype(np.int64) - months[0].astype(np.int64)
    good = known & valid_time & (offsets >= 0) & (offsets < len(months))
    pos, col = positions[good], offsets[good]
    # add.at is essential: counts[pos, col] += 1 loses repeated-index updates.
    np.add.at(counts, (pos, col), 1)
    np.bitwise_or.at(source_mask, (pos, col), source_bit)
    if daily is not None:
        day = (stamps[good].astype("datetime64[D]") - months[0].astype("datetime64[D]")).astype(np.int64)
        daily[pos, day] = True  # same day in multiple sources counts once
    return {
        "rows": len(stamps), "unknown_or_null_client": int((~known).sum()),
        "null_timestamp_known_client": int((known & ~valid_time).sum()),
        "outside_range_known_valid": int((known & valid_time & ~good).sum()),
        "retained_events": int(good.sum()),
    }


def aggregate(data_root, start, end, batch_size=100_000, active_days=False):
    months = month_range(start, end)
    clients = pq.read_table(data_root / "client_split", columns=["client_id"])["client_id"].combine_chunks()
    if clients.null_count or len(pc.unique(clients)) != len(clients) or not len(clients):
        raise ValueError("client_split.client_id must be nonempty, unique and non-null")
    clients = pc.take(clients, pc.sort_indices(clients))
    shape = (len(clients), len(months))
    counts = np.zeros(shape, dtype=np.uint64)  # wide accumulator, checked downcast
    masks = np.zeros(shape, dtype=np.uint8)  # bits 1=trx, 2=geo, 4=dialog
    ndays = int((np.datetime64(end, "D") - months[0].astype("datetime64[D]")).astype(int))
    daily = np.zeros((len(clients), ndays), dtype=bool) if active_days else None
    audit = {}
    for source_no, source in enumerate(SOURCES):
        files = sorted((data_root / "detail" / source).rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No Parquet files in detail/{source}")
        totals = dict.fromkeys(("rows", "unknown_or_null_client", "null_timestamp_known_client",
                              "outside_range_known_valid", "retained_events"), 0)
        for file_no, path in enumerate(files, 1):
            reader = pq.ParquetFile(path)
            timestamp_type = reader.schema_arrow.field("event_time").type
            if not pa.types.is_timestamp(timestamp_type) or timestamp_type.tz is not None:
                raise ValueError(f"Expected naive timestamp, got {timestamp_type}: {path}")
            for batch in reader.iter_batches(batch_size=batch_size, columns=["client_id", "event_time"]):
                positions = pc.fill_null(pc.index_in(batch.column(0), value_set=clients), -1).to_numpy()
                times = batch.column(1).to_numpy(zero_copy_only=False)
                stats = update_matrix(counts, masks, positions, times, months, 1 << source_no, daily)
                for key, value in stats.items():
                    totals[key] += value
            if file_no % 100 == 0 or file_no == len(files):
                print(f"{source}: {file_no}/{len(files)} files, {totals['rows']:,} rows", flush=True)
        assert totals["rows"] == sum(totals[k] for k in totals if k != "rows")
        audit[source] = {"files": len(files), **totals}
    assert int(counts.sum()) == sum(s["retained_events"] for s in audit.values())
    if int(counts.max()) > np.iinfo(np.uint32).max:
        raise OverflowError("A customer-month exceeds uint32; do not downcast")
    counts = counts.astype(np.uint32)
    days = np.empty((0, 0), dtype=np.uint8)
    if daily is not None:
        days = np.zeros(shape, dtype=np.uint8)
        edges = np.append(months, np.datetime64(end, "M")).astype("datetime64[D]")
        for col in range(len(months)):
            left, right = (edges[col:col+2] - edges[0]).astype(int)
            days[:, col] = daily[:, left:right].sum(axis=1)
    return counts, masks, days, np.asarray(clients.to_pylist(), dtype=str), months.astype(str), audit


def row_order(counts, mode="activity"):
    if mode == "id":
        return np.arange(len(counts))
    # Primary key: number of active months; secondary: total events; stable ID ties.
    return np.lexsort((np.arange(len(counts)), -counts.sum(axis=1).astype(np.int64),
                       -(counts > 0).sum(axis=1)))


def grouped_presence(counts, order, max_rows):
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    edges = np.linspace(0, len(order), min(max_rows, len(order)) + 1, dtype=int)
    raster = np.stack([(counts[order[a:b]] > 0).mean(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    return raster, edges


def plot_overview(counts, months, order, output, max_rows=1000):
    from matplotlib.figure import Figure
    from matplotlib.colors import LinearSegmentedColormap
    raster, edges = grouped_presence(counts, order, max_rows)
    fig = Figure(figsize=(14, 9), layout="constrained")
    axes = fig.subplots(2, 1, height_ratios=[1, 5], sharex=True)
    active = (counts > 0).mean(axis=0) * 100
    axes[0].plot(np.arange(len(months)), active, color="#267969", marker="o", linewidth=2)
    axes[0].set_ylabel("Customers with log (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", alpha=.15)
    cmap = LinearSegmentedColormap.from_list("coverage", ["#f6f4ec", "#a7cbb5", "#206b5e"])
    image = axes[1].imshow(raster, aspect="auto", interpolation="nearest", vmin=0, vmax=1,
                           cmap=cmap, origin="upper")
    axes[1].set_xticks(np.arange(len(months)), months, rotation=50, ha="right")
    ticks = np.unique(np.linspace(0, len(raster) - 1, min(7, len(raster)), dtype=int))
    axes[1].set_yticks(ticks, [f"{edges[t]+1:,}–{edges[t+1]:,}" for t in ticks])
    axes[1].set_ylabel("Customer ranks (grouped rows; not individual IDs)")
    axes[1].set_xlabel("Stored calendar month")
    fig.colorbar(image, ax=axes[1], label="Fraction of customers with ≥1 event in each row group")
    fig.suptitle(f"Behavior coverage — union of trx + geo + dialog | {len(counts):,} customers")
    fig.savefig(output, dpi=160, facecolor="#f6f4ec")
    return raster.shape


def select_heatmap_rows(counts, order, sample_size=25, method="random", seed=42, page_start=0):
    """Select actual customers, never aggregate rows in an annotated heatmap."""
    if not 1 <= sample_size <= 100:
        raise ValueError("Annotated heatmap supports 1–100 customers; 20–40 recommended")
    if method == "random":
        return np.random.default_rng(seed).choice(len(counts), min(sample_size, len(counts)), replace=False)
    if method != "page" or page_start < 0 or page_start >= len(order):
        raise ValueError("Invalid sampling method or page-start outside customer range")
    return order[page_start:page_start + sample_size]


def plot_annotated(counts, months, ids, order, output_dir, *, sample_size=25,
                   method="random", seed=42, page_start=0, display_start=None, display_end=None):
    """Reference-style Greens heatmap: one customer per row, exact event counts."""
    from matplotlib.figure import Figure
    from matplotlib.colors import Normalize
    selected = select_heatmap_rows(counts, order, sample_size, method, seed, page_start)
    keep = np.ones(len(months), dtype=bool)
    if display_start:
        month_range(display_start, str(np.datetime64(display_start, "M") + np.timedelta64(1, "M")))
        keep &= months >= display_start
    if display_end:
        month_range(str(np.datetime64(display_end, "M") - np.timedelta64(1, "M")), display_end)
        keep &= months < display_end
    if not keep.any():
        raise ValueError("No cached months in requested heatmap display range")
    shown_months = months[keep]
    values = counts[selected][:, keep]
    labels = [f"KH-{int(i)+1:06d}" for i in selected]
    nrows, ncols = values.shape
    # Cell sizes grow with the sample, never with all 100k customers.
    fig = Figure(figsize=(max(12, ncols * .49 + 2), max(5, nrows * .34 + 2.5)))
    ax = fig.subplots()
    vmax = max(1, int(values.max()))
    norm = Normalize(vmin=0, vmax=vmax)
    image = ax.imshow(values, cmap="Greens", norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(ncols), shown_months, rotation=55, ha="right", fontsize=9)
    ax.set_yticks(np.arange(nrows), labels, fontsize=9)
    ax.set_xticks(np.arange(ncols + 1) - .5, minor=True)
    ax.set_yticks(np.arange(nrows + 1) - .5, minor=True)
    ax.grid(which="minor", color="white", linewidth=.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row in range(nrows):
        for col in range(ncols):
            value = int(values[row, col])
            rgba = image.cmap(norm(value))
            luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
            ax.text(col, row, f"{value:,}", ha="center", va="center",
                    fontsize=8 if value < 10000 else 6.5,
                    color="white" if luminance < .48 else "#202823")
    cbar = fig.colorbar(image, ax=ax, fraction=.035, pad=.035)
    cbar.set_label("Số sự kiện (trx + geo + dialog)", fontsize=11)
    ax.set_xlabel("Tháng", fontsize=11)
    ax.set_ylabel("Khách hàng · mỗi hàng là một khách", fontsize=11)
    ax.set_title("Ma trận độ phủ hoạt động của khách hàng theo tháng", fontsize=17, weight="bold", pad=28)
    selection = (f"Mẫu ngẫu nhiên {nrows}/{len(counts):,} khách · seed={seed}" if method == "random"
                 else f"Hạng {page_start+1:,}–{page_start+nrows:,} trong thứ tự đã chọn · {len(counts):,} khách tổng cộng")
    fig.text(.5, .045, selection + " · số trong ô = tổng log, không phải số ngày", ha="center", fontsize=9, color="#526963")
    fig.text(.5, .022, "0 = không ghi nhận log · mã KH ánh xạ sang client_id trong heatmap_sample.csv", ha="center", fontsize=9, color="#526963")
    fig.subplots_adjust(left=.12, right=.93, top=.90, bottom=.18)
    png = output_dir / "heatmap_annotated.png"
    fig.savefig(png, dpi=180, facecolor="white")
    fig.savefig(output_dir / "heatmap_annotated.pdf", facecolor="white")
    with (output_dir / "heatmap_sample.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["display_label", "client_id", *shown_months])
        for label, index, row in zip(labels, selected, values):
            writer.writerow([label, ids[index], *row.tolist()])
    metadata = {"sample_method": method, "sample_size": nrows, "population": len(counts),
                "seed": seed if method == "random" else None,
                "page_start_zero_based": page_start if method == "page" else None,
                "months": shown_months.tolist(), "color_scale": "linear", "color_min": 0,
                "color_max": int(values.max()), "clipped_cells": 0,
                "cell_definition": "sum of trx + geo + dialog events per actual customer-month",
                "population_is_not_fully_shown": nrows < len(counts)}
    (output_dir / "heatmap_sample_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return png


def write_viewer(counts, masks, days, months, order, output):
    template = Path(__file__).with_name("behavior_coverage_viewer.html").read_text(encoding="utf-8")
    def b64(a): return base64.b64encode(a.tobytes()).decode("ascii")
    payload = {
        "rows": len(counts), "months": months.tolist(),
        "counts": b64(counts.astype("<u4", copy=False)),
        "masks": b64(masks), "days": b64(days),
        "order": b64(order.astype("<u4")),
    }
    output.write_text(template.replace("__MATRIX_DATA__", json.dumps(payload, separators=(",", ":"))), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/MBD-mini")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/behavior_coverage")
    parser.add_argument("--start", default="2020-12", help="YYYY-MM, inclusive")
    parser.add_argument("--end", default="2023-01", help="YYYY-MM, exclusive; first day is cutoff")
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--max-display-rows", type=int, default=1000)
    parser.add_argument("--row-order", choices=["activity", "id"], default="activity")
    parser.add_argument("--active-days", action="store_true", help="Also compute union of calendar days, not sum of source-days")
    parser.add_argument("--plot-only", action="store_true", help="Reuse output-dir/matrix.npz without scanning raw")
    parser.add_argument("--annotated", action="store_true", help="Also export a reference-style heatmap with exact counts in each cell")
    parser.add_argument("--sample-size", type=int, default=25, help="Customers in annotated heatmap, 1–100")
    parser.add_argument("--sample-method", choices=["random", "page"], default="random")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--page-start", type=int, default=0, help="Zero-based row offset for sample-method page")
    parser.add_argument("--heatmap-start", help="Optional annotated display start, YYYY-MM inclusive; does not rescan raw")
    parser.add_argument("--heatmap-end", help="Optional annotated display end, YYYY-MM exclusive; does not rescan raw")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.max_display_rows < 1:
        parser.error("batch-size and max-display-rows must be positive")
    if args.annotated and (not 1 <= args.sample_size <= 100 or args.page_start < 0 or args.sample_seed < 0):
        parser.error("sample-size must be 1–100; page-start and sample-seed must be nonnegative")
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "matrix.npz"
    if args.plot_only:
        with np.load(cache, allow_pickle=False) as data:
            counts, masks, days, ids, months = (data[k] for k in ("counts", "source_mask", "active_days", "client_ids", "months"))
        audit = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))["source_audit"]
        if args.active_days and not days.size:
            parser.error("Cache has no active_days; rescan once with --active-days")
    else:
        counts, masks, days, ids, months, audit = aggregate(args.data_root, args.start, args.end, args.batch_size, args.active_days)
        np.savez_compressed(cache, counts=counts, source_mask=masks, active_days=days, client_ids=ids, months=months)
    order = row_order(counts, args.row_order)
    shape = plot_overview(counts, months, order, args.output_dir / "overview.png", args.max_display_rows)
    write_viewer(counts, masks, days, months, order, args.output_dir / "explorer.html")
    if args.annotated:
        png = plot_annotated(counts, months, ids, order, args.output_dir,
                             sample_size=args.sample_size, method=args.sample_method,
                             seed=args.sample_seed, page_start=args.page_start,
                             display_start=args.heatmap_start, display_end=args.heatmap_end)
        print(f"Annotated heatmap: {png}", flush=True)
    lookup = np.array([i.bit_count() for i in range(8)], dtype=np.uint8)
    with (args.output_dir / "monthly_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["month", "events_all_sources", "customers_any_source", "customers_all_three_sources"])
        for col, month in enumerate(months):
            writer.writerow([month, int(counts[:, col].sum()), int((counts[:, col] > 0).sum()), int((lookup[masks[:, col]] == 3).sum())])
    summary = {
        "shape": list(counts.shape), "count_dtype": str(counts.dtype),
        "count_memory_mb": counts.nbytes / 1e6, "source_mask_memory_mb": masks.nbytes / 1e6,
        "active_days_available": bool(days.size), "display_shape": list(shape), "row_order": args.row_order,
        "months": months.tolist(), "events_retained": int(counts.sum()), "source_audit": audit,
        "count_definition": "sum of recorded trx+geo+dialog events per customer and calendar month; no cross-source deduplication",
        "presence_definition": "count > 0 (at least one source); all-three coverage is source_mask == 7",
        "quality_policy": "known non-null client_id and valid naive event_time; missing other attributes retained",
        "zero_definition": "no retained event in this month; not proof of missing collection or actual inactivity",
        "time_policy": "stored naive calendar; whole-month start inclusive/end exclusive; boundary months may be partial",
        "active_days_definition": "union of dates across all three sources; no date counted more than once",
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("shape", "count_memory_mb", "events_retained", "elapsed_seconds")}), flush=True)
    print(f"Open {args.output_dir / 'explorer.html'}", flush=True)


if __name__ == "__main__":
    main()
