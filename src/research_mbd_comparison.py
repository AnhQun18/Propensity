"""Compare local MBD-mini with public MBD split/targets; export aggregates only.

Downloads about 102 MB of public archives, never extracts archive paths, and
does not download the full event histories. Run with the project Python.
"""
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import io
import json
import re
import tarfile
import urllib.request

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "mbd_comparison"
CACHE = ROOT / "data" / "research_mbd_sources"
MINI = ROOT / "data" / "MBD-mini"
LABELS = [f"target_{i}" for i in range(1, 5)]
FULL_LABELS = ["bcard_target", "cred_target", "zp_target", "acquiring_target"]


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def archive_frames(path):
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if member.isfile() and member.name.endswith(".parquet"):
                with archive.extractfile(member) as file:
                    frame = pq.ParquetFile(io.BytesIO(file.read())).read().to_pandas()
                for key in ("fold", "is_balanced"):
                    match = re.search(rf"(?:^|/){key}=([^/]+)", member.name)
                    if match and key not in frame:
                        frame[key] = int(match.group(1))
                yield frame


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for name in ("MBD", "MBD-mini"):
        info = get_json(f"https://huggingface.co/api/datasets/ai-lab/{name}")
        revision = info["sha"]
        entries = get_json(f"https://huggingface.co/api/datasets/ai-lab/{name}/tree/{revision}")
        metadata[name] = {"revision": revision,
                          "files": {x["path"]: x["size"] for x in entries if x["type"] == "file"}}
    revision = metadata["MBD"]["revision"]

    def download(name):
        path = CACHE / f"{revision[:12]}_{name}"
        expected = metadata["MBD"]["files"][name]
        if not path.exists() or path.stat().st_size != expected:
            url = f"https://huggingface.co/datasets/ai-lab/MBD/resolve/{revision}/{name}"
            with urllib.request.urlopen(url, timeout=120) as response, path.open("wb") as file:
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)
        if path.stat().st_size != expected:
            raise ValueError(f"Incomplete download: {name}")
        print(f"Ready: {name} ({expected:,} bytes)", flush=True)
        return path

    with ThreadPoolExecutor(max_workers=2) as pool:
        split_path, target_path = list(pool.map(download, ["client_split.tar.gz", "targets.tar.gz"]))
    mini_split = pq.read_table(MINI / "client_split").to_pandas()
    mini_targets = pq.read_table(MINI / "targets").to_pandas()
    mini_ids = set(mini_split.client_id)
    full_split = pd.concat(archive_frames(split_path), ignore_index=True)
    full_ids = set(full_split.client_id)
    joined_split = mini_split.merge(full_split, on="client_id", suffixes=("_mini", "_full"))
    result = {
        "checked_date": "2026-11-04",
        "remote": metadata,
        "scope": "Read all full MBD client_split and targets, compare to local mini. Full event archives not downloaded.",
        "full_split": {
            "rows": len(full_split), "unique_clients": full_split.client_id.nunique(),
            "columns": list(full_split.columns),
            "balanced_counts": full_split.is_balanced.value_counts(dropna=False).to_dict(),
            "fold_counts": full_split.fold.value_counts(dropna=False).to_dict(),
        },
        "mini_split": {
            "rows": len(mini_split), "unique_clients": len(mini_ids),
            "columns": list(mini_split.columns),
            "fold_counts": mini_split.fold.value_counts(dropna=False).to_dict(),
            "clients_absent_from_full": len(mini_ids - full_ids),
            "fold_mismatch_rows": int((joined_split.fold_mini != joined_split.fold_full).sum()),
            "fold_cross_tab_mini_rows_full_columns": pd.crosstab(joined_split.fold_mini, joined_split.fold_full).to_dict(orient="index"),
            "full_balanced_counts_for_mini": joined_split.is_balanced.value_counts(dropna=False).to_dict(),
        },
    }
    rows = 0
    monthly, positives, balanced_rows, balanced_positives = Counter(), Counter(), Counter(), Counter()
    target_ids = set()
    subset_frames = []
    schemas = set()
    file_count = 0
    for frame in archive_frames(target_path):
        file_count += 1
        rows += len(frame)
        schemas.add(tuple(frame.columns))
        monthly.update(frame.mon.value_counts().to_dict())
        positives.update({c: int(frame[c].eq(1).sum()) for c in FULL_LABELS})
        target_ids.update(frame.client_id)
        for flag, group in frame.groupby("is_balanced", observed=True):
            balanced_rows[int(flag)] += len(group)
            for c in FULL_LABELS:
                balanced_positives[f"{flag}:{c}"] += int(group[c].eq(1).sum())
        subset_frames.append(frame.loc[frame.client_id.isin(mini_ids)])
    subset = pd.concat(subset_frames, ignore_index=True)
    left_labels = mini_targets.sort_values(["client_id", "mon"]).reset_index(drop=True)
    right_labels = subset.sort_values(["client_id", "mon"]).reset_index(drop=True)
    if not left_labels[["client_id", "mon"]].equals(right_labels[["client_id", "mon"]]):
        raise ValueError("Cannot align mini/full target keys")
    label_mismatches = {mini: {full: int(left_labels[mini].ne(right_labels[full]).sum())
                              for full in FULL_LABELS} for mini in LABELS}
    mapping = {full: mini for mini, counts in label_mismatches.items()
               for full, count in counts.items() if count == 0}
    if len(mapping) != 4 or len(set(mapping.values())) != 4:
        raise ValueError(f"Label mapping is not uniquely exact: {label_mismatches}")
    result["label_mapping_full_to_mini"] = mapping
    result["label_pair_mismatch_counts"] = label_mismatches
    subset = subset.rename(columns=mapping)
    columns = sorted(set(mini_targets.columns) & set(subset.columns))
    a = mini_targets[columns].sort_values(["client_id", "mon"]).reset_index(drop=True)
    b = subset[columns].sort_values(["client_id", "mon"]).reset_index(drop=True)
    mismatches = {}
    if len(a) == len(b):
        for col in columns:
            aa, bb = a[col].astype(object), b[col].astype(object)
            mismatches[col] = int((~(aa.eq(bb) | (aa.isna() & bb.isna()))).sum())
    result["full_targets"] = {
        "files": file_count, "rows": rows, "unique_clients": len(target_ids),
        "schemas": [list(x) for x in sorted(schemas)], "months": dict(sorted(monthly.items())),
        "positive_counts": dict(positives),
        "positive_rates": {c: positives[c] / rows for c in FULL_LABELS},
        "balanced_rows": dict(balanced_rows), "balanced_positive_counts": dict(balanced_positives),
        "clients_absent_from_split": len(target_ids - full_ids),
    }
    result["mini_targets_comparison"] = {
        "mini_rows": len(a), "full_rows_for_mini_clients": len(b),
        "mini_duplicate_client_months": int(a.duplicated(["client_id", "mon"]).sum()),
        "full_subset_duplicate_client_months": int(b.duplicated(["client_id", "mon"]).sum()),
        "column_mismatch_counts_after_sort": mismatches,
        "mini_positive_counts": {c: int(a[c].eq(1).sum()) for c in LABELS},
        "mini_positive_rates": {c: float(a[c].eq(1).mean()) for c in LABELS},
    }
    result["mini_physical_schema_variants"] = {}
    for name in ["client_split", "targets", "detail/trx", "detail/geo", "detail/dialog", "ptls/trx", "ptls/geo", "ptls/dialog"]:
        variants = Counter(tuple(pq.ParquetFile(p).schema_arrow.names) for p in (MINI / name).rglob("*.parquet"))
        result["mini_physical_schema_variants"][name] = [{"columns": list(cols), "files": count} for cols, count in variants.items()]
    (OUT / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=int) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "mini_physical_schema_variants"}, indent=2, default=int), flush=True)


if __name__ == "__main__":
    main()
