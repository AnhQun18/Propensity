"""Privacy-safe aggregate EDA report exporter (Markdown + CSV + JSON + Mermaid ERD)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .formatting import json_safe, safe_name


def _normalize_sections(result_or_sections: Any):
    if hasattr(result_or_sections, "sections") and isinstance(result_or_sections.sections, Mapping):
        norm = {}
        for sec_name, sec in result_or_sections.sections.items():
            norm[sec_name] = {
                "tables": dict(getattr(sec, "tables", {})),
                "findings": list(getattr(sec, "findings", [])),
                "metadata": dict(getattr(sec, "metadata", {})),
            }
        meta = dict(getattr(result_or_sections, "metadata", {}))
        edges = list(getattr(result_or_sections, "erd_edges", []))
        ds_name = getattr(result_or_sections, "dataset_name", None)
        return norm, meta, edges, ds_name

    if isinstance(result_or_sections, Mapping):
        norm = {}
        for sec_name, val in result_or_sections.items():
            if isinstance(val, Mapping) and "tables" in val and isinstance(val["tables"], Mapping):
                norm[sec_name] = {
                    "tables": dict(val["tables"]),
                    "findings": list(val.get("findings", [])),
                    "metadata": dict(val.get("metadata", {})),
                }
            elif isinstance(val, Mapping):
                norm[sec_name] = {"tables": dict(val), "findings": [], "metadata": {}}
            elif isinstance(val, pd.DataFrame):
                norm[sec_name] = {"tables": {sec_name: val}, "findings": [], "metadata": {}}
            else:
                raise TypeError(f"Unsupported section value for {sec_name!r}: {type(val).__name__}")
        return norm, {}, [], None

    raise TypeError("Expected a ProfileResult-like object or a mapping of sections")


def export_report(
    result_or_sections: Any,
    output_dir: str | Path,
    *,
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
    erd_edges: Sequence[Mapping[str, str]] | None = None,
    figures: Mapping[str, Any] | None = None,
    forbidden_columns: Sequence[str] = ("client_id", "user_id", "customer_id", "entity_id"),
    protected_paths: Sequence[str | Path] = (),
) -> Path:
    """Export aggregate EDA tables, figures, metadata.json, manifest.json, and report.md safely.

    Blocks exporting any table containing an identifier in `forbidden_columns` before touching disk.
    """
    destination = Path(output_dir).resolve()
    for raw in protected_paths:
        p = Path(raw).resolve()
        if destination == p or p in destination.parents or destination in p.parents:
            raise ValueError(f"Cannot export report overlapping protected path: {p}")

    sections, embedded_meta, embedded_edges, ds_name = _normalize_sections(result_or_sections)
    edges = list(erd_edges) if erd_edges is not None else embedded_edges
    forbidden = {c.lower() for c in forbidden_columns}

    for sec_name, sec in sections.items():
        safe_name(sec_name)
        for tbl_name, frame in sec["tables"].items():
            safe_name(tbl_name)
            leaked = sorted(c for c in frame.columns if str(c).lower() in forbidden)
            if leaked:
                raise ValueError(
                    f"Privacy guard blocked export of {sec_name}.{tbl_name}: contains forbidden identifier column(s) {leaked}"
                )
    if figures:
        for fig_name in figures:
            safe_name(fig_name)

    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"tables": [], "figures": [], "metadata": "metadata.json", "report": "report.md"}
    report_title = title or (f"EDA Report: {ds_name}" if ds_name else "Aggregate EDA Report")
    lines = [f"# {report_title}", ""]

    if edges:
        lines.extend(["## Data Relationships", "", "```mermaid", "erDiagram"])
        for edge in edges:
            lines.append(f"    {edge.get('left', 'entity')} ||--o{{ {edge.get('right', 'events')} : {edge.get('key', 'id')}")
        lines.extend(["```", ""])

    for sec_name, sec in sections.items():
        lines.extend([f"## {sec_name}", ""])
        for finding in sec["findings"]:
            lines.append(f"- {finding}")
        if sec["findings"]:
            lines.append("")
        folder = destination / "tables" / sec_name
        for tbl_name, frame in sec["tables"].items():
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{tbl_name}.csv"
            frame.to_csv(path, index=False, encoding="utf-8-sig")
            rel = path.relative_to(destination).as_posix()
            manifest["tables"].append({
                "section": sec_name,
                "name": tbl_name,
                "path": rel,
                "rows": int(len(frame)),
                "columns": [str(c) for c in frame.columns],
            })
            lines.append(f"- [{tbl_name}]({rel}): {len(frame):,} rows")
        lines.append("")

    if figures:
        fig_dir = destination / "figures"
        fig_dir.mkdir(exist_ok=True)
        lines.extend(["## Figures", ""])
        for fig_name, fig in figures.items():
            path = fig_dir / f"{fig_name}.png"
            try:
                fig.savefig(path, dpi=150)
            finally:
                fig.clear()
            rel = path.relative_to(destination).as_posix()
            manifest["figures"].append(rel)
            lines.extend([f"![{fig_name}]({rel})", ""])

    merged_meta = {
        **embedded_meta,
        **(metadata or {}),
        "sections": {k: v["metadata"] for k, v in sections.items()},
    }
    (destination / "metadata.json").write_text(
        json.dumps(json_safe(merged_meta), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return destination
