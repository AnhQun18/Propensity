"""Publish computed results under docs/<dataset>, without rerunning statistics."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from .result import ProfileResult, DatasetResult


def _json_clean(value):
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, (type, np.dtype)):
        return str(value)
    return value


def _markdown(table, limit=20):
    if table.empty:
        return "No observations or no applicable result."
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    text = ["| " + " | ".join(cell(c) for c in table.columns) + " |",
            "| " + " | ".join("---" for _ in table.columns) + " |"]
    text.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in table.head(limit).itertuples(index=False, name=None))
    if len(table) > limit:
        text.append(f"\nShowing {limit} of {len(table):,} rows; see CSV for the complete result.")
    return "\n".join(text)


def _write_profile(result, root, table_dir, figure_dir, *, plots):
    table_dir.mkdir(parents=True, exist_ok=True)
    parts = [f"# {result.spec.dataset} — descriptive statistics and EDA",
             f"Row unit: {result.spec.row_unit}. Total rows: {result.metadata['total_rows']:,}; analyzed rows: {result.metadata['analyzed_rows']:,}.",
             "Missing values and outliers are reported without modifying input data. Numeric moments use finite values. No model is trained."]
    for finding in result.findings:
        parts.append(f"> {finding['code']}: {finding['message']}")
    for name, table in result.tables.items():
        path = table_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        parts.extend([f"## {name.replace('_', ' ')}", f"Scope: {result.metadata['table_scopes'][name]}.",
                      f"[Download CSV]({path.relative_to(root).as_posix()})", _markdown(table)])
    if plots:
        from .visualization import numerical, categorical, quality, temporal, correlation, target
        import matplotlib.pyplot as plt
        figure_dir.mkdir(parents=True, exist_ok=True)
        def save(name, fig, note):
            path = figure_dir / f"{name}.png"
            try:
                fig.savefig(path, bbox_inches="tight", dpi=130)
            finally:
                plt.close(fig)
            parts.extend([f"![{name}]({path.relative_to(root).as_posix()})", note])
        parts.append("## Figures")
        sample_scope = f"Analyzed rows: {result.metadata['analyzed_rows']:,}; mode: {result.spec.mode}."
        if not result.tables["missing"].empty:
            save("missing", quality.missing_bar(result.tables["missing"]), "Missing percentages use the full scan.")
        # Bound exported chart counts; complete numerical tables remain available.
        for i, row in enumerate(result.tables.get("numeric_summary", pd.DataFrame()).head(6).itertuples()):
            save(f"histogram_{i}", numerical.histogram(result._frame[row.column]), f"{row.column}. {sample_scope}")
            save(f"boxplot_{i}", numerical.boxplot(result._frame[row.column]), f"{row.column}. No clipping. {sample_scope}")
        frequencies = result.tables.get("categorical_summary", pd.DataFrame())
        if not frequencies.empty:
            for i, column in enumerate(frequencies.column.unique()[:6]):
                save(f"categories_{i}", categorical.bar(frequencies[frequencies.column == column]), f"{column}. {sample_scope}")
        counts = result.tables.get("temporal_counts", pd.DataFrame())
        if not counts.empty:
            for i, column in enumerate(counts.column.unique()[:6]):
                save(f"activity_{i}", temporal.activity(counts[counts.column == column]), f"{column}. {sample_scope} Zero means no recorded rows in that period.")
        corr = result.tables.get("correlations", pd.DataFrame())
        if not corr.empty:
            save("correlation", correlation.heatmap(corr), f"Association does not imply causation. {sample_scope}")
        labels = result.tables.get("target_summary", pd.DataFrame())
        if not labels.empty:
            for i, column in enumerate(labels.target.unique()[:6]):
                save(f"target_{i}", target.distribution(labels[labels.target == column]), f"{column}. Missing labels are not negative labels. {sample_scope}")
    return "\n\n".join(parts) + "\n"


def export_report(result, output_dir=None, *, plots=False, notebook=False):
    """Export tables/metadata/Markdown. Optional plots require the plots extra.

    Notebook export is a static, portable report with relative image links,
    not an executable reconstruction of arbitrary caller code.
    """
    if not isinstance(result, (ProfileResult, DatasetResult)):
        raise TypeError("Expected ProfileResult or DatasetResult")
    name = result.spec.dataset if isinstance(result, ProfileResult) else result.dataset
    root = Path(output_dir) if output_dir is not None else Path("docs") / name
    root.mkdir(parents=True, exist_ok=True)
    if isinstance(result, ProfileResult):
        text = _write_profile(result, root, root / "tables", root / "figures", plots=plots)
        metadata = {**result.metadata, "findings": result.findings}
    else:
        sections = [f"# {name} — multi-table EDA", "Tables are profiled separately. Relationship checks do not merge data."]
        metadata = {"dataset": name, "tables": {}}
        for table_name, item in result.profiles.items():
            text = _write_profile(item, root, root / "tables" / table_name, root / "figures" / table_name, plots=plots)
            sections.append(f"## Table: {table_name}\n\n" + text)
            metadata["tables"][table_name] = {**item.metadata, "findings": item.findings}
        if not result.relationships.empty:
            directory = root / "relationships"
            directory.mkdir(exist_ok=True)
            result.relationships.to_csv(directory / "key_checks.csv", index=False)
            coverage_columns = [c for c in ["relationship", "left", "right", "left_matched_rows", "left_coverage_pct", "scope"] if c in result.relationships]
            result.relationships[coverage_columns].to_csv(directory / "coverage.csv", index=False)
            sections.extend(["## Relationship checks", _markdown(result.relationships)])
        text = "\n\n".join(sections) + "\n"
    (root / "README.md").write_text(text, encoding="utf-8")
    (root / "metadata.json").write_text(json.dumps(_json_clean(metadata), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if notebook:
        import nbformat
        nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(text)])
        nbformat.write(nb, root / "eda_report.ipynb")
    return root
