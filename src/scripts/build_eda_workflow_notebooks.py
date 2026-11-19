"""Copy EDA analyses into numbered, topical notebooks without executing raw data."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "notebooks"

# Each entry is one coherent analysis. Long analyses stand alone; shorter related
# checks share a notebook. Source files remain in src as requested.
GROUPS = [
    ("02_data_inventory", "Kiểm kê dữ liệu và khóa nối", [
        "src/audit_mbd_inventory.py", "src/audit_mbd_join_questions.py"]),
    ("05_label_cohorts", "Nhãn mua và cohort khách hàng", [
        "src/audit_any_product_cohorts.py", "src/plot_target_any.py",
        "src/scripts/analyze_target_focused_moscow.py",
        "src/mbd_mini_eda/descriptive/targets.py"]),
    ("06_temporal_coverage", "Độ phủ thời gian của các nguồn", [
        "src/audit_mbd_temporal_coverage.py"]),
    ("07_full_history_missingness", "Thiếu log trong toàn bộ lịch sử", [
        "src/audit_mbd_full_history_missingness.py",
        "src/scripts/analyze_full_population_temporal_eda.py"]),
    ("08_transaction_quality", "Chất lượng dữ liệu giao dịch", [
        "src/scripts/audit_trx_nulls.py", "src/scripts/audit_trx_zero_negative.py",
        "src/scripts/audit_trx_outliers.py",
        "src/scripts/extend_eda_trx_quality.py",
        "src/scripts/check_trans_count_target_history.py"]),
    ("09_transaction_timing", "Nhịp giao dịch và thời điểm sự kiện", [
        "src/scripts/audit_trx_monthly_coverage.py",
        "src/scripts/extend_eda_event_timing.py",
        "src/scripts/extend_eda_daily_moscow.py"]),
    ("10_transaction_amount", "Amount và hành vi tài chính quanh nhãn", [
        "src/scripts/analyze_amount11_target_quantiles.py",
        "src/scripts/analyze_purchase_event_financial_eda.py"]),
    ("11_geo_quality", "Chất lượng, độ phủ và hoạt động GEO", [
        "src/scripts/audit_geo_special_values.py",
        "src/scripts/analyze_geo_cell_sizes.py",
        "src/scripts/analyze_geo_temporal_coverage.py",
        "src/scripts/analyze_geo_calendar_patterns.py",
        "src/scripts/analyze_geo_frequency_recency.py",
        "src/scripts/analyze_geo_frequency_deciles.py",
        "src/scripts/analyze_geo_interarrival.py",
        "src/scripts/analyze_geo_trend_purchase.py"]),
    ("12_geo_purchase", "GEO và nhãn mua theo không gian", [
        "src/scripts/analyze_geo_spatial_purchase_moscow.py",
        "src/scripts/analyze_geo_visitor_cell_rates.py"]),
    ("13_geo_visuals", "Biểu đồ và kiểm tra GEO", [
        "src/scripts/plot_geo_g4_concentration.py",
        "src/scripts/plot_geo_purchase_event_study.py",
        "src/scripts/extend_eda_geo_population.py",
        "src/scripts/report_geo_spatial_purchase_moscow.py"]),
    ("14_dialog_timing", "Thời gian, hình học và phân cụm DIALOG", [
        "src/scripts/audit_dialog_timeline.py",
        "src/scripts/plot_dialog_label_date_transition.py",
        "src/scripts/extend_eda_dialog_geometry.py",
        "src/scripts/extend_eda_dialog_clusters.py"]),
    ("15_cross_source", "Quan hệ giữa các nguồn dữ liệu", [
        "src/scripts/analyze_cross_table_relationships.py"]),
    ("16_customer_lifecycle", "Hành vi khách hàng qua thời gian", [
        "src/scripts/extend_eda_lifecycle.py",
        "src/scripts/extend_eda_monthly_features.py",
        "src/scripts/extend_eda_paired_checks.py"]),
    ("17_eda_figures", "Hoàn thiện biểu đồ EDA theo Moscow", [
        "src/scripts/complete_eda_figures_moscow.py"]),
]


def source_cells(relative_path: str):
    """Split a Python file at top-level statements, retaining exact source order."""
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source, filename=relative_path)
    cuts = [1]
    current_start = 1
    for node in tree.body:
        if node.end_lineno - current_start >= 110 and node.lineno > current_start:
            cuts.append(node.lineno)
            current_start = node.lineno
    cuts.append(len(lines) + 1)
    chunks = []
    for start, end in zip(cuts, cuts[1:]):
        chunk = "".join(lines[start - 1:end - 1]).strip("\n")
        if chunk.strip():
            ast.parse(chunk, filename=relative_path)
            chunks.append(nbf.v4.new_code_cell(chunk))
    return chunks, hashlib.sha256(source.encode("utf-8")).hexdigest()


def build_one(stem: str, title: str, files: list[str]):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["eda_sources"] = {}
    nb.cells = [nbf.v4.new_markdown_cell(
        f"# {title}\n\n"
        "Notebook chứa mã phân tích theo đúng thứ tự từ các file nguồn trong `src/`. "
        "Chạy với kernel của `.venv` tại repo. Các bước có thể đọc toàn bộ raw và ghi "
        "vào `outputs/`; kiểm tra phạm vi và quy ước thời gian của từng phần trước "
        "khi diễn giải. Mã cũ dùng UTC hoặc mẫu là phương pháp cũ, chưa được coi "
        "là kết quả Moscow/toàn bộ khách hàng. Script nguồn vẫn được giữ để đối chiếu."
    ), nbf.v4.new_code_cell(
        "from pathlib import Path\nimport sys\nimport time\n"
        "import pandas as pd\n"
        "from IPython.display import display, Image, SVG, Markdown\n"
        "ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] "
        "if (p / 'pyproject.toml').exists())\n"
        "sys.path.insert(0, str(ROOT / 'src'))\n"
        "sys.path.insert(0, str(ROOT / 'src/scripts'))\n"
        "sys.path.insert(0, str(ROOT / 'src/mbd_mini_eda_fixes'))\n"
        "def show_generated_results(started_ns, source):\n"
        "    changed = sorted((p for base in (ROOT / 'outputs', ROOT / 'docs') "
        "if base.exists() for p in base.rglob('*') "
        "if p.is_file() and p.stat().st_mtime_ns >= started_ns), "
        "key=lambda p: p.as_posix())\n"
        "    display(Markdown(f'### Kết quả: `{source}`'))\n"
        "    if not changed:\n"
        "        display(Markdown('Không có file kết quả mới từ phần này.'))\n"
        "        return\n"
        "    images = [p for p in changed if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.svg')]\n"
        "    png_stems = {p.with_suffix('') for p in images if p.suffix.lower() == '.png'}\n"
        "    for p in images:\n"
        "        if p.suffix.lower() == '.svg' and p.with_suffix('') in png_stems:\n"
        "            continue\n"
        "        display(Markdown(f'**{p.relative_to(ROOT).as_posix()}**'))\n"
        "        display(SVG(filename=str(p)) if p.suffix.lower() == '.svg' else Image(filename=str(p)))\n"
        "    tables = [p for p in changed if p.suffix.lower() == '.csv']\n"
        "    for p in tables:\n"
        "        display(Markdown(f'**{p.relative_to(ROOT).as_posix()}** — 5 dòng đầu'))\n"
        "        try:\n"
        "            display(pd.read_csv(p, nrows=5))\n"
        "        except Exception as exc:\n"
        "            display(Markdown(f'Không đọc được bảng: `{type(exc).__name__}: {exc}`'))\n"
        "    others = [p.relative_to(ROOT).as_posix() for p in changed "
        "if p not in images and p not in tables]\n"
        "    if others:\n"
        "        display(Markdown('File kết quả khác: ' + ', '.join(f'`{p}`' for p in others)))"
    )]
    for relative_path in files:
        nb.cells.append(nbf.v4.new_markdown_cell(
            f"## {Path(relative_path).stem.replace('_', ' ').title()}\n\n"
            f"Nguồn: `{relative_path}`. Chạy các ô của phần này theo thứ tự."
        ))
        module_name = relative_path.removeprefix("src/").removesuffix(".py").replace("/", ".")
        package_name = module_name.rpartition(".")[0]
        module_source = relative_path.startswith("src/mbd_mini_eda/") or relative_path.startswith("src/mbd_mini_eda_fixes/")
        mode = (f"__name__ = {module_name!r}\n__package__ = {package_name!r}\n"
                if module_source else "__name__ = '__main__'\n__package__ = None\n")
        argv = "sys.argv = [__file__]\n"
        if relative_path == "src/scripts/audit_trx_zero_negative.py":
            argv = (
                "sys.argv = [__file__, '--output-dir', "
                "str(ROOT / 'outputs/notebook_runs/trx_zero_negative' / "
                "str(time.time_ns()))]\n"
            )
        nb.cells.append(nbf.v4.new_code_cell(
            f"__file__ = str(ROOT / {relative_path!r})\n"
            + argv
            + "__eda_started_ns = time.time_ns()\n" + mode
        ))
        chunks, digest = source_cells(relative_path)
        nb.cells.extend(chunks)
        nb.cells.append(nbf.v4.new_code_cell(
            f"show_generated_results(__eda_started_ns, {relative_path!r})"
        ))
        nb.metadata["eda_sources"][relative_path] = digest
    for index, cell in enumerate(nb.cells):
        cell.id = hashlib.sha256(
            f"{stem}:{index}:{cell.cell_type}:{cell.source}".encode("utf-8")
        ).hexdigest()[:16]
    path = NOTEBOOKS / f"{stem}.ipynb"
    nbf.validate(nb)
    nbf.write(nb, path)
    return path


def main():
    selected = set(sys.argv[1:])
    for stem, title, files in GROUPS:
        if selected and stem not in selected:
            continue
        path = build_one(stem, title, files)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
