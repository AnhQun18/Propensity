"""Xuất bảng tổng hợp, PNG và báo cáo có phạm vi tính toán rõ ràng."""

import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

from .models import AnalysisResult


def render_plan(plan: dict) -> str:
    """Định dạng kế hoạch thành Markdown; không ghi file."""
    lines = [
        "# Kế hoạch thống kê mô tả và EDA MBD-mini",
        "",
        "Trạng thái: các module đã triển khai. Dùng lệnh run để phân tích.",
        "Lệnh này chỉ in kế hoạch, không đọc dữ liệu gốc hoặc tạo báo cáo kết quả.",
        "",
        f"Dữ liệu dự kiến: `{plan['config']['data_root']}`",
        f"Đầu ra dự kiến: `{plan['config']['output_dir']}`",
        "",
    ]
    for index, step in enumerate(plan["steps"], start=1):
        lines.extend([
            f"## {index}. {step['title']}",
            "",
            f"- Module: `{step['module']}.run(context)`",
            f"- Nhóm: {step['stage']}; cấp phân tích: {step['grain']}",
            "- Đầu vào: " + ", ".join(step["inputs"]),
            "- Bảng đầu ra: " + ", ".join(step["outputs"]),
        ])
        if step["depends_on"]:
            lines.append("- Sau các bước: " + ", ".join(step["depends_on"]))
        if step["required_artifacts"]:
            lines.append("- Bảng nội bộ cần có: " + ", ".join(step["required_artifacts"]))
        if step["creates_artifacts"]:
            lines.append("- Tạo bảng nội bộ: " + ", ".join(step["creates_artifacts"]))
        lines.append("")
    return "\n".join(lines)


def _safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", value) or value in {".", ".."}:
        raise ValueError(f"Tên output không hợp lệ: {value}")
    return value


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return value


def export_report(result: AnalysisResult, output_dir: str | Path | None = None, *, plots=False) -> Path:
    """Xuất kết quả đã tính; manifest.json chỉ rõ các file của lần xuất hiện tại."""
    from .visualization import descriptive_figures

    destination = Path(output_dir or result.config.output_dir).resolve()
    if destination.is_relative_to(result.config.data_root.resolve()):
        raise ValueError("Không xuất báo cáo bên trong thư mục dữ liệu nguồn")
    figures = {}
    if plots:
        figures.update(descriptive_figures(result))
    # Kiểm tra trước khi ghi để tránh xuất một báo cáo chứa bảng cấp cá nhân.
    for section, item in result.sections.items():
        _safe_name(section)
        for name, frame in item.tables.items():
            _safe_name(name)
            if "client_id" in frame:
                raise ValueError(f"Chỉ xuất bảng tổng hợp, không xuất client_id: {section}.{name}")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"tables": [], "figures": [], "metadata": "metadata.json", "report": "report.md"}
    lines = ["# Bảy bước thống kê mô tả MBD-mini", "",
             "Phạm vi sự kiện: **" + result.metadata.get("event_scope", "unknown") + "**.", "",
             "Bảng khách hàng và nhãn được đọc toàn bộ. Mỗi mục ghi grain và scope trong metadata.json.",
             "Mean/std/min/max số tiền dùng toàn bộ giá trị hữu hạn đã quét; phân vị, outlier và histogram dùng mẫu ngẫu nhiên được khai báo.",
             "Kiểm tra trùng có scope riêng; số trùng trên mẫu không suy rộng thành số trùng toàn bộ.", "",
             "Timezone: **" + (result.config.timezone or "naive_as_stored") + "**.",
             "Thời gian phân tích: " + str(result.metadata.get("duration_seconds", "unknown")) + " giây (chưa gồm xuất báo cáo).", "",
             "## Quan hệ dữ liệu", "", "```mermaid", "erDiagram",
             "    clients ||--o{ targets : client_id", "    clients ||--o{ transactions : client_id",
             "    clients ||--o{ geo : client_id", "    clients ||--o{ dialog : client_id", "```", ""]
    if result.metadata.get("event_scope") == "prefix":
        lines.extend(["**Đây là lần chạy thử trên prefix sự kiện, không đại diện cho quần thể.**", ""])
    quality = result.sections.get("descriptive.quality")
    targets = result.sections.get("descriptive.targets")
    lines.extend(["## Các số liệu chính", ""])
    if quality is not None:
        missing = quality.tables.get("missing_values", pd.DataFrame())
        if not missing.empty:
            top = missing.sort_values("missing_rate", ascending=False).iloc[0]
            lines.append(f"- Thuộc tính có tỷ lệ null cao nhất: {top.source}.{top.column}, "
                         f"{int(top.missing_n):,}/{int(top.rows):,} dòng ({top.missing_rate:.2%}).")
        coverage = quality.tables.get("key_coverage", pd.DataFrame())
        for row in coverage.itertuples(index=False):
            lines.append(f"- {row.source}: {row.customers_with_valid_time_log:,}/{row.reference_customers:,} "
                         "khách có log với ID/thời gian hợp lệ trong phần đã quét.")
    if targets is not None:
        prevalence = targets.tables.get("target_prevalence", pd.DataFrame())
        for row in prevalence.itertuples(index=False):
            lines.append(f"- {row.target}: {row.positive_n:,}/{row.labelled_n:,} nhãn quan sát là dương ({row.rate:.4%}).")
    lines.extend(["", "Các số liệu mô tả dùng mẫu số khác nhau; xem từng bảng trước khi so sánh.", ""])
    for name, item in result.sections.items():
        lines.extend(["## " + name, "", "Trạng thái: " + item.metadata.get("status", "completed") + ".", ""])
        for finding in item.findings:
            lines.extend(["- " + finding])
        lines.append("")
        folder = destination / "tables" / name
        for table, frame in item.tables.items():
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (table + ".csv")
            frame.to_csv(path, index=False, encoding="utf-8-sig")
            relative = path.relative_to(destination).as_posix()
            manifest["tables"].append({"section": name, "name": table, "path": relative, "rows": len(frame),
                                       "columns": list(frame.columns)})
            lines.append(f"- [{table}]({relative}): {len(frame):,} dòng tổng hợp.")
        lines.append("")
    if figures:
        folder = destination / "figures"
        folder.mkdir(exist_ok=True)
        lines.extend(["## Biểu đồ", ""])
        for name, fig in figures.items():
            path = folder / (_safe_name(name) + ".png")
            try:
                fig.savefig(path, dpi=150)
            finally:
                fig.clear()
            relative = path.relative_to(destination).as_posix()
            manifest["figures"].append(relative)
            lines.extend([f"![{name}]({relative})", ""])
    metadata = {**result.metadata, "config": result.config.to_dict(),
                "sections": {key: item.metadata for key, item in result.sections.items()}}
    (destination / "metadata.json").write_text(json.dumps(_json_value(metadata), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return destination
