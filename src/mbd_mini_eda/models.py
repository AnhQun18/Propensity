"""Ngữ cảnh, bảng tổng hợp và metadata của từng bước phân tích."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .config import MBDMiniConfig

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class AnalysisContext:
    """Ngữ cảnh một lần phân tích.

    sources trỏ tới các thư mục Parquet, được đọc theo batch ở io.py.
    artifacts chỉ giữ bảng khóa, nhãn và bảng tổng hợp vừa RAM; không giữ toàn bộ
    sự kiện. Tên artifact gồm clients, targets và customer_summary.
    section_results giữ kết quả bảy mục thống kê mô tả.
    """

    config: MBDMiniConfig
    sources: dict[str, Path]
    artifacts: dict[str, "pd.DataFrame"] = field(default_factory=dict)
    section_results: dict[str, "SectionResult"] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict, repr=False)

    def __enter__(self):
        return self

    def close(self):
        engine = self.runtime.pop("engine", None)
        if engine is not None:
            engine.close()

    def __exit__(self, *exc):
        self.close()


@dataclass
class SectionResult:
    """Bảng tổng hợp và nhận xét của một mục phân tích.

    metadata cần ghi population, grain, scope (full/sample), cửa sổ thời gian,
    mẫu số và số quan sát hợp lệ cho các bảng liên quan.
    """

    section: str
    tables: dict[str, "pd.DataFrame"] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Kết quả toàn bộ phiên phân tích; exporter chỉ nhận các kết quả đã tính."""

    config: MBDMiniConfig
    sections: dict[str, SectionResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
