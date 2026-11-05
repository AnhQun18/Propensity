"""Cấu hình I/O, lấy mẫu và lựa chọn căn chỉnh khoảng nhãn."""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class MBDMiniConfig:
    """Các đường dẫn mặc định tính từ cwd; đường dẫn JSON tính từ file JSON.

    history_days là các cửa sổ dự kiến so sánh, không phải điều kiện lọc khách.
    Việc tạo feature theo tháng cần thêm bảng label_windows đã xác minh.
    """

    data_root: Path = Path("data/MBD-mini")
    output_dir: Path = Path("docs/mbd_mini_eda")
    batch_size: int = 100_000
    history_days: tuple[int, ...] = (30, 84, 180)
    random_seed: int = 42
    sample_size: int = 5000
    max_categories: int = 20
    embedding_size: int = 768
    duplicate_mode: str = "sample"
    label_alignment: str = "unresolved"
    label_windows_path: Path | None = None
    analysis_months: tuple[str, ...] = ()
    min_group_size: int = 30
    min_positive_count: int = 5
    correlation_sample_size: int = 50000
    max_batches_per_source: int | None = None
    work_dir: Path | None = None
    timezone: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("batch_size phải là số nguyên dương")
        if not isinstance(self.history_days, (list, tuple)) or not self.history_days:
            raise ValueError("history_days phải là danh sách cửa sổ không rỗng")
        if any(isinstance(day, bool) or not isinstance(day, int) or day < 1 for day in self.history_days):
            raise ValueError("Mỗi history_days phải là số nguyên dương")
        if len(set(self.history_days)) != len(self.history_days):
            raise ValueError("history_days không được trùng lặp")
        object.__setattr__(self, "history_days", tuple(self.history_days))
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int) or self.random_seed < 0:
            raise ValueError("random_seed phải là số nguyên không âm")
        for key in ("sample_size", "max_categories", "embedding_size", "min_group_size",
                    "min_positive_count", "correlation_sample_size"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} phải là số nguyên dương")
        if self.duplicate_mode not in {"exact", "sample", "off"}:
            raise ValueError("duplicate_mode phải là exact, sample hoặc off")
        if self.label_alignment not in {"unresolved", "calendar_month", "explicit"}:
            raise ValueError("label_alignment phải là unresolved, calendar_month hoặc explicit")
        if self.label_alignment == "explicit" and self.label_windows_path is None:
            raise ValueError("explicit yêu cầu label_windows_path")
        for key in ("label_windows_path", "work_dir"):
            value = getattr(self, key)
            if value is not None:
                object.__setattr__(self, key, Path(value))
        if isinstance(self.analysis_months, str):
            raise ValueError("analysis_months phải là danh sách")
        object.__setattr__(self, "analysis_months", tuple(self.analysis_months))
        if self.max_batches_per_source is not None:
            value = self.max_batches_per_source
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("max_batches_per_source phải là số nguyên dương hoặc null")

    def to_dict(self) -> dict:
        from dataclasses import asdict
        result = asdict(self)
        for key in ("data_root", "output_dir", "label_windows_path", "work_dir"):
            result[key] = str(result[key]) if result[key] is not None else None
        result["history_days"] = list(self.history_days)
        result["analysis_months"] = list(self.analysis_months)
        return result


def load_config(path: str | Path) -> MBDMiniConfig:
    """Chỉ đọc JSON cấu hình; không kiểm kê hoặc mở file dữ liệu."""
    path = Path(path).resolve()
    options = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(options, dict):
        raise ValueError("Cấu hình phải là một JSON object")
    for key, default in (("data_root", "../data/MBD-mini"), ("output_dir", "../docs/mbd_mini_eda")):
        value = Path(options.get(key, default))
        options[key] = value.resolve() if value.is_absolute() else (path.parent / value).resolve()
    for key in ("label_windows_path", "work_dir"):
        if options.get(key) is not None:
            options[key] = (path.parent / options[key]).resolve()
    return MBDMiniConfig(**options)
