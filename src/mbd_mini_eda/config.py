"""Cấu hình bảy bước thống kê mô tả, đọc theo batch và bộ nhớ SQLite."""

from dataclasses import dataclass
import json
from pathlib import Path
import warnings


LEGACY_OPTIONS = frozenset({
    "history_days", "label_alignment", "label_windows_path", "analysis_months",
    "min_group_size", "min_positive_count", "correlation_sample_size",
})


@dataclass(frozen=True)
class MBDMiniConfig:
    """Các đường dẫn mặc định tính từ cwd; đường dẫn JSON tính từ file JSON.

    Batch và cache mặc định dành cho máy 16 GB; không nạp toàn bộ sự kiện vào RAM.
    """

    data_root: Path = Path("data/MBD-mini")
    output_dir: Path = Path("docs/mbd_mini_eda")
    batch_size: int = 250_000
    dialog_batch_size: int = 4096
    sqlite_cache_mb: int = 256
    random_seed: int = 42
    sample_size: int = 5000
    max_categories: int = 20
    embedding_size: int = 768
    duplicate_mode: str = "sample"
    max_batches_per_source: int | None = None
    work_dir: Path | None = None
    timezone: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int) or self.random_seed < 0:
            raise ValueError("random_seed phải là số nguyên không âm")
        for key in ("batch_size", "dialog_batch_size", "sqlite_cache_mb", "sample_size",
                    "max_categories", "embedding_size"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} phải là số nguyên dương")
        if self.duplicate_mode not in {"exact", "sample", "off"}:
            raise ValueError("duplicate_mode phải là exact, sample hoặc off")
        if self.work_dir is not None:
            object.__setattr__(self, "work_dir", Path(self.work_dir))
        if self.max_batches_per_source is not None:
            value = self.max_batches_per_source
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("max_batches_per_source phải là số nguyên dương hoặc null")

    def to_dict(self) -> dict:
        from dataclasses import asdict
        result = asdict(self)
        for key in ("data_root", "output_dir", "work_dir"):
            result[key] = str(result[key]) if result[key] is not None else None
        return result


def load_config(path: str | Path) -> MBDMiniConfig:
    """Chỉ đọc JSON cấu hình; không kiểm kê hoặc mở file dữ liệu."""
    path = Path(path).resolve()
    options = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(options, dict):
        raise ValueError("Cấu hình phải là một JSON object")
    legacy = sorted(set(options) & LEGACY_OPTIONS)
    if legacy:
        warnings.warn("Cấu hình cũ: bỏ qua " + ", ".join(legacy)
                      + "; phiên bản này chỉ chạy 7 bước thống kê mô tả.", UserWarning, stacklevel=2)
        for key in legacy:
            options.pop(key)
    for key, default in (("data_root", "../data/MBD-mini"), ("output_dir", "../docs/mbd_mini_eda")):
        value = Path(options.get(key, default))
        options[key] = value.resolve() if value.is_absolute() else (path.parent / value).resolve()
    for key in ("work_dir",):
        if options.get(key) is not None:
            options[key] = (path.parent / options[key]).resolve()
    return MBDMiniConfig(**options)
