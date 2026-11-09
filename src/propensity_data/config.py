"""Validated configuration; all relative paths are relative to the JSON file."""
from dataclasses import asdict, dataclass
from datetime import date
import json
from pathlib import Path
import re


def month_start(value: str) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}", value):
        raise ValueError("Months must use YYYY-MM")
    return date.fromisoformat(value + "-01")


@dataclass(frozen=True)
class PreparationConfig:
    data_root: Path
    output_dir: Path
    first_month: str = "2022-02"
    last_month: str = "2023-01"
    windows_days: tuple[int, ...] = (30, 84)
    memory_limit: str = "6GB"
    threads: int = 2
    label_delay_days: int = 0
    calibration_month: str = "2022-12"
    test_month: str = "2023-01"
    validation_months: tuple[str, ...] = ("2022-09", "2022-10", "2022-11")
    include_dialog_embeddings: bool = False
    embedding_size: int = 768
    embedding_window_days: int = 84

    def __post_init__(self):
        for key in ("data_root", "output_dir"):
            object.__setattr__(self, key, Path(getattr(self, key)).resolve())
        for key in ("windows_days", "validation_months"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        if (self.output_dir == self.data_root or self.data_root in self.output_dir.parents
                or self.output_dir in self.data_root.parents):
            raise ValueError("Output and raw-data trees must not overlap")
        for key in ("threads", "embedding_size", "embedding_window_days"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.label_delay_days) is not int or self.label_delay_days < 0:
            raise ValueError("label_delay_days must be a nonnegative integer")
        if type(self.include_dialog_embeddings) is not bool:
            raise ValueError("include_dialog_embeddings must be boolean")
        if not self.windows_days or any(type(w) is not int or w < 1 for w in self.windows_days):
            raise ValueError("windows_days must contain positive integers")
        if len(set(self.windows_days)) != len(self.windows_days):
            raise ValueError("Duplicate windows")
        if self.include_dialog_embeddings and self.embedding_window_days not in self.windows_days:
            raise ValueError("embedding_window_days must occur in windows_days")
        if not isinstance(self.memory_limit, str) or not re.fullmatch(r"[1-9]\d*(MB|GB)", self.memory_limit):
            raise ValueError("memory_limit must be a positive integer followed by MB or GB")
        months = [self.first_month, self.last_month, self.calibration_month, self.test_month,
                  *self.validation_months]
        for value in months:
            month_start(value)
        if not self.first_month <= self.calibration_month < self.test_month <= self.last_month:
            raise ValueError("Require first <= calibration < test <= last")
        if (not self.validation_months or tuple(sorted(set(self.validation_months))) != self.validation_months
                or not all(self.first_month < m < self.calibration_month for m in self.validation_months)):
            raise ValueError("Validation months must be ordered, unique, and before calibration")

    def to_dict(self):
        result = asdict(self)
        for key in ("data_root", "output_dir"):
            result[key] = str(result[key])
        return result


def load_config(path: str | Path) -> PreparationConfig:
    path = Path(path).resolve()
    options = json.loads(path.read_text(encoding="utf-8-sig"))
    for key in ("data_root", "output_dir"):
        options[key] = (path.parent / options[key]).resolve()
    return PreparationConfig(**options)
