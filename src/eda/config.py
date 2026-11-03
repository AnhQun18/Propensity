"""Serializable configuration shared by the Python API and CLI."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re


def safe_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[\w][\w.-]*", value) or value in {".", ".."}:
        raise ValueError(f"Expected a simple dataset/table name, got {value!r}")
    return value


@dataclass
class DatasetSpec:
    dataset: str = "dataset"
    row_unit: str = "observation"
    id_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] | None = None
    categorical_columns: list[str] | None = None
    datetime_columns: dict[str, dict] = field(default_factory=dict)
    targets: dict[str, str] = field(default_factory=dict)
    exclude_columns: list[str] = field(default_factory=list)
    group_columns: list[str] = field(default_factory=list)
    column_metadata: dict[str, dict] = field(default_factory=dict)
    valid_ranges: dict[str, list] = field(default_factory=dict)
    mode: str = "exact"
    chunk_size: int = 100_000
    sample_size: int = 50_000
    random_seed: int = 42
    max_correlation_columns: int = 30
    max_categories: int = 20
    rare_threshold: float = 0.01
    time_frequency: str = "D"
    event_user: str | None = None
    event_time: str | None = None
    event_type: str | None = None
    hash_sources: bool = True
    independent_observations: bool = False

    def __post_init__(self):
        safe_name(self.dataset)
        if self.mode not in {"exact", "approximate"}:
            raise ValueError("mode must be exact or approximate")
        for name in ("chunk_size", "sample_size", "max_correlation_columns", "max_categories"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if any(kind not in {"binary", "multiclass", "regression"} for kind in self.targets.values()):
            raise ValueError("target type must be binary, multiclass or regression")
        if not 0 <= self.rare_threshold <= 1:
            raise ValueError("rare_threshold must be between 0 and 1")
        if (self.event_user is None) != (self.event_time is None):
            raise ValueError("event_user and event_time must be supplied together")
        if self.event_time is not None and self.event_time not in self.datetime_columns:
            raise ValueError("event_time must be declared in datetime_columns")
        for column, bounds in self.valid_ranges.items():
            if len(bounds) != 2 or (bounds[0] is not None and bounds[1] is not None and bounds[0] > bounds[1]):
                raise ValueError(f"Invalid range for {column}")

    def to_dict(self):
        return asdict(self)


@dataclass
class FileSource:
    """One logical table. Multiple paths are concatenated, never joined."""
    paths: list[str]
    format: str = "auto"
    read_options: dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.paths, (str, Path)):
            self.paths = [str(self.paths)]
        if not self.paths:
            raise ValueError("At least one input path is required")
        if self.format not in {"auto", "csv", "parquet"}:
            raise ValueError("format must be auto, csv or parquet")
        if "chunksize" in self.read_options or "iterator" in self.read_options:
            raise ValueError("Use config.chunk_size instead of reader chunksize/iterator")


@dataclass
class Relationship:
    name: str
    left: str
    right: str
    left_on: list[str]
    right_on: list[str]
    cardinality: str = "many_to_one"

    def __post_init__(self):
        safe_name(self.name)
        if not self.left_on or len(self.left_on) != len(self.right_on):
            raise ValueError("Relationship keys must be nonempty lists of equal length")
        if self.cardinality not in {"one_to_one", "one_to_many", "many_to_one", "many_to_many"}:
            raise ValueError("Invalid relationship cardinality")
