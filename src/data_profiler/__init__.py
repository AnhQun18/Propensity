"""Single-pass out-of-core descriptive profiler for general tabular & multi-table datasets."""

from .engine import SQLiteProfileEngine, SourceProfileState
from .grouper import PrimaryCodeGrouper
from .pipeline import describe_table, profile_dataset, profile_table
from .spec import (
    DatasetSpec,
    ProfileResult,
    ProfilerConfig,
    SectionResult,
    TableSpec,
    auto_infer_table_spec,
)
from .streaming import (
    OnlineMoments,
    UniformSample,
    canonical_value,
    grouped_target_rates,
    iter_row_keys,
    numeric_summary,
)

__all__ = [
    "TableSpec",
    "DatasetSpec",
    "ProfilerConfig",
    "SectionResult",
    "ProfileResult",
    "auto_infer_table_spec",
    "OnlineMoments",
    "UniformSample",
    "canonical_value",
    "iter_row_keys",
    "numeric_summary",
    "grouped_target_rates",
    "PrimaryCodeGrouper",
    "SQLiteProfileEngine",
    "SourceProfileState",
    "profile_dataset",
    "profile_table",
    "describe_table",
]
