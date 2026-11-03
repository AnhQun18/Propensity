"""Structured results, separate from computation and publication."""
from dataclasses import dataclass, field
import pandas as pd
from .config import DatasetSpec


@dataclass
class ProfileResult:
    spec: DatasetSpec
    tables: dict[str, pd.DataFrame]
    metadata: dict
    findings: list[dict] = field(default_factory=list)
    _frame: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)


@dataclass
class DatasetResult:
    dataset: str
    profiles: dict[str, ProfileResult]
    relationships: pd.DataFrame = field(default_factory=pd.DataFrame)
