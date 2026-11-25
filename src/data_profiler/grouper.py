"""Generalized primary/secondary/rare code grouper (e.g., currency, region, channel)."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PrimaryCodeGrouper:
    """Classify codes into primary vs secondary vs missing without conflating missing with secondary."""

    primary_values: Collection[Any]
    rare_values: Collection[Any] = ()
    rare_label: str = "other"
    missing_label: str = "missing"
    prefix: str = "code"

    def flags(self, series: pd.Series) -> pd.DataFrame:
        """Return nullable boolean flags (`is_primary`, `is_secondary`, `is_missing`)."""
        known = series.notna()
        is_primary = series.isin(self.primary_values).astype("boolean").mask(~known)
        return pd.DataFrame(
            {
                f"{self.prefix}_is_primary": is_primary,
                f"{self.prefix}_is_secondary": ~is_primary,
                f"{self.prefix}_missing": ~known,
            },
            index=series.index,
        )

    def group(self, series: pd.Series) -> pd.Series:
        """Group rare values into `rare_label` and nulls into `missing_label`, preserving all other codes."""
        known = series.notna()
        as_str = series.astype("string")
        if self.rare_values:
            as_str = as_str.mask(known & series.isin(self.rare_values), self.rare_label)
        return as_str.fillna(self.missing_label).rename(f"{self.prefix}_group")
