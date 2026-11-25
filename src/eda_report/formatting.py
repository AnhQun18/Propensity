"""Number formatting, filename sanitization, and strict JSON serialization."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


def format_number(value: float | int, decimals: int = 0, *, locale: str = "en") -> str:
    """Format numbers for English ('1,234.56') or Vietnamese/European ('1.234,56') displays."""
    formatted = f"{value:,.{decimals}f}"
    if locale.lower() in {"vi", "vn", "eu", "de", "fr"}:
        return formatted.replace(",", "~").replace(".", ",").replace("~", ".")
    return formatted


def safe_name(value: str) -> str:
    """Validate safe output table/section/figure filenames (no path traversal)."""
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", value) or value in {".", ".."}:
        raise ValueError(f"Unsafe output name: {value!r}")
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return value
