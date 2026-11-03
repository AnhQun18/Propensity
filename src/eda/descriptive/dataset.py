"""Dataset size, column inventory and explicit opt-in preview."""
import pandas as pd
from ..utils.stats import tokens


def overview(frame, row_unit="observation"):
    return pd.DataFrame([dict(rows=len(frame), columns=len(frame.columns), row_unit=row_unit,
                              memory_bytes=int(frame.memory_usage(deep=True).sum()))])


def column_profile(frame, roles=None, metadata=None):
    rows = []
    for column in frame:
        s = tokens(frame[column])
        observed = s[s != "missing:"]
        missing = len(s) - len(observed)
        rows.append(dict(column=column, dtype=str(frame[column].dtype),
                         role=(roles or {}).get(column, "unassigned"), missing_n=missing,
                         missing_pct=missing / len(s) * 100 if len(s) else float("nan"),
                         distinct_nonnull=observed.nunique(), constant=observed.nunique() == 1,
                         meaning=(metadata or {}).get(column, {}).get("meaning", ""),
                         unit=(metadata or {}).get(column, {}).get("unit", "")))
    return pd.DataFrame(rows)


def preview(frame, n=5):
    """Return raw records only when explicitly called; not exported by default."""
    if n < 0:
        raise ValueError("n must be nonnegative")
    return frame.head(n).copy()
