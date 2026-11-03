"""Event summaries and explicit, bounded customer-by-day sparse matrices."""
import pandas as pd
from ..utils.stats import tokens, missing_mask
from ..descriptive.temporal import parse_time


def summarize(frame, user, time, time_options=None):
    parsed = parse_time(frame[time], time_options)
    valid = ~missing_mask(frame[user]) & parsed.notna()
    events = pd.DataFrame({"user": tokens(frame.loc[valid, user]), "time": parsed[valid]})
    events["day"] = events.time.dt.normalize()
    events = events.sort_values(["user", "time"])
    events["gap_seconds"] = events.groupby("user").time.diff().dt.total_seconds()
    return events.groupby("user", sort=False).agg(
        event_n=("time", "size"), active_days=("day", "nunique"), first_event=("time", "min"),
        last_event=("time", "max"), mean_gap_seconds=("gap_seconds", "mean")).reset_index()


def customer_day_matrix(frame, user, time, *, start, end, time_options=None,
                        binary=True, max_days=3660):
    """Inclusive observed interval only. Caller must supply verified coverage bounds."""
    parsed = parse_time(frame[time], time_options)
    bounds = pd.DatetimeIndex([pd.Timestamp(start), pd.Timestamp(end)])
    if parsed.dt.tz is not None:
        bounds = bounds.tz_localize(parsed.dt.tz) if bounds.tz is None else bounds.tz_convert(parsed.dt.tz)
    elif bounds.tz is not None:
        raise ValueError("Bounds and timestamps must use compatible timezones")
    lower, upper = bounds.normalize()
    if upper < lower or (upper - lower).days + 1 > max_days:
        raise ValueError("Invalid or excessively large matrix interval")
    valid_times = parsed.dropna()
    if valid_times.empty or lower < valid_times.min().normalize() or upper > valid_times.max().normalize():
        raise ValueError("Matrix interval must lie within the observed time range")
    days = pd.date_range(lower, upper, freq="D")
    valid = parsed.notna() & ~missing_mask(frame[user])
    valid &= parsed.dt.normalize().between(lower, upper)
    users = tokens(frame.loc[valid, user])
    user_codes, user_index = pd.factorize(users, sort=True)
    day_codes = days.get_indexer(parsed[valid].dt.normalize())
    from scipy.sparse import coo_matrix
    import numpy as np
    matrix = coo_matrix((np.ones(len(user_codes), dtype=np.int64), (user_codes, day_codes)),
                        shape=(len(user_index), len(days))).tocsr()
    if binary:
        matrix.data[:] = 1
    return matrix, pd.Index(user_index, name=user), days
