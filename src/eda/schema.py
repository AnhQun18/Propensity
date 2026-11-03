"""Resolve declared semantic roles without guessing business meaning."""
import pandas as pd


def resolve_schema(frame, config):
    if len(frame.columns) == 0:
        raise ValueError("At least one column is required")
    if not frame.columns.is_unique or any(not isinstance(c, str) for c in frame.columns):
        raise ValueError("Column names must be unique strings")
    declared = (config.id_columns + (config.numeric_columns or []) +
                (config.categorical_columns or []) + list(config.datetime_columns) +
                list(config.targets) + config.exclude_columns + config.group_columns +
                list(config.column_metadata) + list(config.valid_ranges) +
                [c for c in (config.event_user, config.event_time, config.event_type) if c is not None])
    missing = sorted(set(declared).difference(frame.columns))
    if missing:
        raise ValueError(f"Columns not found: {missing}")
    role_sets = [set(config.id_columns), set(config.targets), set(config.datetime_columns),
                 set(config.exclude_columns), set(config.numeric_columns or []),
                 set(config.categorical_columns or [])]
    for i, role in enumerate(role_sets):
        for other in role_sets[i + 1:]:
            if role & other:
                raise ValueError(f"Conflicting column roles: {sorted(role & other)}")
    reserved = set().union(*role_sets[:4])
    numeric = config.numeric_columns
    if numeric is None:
        numeric = [c for c in frame if c not in reserved and
                   c not in (config.categorical_columns or []) and
                   pd.api.types.is_numeric_dtype(frame[c]) and
                   not pd.api.types.is_bool_dtype(frame[c])]
    if any(not pd.api.types.is_numeric_dtype(frame[c]) for c in numeric):
        raise ValueError("numeric_columns must have numeric dtypes; declare reader dtypes explicitly")
    categorical = config.categorical_columns
    if categorical is None:
        categorical = [c for c in frame if c not in reserved and c not in numeric and
                       not pd.api.types.is_datetime64_any_dtype(frame[c])]
    roles = {c: "unassigned" for c in frame}
    for role, columns in [("id", config.id_columns), ("numeric", numeric),
                          ("categorical", categorical), ("datetime", config.datetime_columns),
                          ("target", config.targets), ("excluded", config.exclude_columns)]:
        for column in columns:
            roles[column] = role
    return roles
