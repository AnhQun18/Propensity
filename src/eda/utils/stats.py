"""Pure tabular summaries. No file writes or data cleaning."""
import json
import numpy as np
import pandas as pd


def value_token(value):
    """Typed stable keys for lists, arrays and literal missing-marker strings."""
    if isinstance(value, (np.ndarray, list, tuple)):
        return "sequence:" + json.dumps([value_token(v) for v in value], ensure_ascii=False)
    if isinstance(value, dict):
        return "mapping:" + json.dumps(sorted((str(k), value_token(v)) for k, v in value.items()))
    if value is None or value is pd.NA or value is pd.NaT:
        return "missing:"
    try:
        if bool(pd.isna(value)):
            return "missing:"
    except (TypeError, ValueError):
        pass
    # Equivalent numeric IDs should match across int/float reader dtypes.
    if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
        number = int(value) if np.isfinite(value) and value == int(value) else value
        return "number:" + str(number)
    return type(value).__name__ + ":" + str(value)


def tokens(series):
    return series.map(value_token)


def missing_mask(series):
    # pandas treats list/array-valued cells as present, without traversing vectors.
    return series.isna()


def numeric_summary(frame, columns):
    rows = []
    for column in columns:
        raw = frame[column]
        values = raw.dropna().to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        s = pd.Series(finite)
        q = s.quantile([.01, .25, .5, .75, .95, .99])
        iqr = q.loc[.75] - q.loc[.25]
        lower, upper = q.loc[.25] - 1.5 * iqr, q.loc[.75] + 1.5 * iqr
        rows.append(dict(column=column, count_nonmissing=len(values), finite_n=len(finite),
                         infinite_n=int(np.isinf(values).sum()), mean=s.mean(), std=s.std(),
                         min=s.min(), p01=q.loc[.01], p25=q.loc[.25], median=q.loc[.5],
                         p75=q.loc[.75], p95=q.loc[.95], p99=q.loc[.99], max=s.max(),
                         skewness=s.skew(), iqr_lower=lower, iqr_upper=upper,
                         outlier_n=int(((finite < lower) | (finite > upper)).sum())))
    return pd.DataFrame(rows)


def categorical_summary(frame, columns, top=20):
    rows = []
    for column in columns:
        s = tokens(frame[column])
        counts = s[s != "missing:"].value_counts()
        for value, n in counts.head(top).items():
            rows.append([column, value, "value", n, n / len(s) * 100 if len(s) else np.nan])
        for label, n in [("remaining", int(counts.iloc[top:].sum())), ("missing", int(s.eq("missing:").sum()))]:
            rows.append([column, None, label, n, n / len(s) * 100 if len(s) else np.nan])
    return pd.DataFrame(rows, columns=["column", "value", "kind", "n", "pct_all_rows"])


def target_summary(frame, targets):
    rows = []
    for column, kind in targets.items():
        s = frame[column]
        observed = ~missing_mask(s)
        if kind == "binary" and not s[observed].isin([0, 1]).all():
            raise ValueError(f"{column}: binary labels must be 0/1; missing is unlabelled")
        if kind == "regression":
            if not pd.api.types.is_numeric_dtype(s):
                raise ValueError(f"{column}: regression target must be numeric")
            continue
        for value, n in tokens(s).value_counts().items():
            unlabelled = value == "missing:"
            rows.append(dict(target=column, target_type=kind, value=value, n=n,
                             labelled_n=int(observed.sum()), missing_n=int((~observed).sum()),
                             pct_labelled=np.nan if unlabelled or not observed.sum() else n / observed.sum() * 100))
    return pd.DataFrame(rows)


def grouped_summary(frame, groups, numeric, targets):
    summaries, rates = [], []
    for group in groups:
        keys = tokens(frame[group])
        for value, part in frame.groupby(keys, sort=False):
            for column in numeric:
                values = part[column].replace([np.inf, -np.inf], np.nan)
                summaries.append(dict(group_column=group, group=value, column=column,
                                      rows=len(part), count=values.count(), mean=values.mean(), median=values.median()))
            for target, kind in targets.items():
                if kind == "binary":
                    n = part[target].count()
                    positive = part[target].eq(1).sum()
                    rates.append(dict(group_column=group, group=value, target=target,
                                      labelled_n=n, positive_n=positive, rate=positive / n if n else np.nan))
    return pd.DataFrame(summaries), pd.DataFrame(rates)


def correlations(frame, columns):
    clean = frame[columns].replace([np.inf, -np.inf], np.nan)
    matrix = clean.corr(method="spearman")
    rows = []
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            rows.append(dict(column_a=a, column_b=b, spearman=matrix.loc[a, b],
                             pair_n=int((clean[a].notna() & clean[b].notna()).sum())))
    return pd.DataFrame(rows, columns=["column_a", "column_b", "spearman", "pair_n"])
