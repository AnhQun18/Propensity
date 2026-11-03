"""Audit join keys without joining raw tables or changing row counts."""
import pandas as pd
from ..utils.stats import tokens, missing_mask


def audit(left, right, relationship):
    def keys(frame, columns):
        absent = set(columns).difference(frame.columns)
        if absent:
            raise ValueError(f"Missing relationship columns: {sorted(absent)}")
        valid = ~pd.concat([missing_mask(frame[c]) for c in columns], axis=1).any(axis=1)
        values = frame.loc[valid, columns].apply(tokens)
        return pd.Series(list(values.itertuples(index=False, name=None)), dtype=object), int((~valid).sum())
    a, missing_a = keys(left, relationship.left_on)
    b, missing_b = keys(right, relationship.right_on)
    ac, bc = a.value_counts(), b.value_counts()
    matched_rows = int(a.isin(set(b)).sum())
    expected_rows = sum(int(n) * int(bc.get(key, 0)) for key, n in ac.items())
    unique_a, unique_b = not a.duplicated().any(), not b.duplicated().any()
    expected = relationship.cardinality
    passes = ((expected not in {"one_to_one", "one_to_many"} or unique_a) and
              (expected not in {"one_to_one", "many_to_one"} or unique_b))
    return pd.DataFrame([dict(relationship=relationship.name, left=relationship.left, right=relationship.right,
                              cardinality=expected, cardinality_pass=passes,
                              left_missing_keys=missing_a, right_missing_keys=missing_b,
                              left_duplicate_keys=int(a.duplicated().sum()), right_duplicate_keys=int(b.duplicated().sum()),
                              left_matched_rows=matched_rows,
                              left_coverage_pct=matched_rows / len(left) * 100 if len(left) else float("nan"),
                              projected_inner_rows=expected_rows)])
