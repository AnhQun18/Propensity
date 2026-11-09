"""Audit any-product labels and historical sparsity; no raw edits or model fitting."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/propensity_feasibility/any_product'


def main():
    labels = [f'target_{i}' for i in range(1, 5)]
    target = pq.read_table(ROOT / 'data/MBD-mini/targets',
                           columns=['client_id', 'mon'] + labels).to_pandas()
    assert not target.duplicated(['client_id', 'mon']).any()
    assert target[['client_id', 'mon']].notna().all().all()
    assert all(target[c].dropna().isin([0, 1]).all() for c in labels)
    known_positive = target[labels].eq(1).any(axis=1)
    known_negative = target[labels].eq(0).all(axis=1)
    target['any_product'] = pd.Series(pd.NA, index=target.index, dtype='Int8')
    target.loc[known_positive, 'any_product'] = 1
    target.loc[known_negative, 'any_product'] = 0
    target['month'] = pd.to_datetime(target.mon).dt.to_period('M').astype(str)
    monthly = target.groupby('month').agg(
        rows=('client_id', 'size'), clients=('client_id', 'nunique'),
        labelled=('any_product', 'count'), positives=('any_product', 'sum'))
    monthly['unknown'] = monthly.rows - monthly.labelled
    monthly['negative'] = monthly.labelled - monthly.positives
    monthly['positive_rate'] = monthly.positives / monthly.labelled

    # Trusted, locally generated cache from review_mbd_eda.py. Its month array
    # is object-typed. It counts observed raw logs, not complete-case events.
    cache_path = ROOT / 'outputs/mbd_review_cache/source_month_counts.npz'
    with np.load(cache_path, allow_pickle=True) as cache:
        cube = cache['counts']
        months = cache['months'].astype(str)
        ids = pd.Index(cache['client_ids'])
        fingerprint = str(cache['source_fingerprint'].item())
    assert ids.is_unique and cube.shape == (len(ids), len(months), 3)
    assert months.tolist() == sorted(months.tolist())
    assert set(target.client_id.astype(str)) == set(ids)
    assert np.array_equal(cube.sum(axis=(0, 1), dtype=np.uint64),
                          [38764930, 66295724, 217205])
    cutoff = '2022-12'
    frame = target[target.month.eq(cutoff)].copy()
    positions = ids.get_indexer(frame.client_id.astype(str))
    assert (positions >= 0).all()
    past = cube[:, months < cutoff, :].sum(axis=1, dtype=np.uint64)[positions]
    recent = cube[:, (months >= '2022-09') & (months < cutoff), :].sum(
        axis=1, dtype=np.uint64)[positions]
    total = past.sum(axis=1)
    source_n = (past > 0).sum(axis=1)
    cohorts = {
        'all_clients': np.ones(len(frame), dtype=bool),
        'no_events_all_available_history': total == 0,
        'has_history_no_events_last_3_calendar_months': (total > 0) & (recent.sum(axis=1) == 0),
        'has_events_last_3_calendar_months': recent.sum(axis=1) > 0,
        'trx_only_all_history': (past[:, 0] > 0) & (source_n == 1),
        'no_trx_but_geo_or_dialog_all_history': (past[:, 0] == 0) & (source_n > 0),
        'all_three_sources_all_history': source_n == 3,
        'one_to_nine_trx_all_history': (past[:, 0] >= 1) & (past[:, 0] < 10),
        'no_trx_last_3_calendar_months': recent[:, 0] == 0,
    }
    positive_total = int(frame.any_product.eq(1).sum())
    records = []
    for name, mask in cohorts.items():
        subset = frame.loc[mask, 'any_product']
        positive = int(subset.eq(1).sum())
        known = int(subset.notna().sum())
        records.append(dict(cohort=name, clients=len(subset), positives=positive,
                            unknown=int(subset.isna().sum()),
                            positive_rate=positive / known if known else None,
                            fraction_all_positives=positive / positive_total))
    summary = dict(
        checked_on='2026-11-08', rows=len(target), clients=target.client_id.nunique(),
        positive_rows=int(known_positive.sum()), negative_rows=int(known_negative.sum()),
        unknown_label_rows=int(target.any_product.isna().sum()),
        raw_label_null_cells=int(target[labels].isna().sum().sum()),
        positive_rate=float(known_positive.sum() / target.any_product.notna().sum()),
        expected_client_months=len(ids) * len(monthly),
        missing_client_month_rows=len(ids) * len(monthly) - len(target),
        cohort_cutoff='2022-12-01', cohort_label_month='2022-12',
        cohort_recent_history='[2022-09-01, 2022-12-01); three calendar months, not 84 days',
        cohort_history='All available cached raw events strictly before cutoff; includes partial boundary days.',
        source_fingerprint=fingerprint,
        cohort_note='Rows overlap except no-history / old-history-only / recent-history partition. Descriptive, not threshold tuning.',
        model_fitted=False,
    )
    assert summary['positive_rows'] + summary['negative_rows'] + summary['unknown_label_rows'] == len(target)
    OUT.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUT / 'monthly.csv')
    pd.DataFrame(records).to_csv(OUT / 'cohorts_2022_12.csv', index=False)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == '__main__':
    main()
