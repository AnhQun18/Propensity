"""Review top transaction users, currency tails, largest amount and daily peaks.

Customer-level CSVs stay in ignored outputs. All-history ranks/quantiles are EDA
references, not thresholds fitted for a historical model cutoff.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json

import duckdb
import numpy as np
import pandas as pd

from mbd_mini_eda.currency import currency_analysis_group


ROOT = Path(__file__).resolve().parents[1]
GROUP = "CASE WHEN currency IS NULL THEN 'missing_currency' WHEN currency IN (5,9,12,16) THEN 'other_currency' ELSE cast(currency AS VARCHAR) END"


def audit(root, output, public):
    output.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    raw = root / 'data/MBD-mini'
    inventory = json.loads((root / 'outputs/mbd_review_cache/source_inventory.json').read_text(encoding='utf-8'))
    for item in inventory['files']:
        stat = (raw / item['path']).stat()
        assert stat.st_size == item['bytes'] and stat.st_mtime_ns == item['mtime_ns'], item['path']
    digest = hashlib.sha256(json.dumps(inventory['files'], sort_keys=True).encode()).hexdigest()
    assert digest == inventory['fingerprint']
    db = duckdb.connect()
    db.execute("SET threads=2")
    db.execute("SET memory_limit='3GB'")
    db.execute(f"ATTACH '{(root / 'outputs/mbd_review_cache/review.duckdb').as_posix()}' AS cache (READ_ONLY)")
    for name, folder in [('targets', 'targets'), ('trx', 'detail/trx')]:
        files = sorted((raw / folder).rglob('*.parquet'))
        db.read_parquet([p.as_posix() for p in files], hive_partitioning=False).create_view(name)
    db.execute('''CREATE TEMP TABLE top_users AS SELECT client_id,sum(events)::BIGINT trx_events
                  FROM cache.cache_month_trx GROUP BY client_id
                  ORDER BY trx_events DESC,client_id LIMIT 50''')
    labels = db.execute('''SELECT t.client_id,t.trx_events,count(*)::BIGINT labelled_months,
        sum(CASE WHEN target_1=1 OR target_2=1 OR target_3=1 OR target_4=1 THEN 1 ELSE 0 END)::BIGINT any_positive_months,
        sum(target_1)::BIGINT target_1_positive_months,sum(target_2)::BIGINT target_2_positive_months,
        sum(target_3)::BIGINT target_3_positive_months,sum(target_4)::BIGINT target_4_positive_months
        FROM top_users t JOIN targets USING(client_id) GROUP BY ALL
        ORDER BY trx_events DESC,client_id''').df()
    assert len(labels) == 50 and labels.labelled_months.eq(12).all()
    labels.insert(0, 'rank', np.arange(1, 51))
    labels.insert(2, 'client_short_id', labels.client_id.str[:12])
    assert labels.client_short_id.is_unique
    labels['any_positive_rate_pct'] = labels.any_positive_months / labels.labelled_months * 100
    for target in range(1, 5):
        labels[f'target_{target}_positive_rate_pct'] = labels[f'target_{target}_positive_months'] / labels.labelled_months * 100
    labels.head(20).to_csv(output / 'top20_trx_customers.csv', index=False)
    print('Top 20 counts and all label rates exported.', flush=True)

    print('Reading transaction types and histories of top 50 users.', flush=True)
    db.execute('''CREATE TEMP TABLE selected_trx AS SELECT client_id,event_time,amount,currency,event_type,event_subtype
                  FROM trx SEMI JOIN top_users USING(client_id)''')
    detail = db.execute('''SELECT client_id,count(*)::BIGINT raw_events,
        min(event_time) first_event,max(event_time) last_event,
        count(DISTINCT cast(event_time AS DATE)) active_days,
        count(*) FILTER(WHERE event_time < TIMESTAMP '2022-01-01') events_before_2022,
        count(*) FILTER(WHERE event_time >= TIMESTAMP '2022-01-01') events_2022,
        median(cast(amount AS DOUBLE)) median_amount_raw,
        median(cast(amount AS DOUBLE)) FILTER(WHERE currency=11) median_amount_currency11
        FROM selected_trx GROUP BY client_id''').df()
    types = db.execute('''SELECT client_id,event_type,event_subtype,count(*)::BIGINT events
        FROM selected_trx GROUP BY ALL ORDER BY client_id,events DESC,event_type,event_subtype''').df()
    types['type_rank'] = types.groupby('client_id').cumcount() + 1
    types.to_csv(output / 'top50_event_type_subtype.csv', index=False)
    dominant = types[types.type_rank.eq(1)][['client_id', 'event_type', 'event_subtype', 'events']].rename(columns={'event_type': 'dominant_event_type', 'event_subtype': 'dominant_event_subtype', 'events': 'dominant_type_events'})
    labels = labels.merge(detail, on='client_id', validate='one_to_one').merge(dominant, on='client_id', validate='one_to_one').sort_values('rank')
    assert labels.raw_events.eq(labels.trx_events).all()
    labels['dominant_type_share_pct'] = labels.dominant_type_events / labels.trx_events * 100
    labels.to_csv(output / 'top50_trx_customers.csv', index=False)
    labels.head(20).to_csv(output / 'top20_trx_customers.csv', index=False)
    db.execute('''CREATE TEMP TABLE max_amount_events AS SELECT * FROM trx
                  WHERE currency=11 AND amount=(SELECT max(amount) FROM trx WHERE currency=11)''')
    maximum = db.execute('SELECT * FROM max_amount_events ORDER BY client_id,event_time').df()
    maximum['amount'] = maximum.amount.astype('float64')
    maximum.to_csv(output / 'maximum_amount_transactions.csv', index=False)
    db.execute('CREATE TEMP TABLE max_owners AS SELECT DISTINCT client_id FROM max_amount_events')
    db.execute('''CREATE TEMP TABLE max_owner_trx AS SELECT client_id,event_time,amount,currency,event_type,event_subtype
                  FROM trx SEMI JOIN max_owners USING(client_id)''')

    print('Computing exact raw, ln(amount) and log1p(amount) quantiles by currency group.', flush=True)
    stats = db.execute(f'''SELECT {GROUP} currency_group,count(*)::BIGINT rows,
        count(*) FILTER(WHERE isfinite(amount))::BIGINT finite_n,
        count(*) FILTER(WHERE amount<=0)::BIGINT nonpositive_n,
        max(cast(amount AS DOUBLE)) max_amount,
        quantile_cont(cast(amount AS DOUBLE),[0.25,0.5,0.75,0.99,0.999]) FILTER(WHERE isfinite(amount)) raw_q,
        quantile_cont(CASE WHEN amount>0 AND isfinite(amount) THEN ln(cast(amount AS DOUBLE)) END,[0.25,0.5,0.75]) ln_q,
        quantile_cont(CASE WHEN amount>=0 AND isfinite(amount) THEN ln(1+cast(amount AS DOUBLE)) END,[0.25,0.5,0.75]) log1p_q
        FROM trx GROUP BY currency_group''').df()
    assert int(stats.rows.sum()) == int(stats.finite_n.sum()) == 38764930
    assert int(stats.loc[stats.currency_group.eq('other_currency'), 'rows'].iloc[0]) == 19
    raw_codes = db.execute('SELECT currency,count(*) AS rows FROM trx GROUP BY currency').df()
    raw_codes['currency_group'] = currency_analysis_group(raw_codes.currency)
    expected_groups = raw_codes.groupby('currency_group').rows.sum().to_dict()
    assert stats.set_index('currency_group').rows.to_dict() == expected_groups
    for prefix, field in [('raw', 'raw_q'), ('ln', 'ln_q'), ('log1p', 'log1p_q')]:
        quantiles = np.stack(stats[field])
        stats[f'{prefix}_q1'], stats[f'{prefix}_median'], stats[f'{prefix}_q3'] = quantiles[:, 0], quantiles[:, 1], quantiles[:, 2]
        stats[f'{prefix}_lower'] = quantiles[:, 0] - 1.5 * (quantiles[:, 2] - quantiles[:, 0])
        stats[f'{prefix}_upper'] = quantiles[:, 2] + 1.5 * (quantiles[:, 2] - quantiles[:, 0])
    stats['raw_p99'] = [q[3] for q in stats.raw_q]
    stats['raw_p999'] = [q[4] for q in stats.raw_q]
    stats['iqr_applicable'] = stats.rows.ge(30) & stats.currency_group.ne('other_currency')
    stats = stats.drop(columns=['raw_q', 'ln_q', 'log1p_q'])
    db.register('fences', stats)
    print('Counting both tails under the raw and logarithmic fences.', flush=True)
    counts = db.execute(f'''SELECT f.currency_group,
        count(*) FILTER(WHERE amount<f.raw_lower OR amount>f.raw_upper)::BIGINT raw_outside,
        count(*) FILTER(WHERE amount>0 AND ln(cast(amount AS DOUBLE))<f.ln_lower)::BIGINT ln_below,
        count(*) FILTER(WHERE amount>0 AND ln(cast(amount AS DOUBLE))>f.ln_upper)::BIGINT ln_above,
        count(*) FILTER(WHERE amount>=0 AND ln(1+cast(amount AS DOUBLE))<f.log1p_lower)::BIGINT log1p_below,
        count(*) FILTER(WHERE amount>=0 AND ln(1+cast(amount AS DOUBLE))>f.log1p_upper)::BIGINT log1p_above,
        count(*) FILTER(WHERE amount>f.raw_p999)::BIGINT above_p999
        FROM trx JOIN fences f ON ({GROUP})=f.currency_group GROUP BY f.currency_group''').df()
    stats = stats.merge(counts, on='currency_group', validate='one_to_one')
    stats['ln_outside'] = stats.ln_below + stats.ln_above
    stats['log1p_outside'] = stats.log1p_below + stats.log1p_above
    for prefix in ['raw', 'ln', 'log1p']:
        source = 'raw_outside' if prefix == 'raw' else prefix + '_outside'
        stats[f'{prefix}_outside_pct'] = stats[source] / stats.finite_n * 100
    masked = [name for name in stats if name.endswith(('_lower', '_upper')) or 'outside' in name or name in ['ln_below', 'ln_above', 'log1p_below', 'log1p_above']]
    stats.loc[~stats.iqr_applicable, masked] = np.nan
    stats.to_csv(public / 'amount_log_iqr_by_currency.csv', index=False)
    q999 = float(stats.loc[stats.currency_group.eq('11'), 'raw_p999'].iloc[0])
    owners = db.execute('''SELECT client_id,count(*)::BIGINT events,
        count(*) FILTER(WHERE currency=11)::BIGINT currency11_events,
        median(cast(amount AS DOUBLE)) FILTER(WHERE currency=11) median_currency11,
        count(*) FILTER(WHERE currency=11 AND amount=?)::BIGINT occurrences_of_max,
        count(*) FILTER(WHERE currency=11 AND amount>?)::BIGINT rows_above_reference_p999,
        sum(cast(amount AS DOUBLE)) FILTER(WHERE currency=11) sum_currency11,
        sum(least(cast(amount AS DOUBLE),?)) FILTER(WHERE currency=11) capped_sum_currency11_eda
        FROM max_owner_trx GROUP BY client_id''', [float(maximum.amount.max()), q999, q999]).df()
    owners['max_to_median_currency11'] = float(maximum.amount.max()) / owners.median_currency11
    owners.to_csv(output / 'maximum_amount_owner_stats.csv', index=False)
    db.execute('''CREATE TEMP TABLE peak_day AS SELECT client_id,event_time,event_type,event_subtype
        FROM trx WHERE event_time>=TIMESTAMP '2022-12-11 00:00:00' AND event_time<TIMESTAMP '2022-12-12 00:00:00' ''')
    peak_clients = db.execute('''SELECT client_id,count(*)::BIGINT events FROM peak_day
        GROUP BY client_id ORDER BY events DESC,client_id''').df()
    assert int(peak_clients.events.sum()) == 65302
    peak_clients.head(50).to_csv(output / 'peak_day_top50_customers.csv', index=False)
    peak_types = db.execute('''SELECT event_type,event_subtype,count(*)::BIGINT events
        FROM peak_day GROUP BY ALL ORDER BY events DESC''').df()
    peak_types.to_csv(public / 'peak_day_event_types.csv', index=False)
    around = db.execute('''SELECT cast(event_time AS DATE) AS "day",count(*)::BIGINT events,
        count(DISTINCT client_id)::BIGINT customers FROM trx
        WHERE event_time>=TIMESTAMP '2022-12-10' AND event_time<TIMESTAMP '2022-12-13'
        GROUP BY "day" ORDER BY "day"''').df()
    around.to_csv(public / 'peak_day_neighbors.csv', index=False)
    counts_by_client = db.execute('''SELECT c.client_id,coalesce(t.n,0) trx,coalesce(g.n,0) geo,coalesce(d.n,0) dialog
        FROM cache.cache_clients c
        LEFT JOIN (SELECT client_id,sum(events) n FROM cache.cache_month_trx GROUP BY client_id) t USING(client_id)
        LEFT JOIN (SELECT client_id,sum(events) n FROM cache.cache_month_geo GROUP BY client_id) g USING(client_id)
        LEFT JOIN (SELECT client_id,sum(events) n FROM cache.cache_month_dialog GROUP BY client_id) d USING(client_id)''').df()
    references = []
    for source in ['trx', 'geo', 'dialog']:
        values = counts_by_client[source].to_numpy(dtype=float)
        q1, median, q3, p99 = np.quantile(values, [.25, .5, .75, .99])
        references.append(dict(source=source, customers=len(values), q1=q1, median=median, q3=q3,
                               p99=p99, above_p99=int((values>p99).sum()), max=int(values.max()),
                               log1p_count_defined_for_zero=True))
    pd.DataFrame(references).to_csv(public / 'customer_count_reference.csv', index=False)
    top20 = labels.head(20)
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), source_fingerprint=digest,
        rank_scope='All available history, UTC, descriptive EDA only; not a training cohort rule.',
        top20=dict(customers=20, trx_events=int(top20.trx_events.sum()),
                   share_all_trx_pct=float(top20.trx_events.sum()/38764930*100),
                   labelled_months=int(top20.labelled_months.sum()),
                   any_positive_months=int(top20.any_positive_months.sum()),
                   any_positive_rate_pct=float(top20.any_positive_months.sum()/top20.labelled_months.sum()*100),
                   clients_ever_positive=int(top20.any_positive_months.gt(0).sum())),
        maximum_amount=dict(amount=float(maximum.amount.max()), rows=len(maximum),
                            clients=int(maximum.client_id.nunique()), raw_p999_currency11_reference=q999,
                            owner_stats=owners.drop(columns='client_id').to_dict('records')),
        peak_day=dict(day='2022-12-11', weekday='Sunday', events=65302,
                      customers=len(peak_clients), max_events_one_client=int(peak_clients.events.max()),
                      top1_share_pct=float(peak_clients.events.head(1).sum()/65302*100),
                      top10_share_pct=float(peak_clients.events.head(10).sum()/65302*100),
                      top20_share_pct=float(peak_clients.events.head(20).sum()/65302*100),
                      median_events_per_client=float(peak_clients.events.median())),
        currency11=stats[stats.currency_group.eq('11')].iloc[0].to_dict(),
        raw_deleted=False, thresholds_fitted_for_model=False)
    (public / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    db.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/tail_review_20261109')
    parser.add_argument('--public-dir', type=Path, default=ROOT / 'docs/propensity_feasibility/tail_review')
    args = parser.parse_args()
    audit(ROOT, args.output_dir, args.public_dir)
