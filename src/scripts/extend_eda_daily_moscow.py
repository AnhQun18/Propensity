"""Daily activity, exact elapsed-time reactivation and cross-source synchrony."""
from pathlib import Path
import json
import calendar
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from complete_eda_figures_moscow import ROOT, OUT, BASE, publish, INDEX, event_studies


def extract():
    c = duckdb.connect()
    c.execute("SET memory_limit='3GB'")
    c.execute('SET threads=2')
    c.execute(f"SET temp_directory='{OUT.as_posix()}/tmp_daily'")
    tables = {}
    for source in ['trx', 'geo', 'dialog']:
        dest = OUT / f'{source}_daily.parquet'
        if not dest.exists():
            print('Raw daily:', source, flush=True)
            path = (ROOT / f'data/MBD-mini/detail/{source}/**/*.parquet').as_posix()
            c.execute(f"""COPY (
                WITH daily AS (SELECT client_id, cast(event_time + INTERVAL '3 hours' AS DATE) AS "day",
                  count(*) events, min(event_time) first_utc, max(event_time) last_utc
                  FROM read_parquet('{path}') GROUP BY 1,2),
                gaps AS (SELECT *,lag(last_utc) OVER(PARTITION BY client_id ORDER BY "day") prev_utc FROM daily)
                SELECT *, epoch(first_utc-prev_utc)/86400.0 gap_days FROM gaps
                ) TO '{dest.as_posix()}' (FORMAT PARQUET)""")
        tables[source] = c.execute(f"SELECT * FROM read_parquet('{dest.as_posix()}')").df()
    c.close()
    return tables


def main():
    tables = extract()
    labels = pd.read_parquet(BASE / 'labels_any.parquet')
    ids = np.sort(labels.client_id.unique())
    panel = pd.read_parquet(BASE / 'all_customer_months.parquet')
    months = pd.date_range(panel.month.min(), panel.month.max(), freq='MS')
    y = labels.pivot(index='client_id', columns='month', values='y').reindex(index=ids, columns=months).to_numpy(dtype=float, copy=True)
    inventory = pd.read_csv(BASE / 'source_inventory.csv').set_index('source')
    arrays = {}
    quality = []
    states = []
    daily_sets = {}
    for source, d in tables.items():
        d['day'] = pd.to_datetime(d.day)
        d['month'] = d.day.dt.to_period('M').dt.to_timestamp()
        expected = int(inventory.loc[source, 'raw_rows'])
        assert int(d.events.sum()) == expected
        assert set(d.client_id).issubset(set(ids))
        assert not d.duplicated(['client_id', 'day']).any()
        first = pd.Timestamp(inventory.loc[source, 'first_month'])
        last = pd.Timestamp(inventory.loc[source, 'last_month'])
        partial = bool(inventory.loc[source, 'first_month_partial'])
        full = (months >= first) & (months <= last)
        if partial: full &= months != first
        days = d.groupby(['client_id', 'month']).size().unstack().reindex(index=ids, columns=months).fillna(0).to_numpy(dtype=float, copy=True)
        days[:, ~full] = np.nan
        arrays[f'{source}_active_days'] = days
        quality.append(dict(source=source, raw_events=expected, daily_events=int(d.events.sum()), customers=d.client_id.nunique()))
        for threshold in [30, 60, 90]:
            r = d[d.gap_days >= threshold].groupby(['client_id', 'month']).size().unstack().reindex(index=ids, columns=months).fillna(0).to_numpy(dtype=float, copy=True)
            r[:, ~full] = np.nan
            arrays[f'{source}_reactivation_{threshold}d'] = r
        # Separate first-observed from subsequent activity, with complete population reconciliation.
        first_month = d.groupby('client_id').month.min().reindex(ids)
        for j, m in enumerate(months):
            if not full[j]:
                states.append(dict(source=source, month=str(m.date()), state='Chưa đủ tháng nguồn', customers=len(ids)))
                continue
            active = days[:, j] > 0
            initial = (first_month == m).to_numpy()
            reactive = arrays[f'{source}_reactivation_30d'][:, j] > 0
            masks = {'Lần đầu thấy': active & initial, 'Tái xuất hiện ≥30d': active & ~initial & reactive,
                     'Có log khác': active & ~initial & ~reactive, 'Không log tháng này': ~active}
            assert np.vstack(list(masks.values())).sum(axis=0).min() == 1
            for name, mask in masks.items(): states.append(dict(source=source, month=str(m.date()), state=name, customers=int(mask.sum())))
        daily_sets[source] = d[['client_id', 'day', 'month']]
    # Event studies reuse the same exhaustive 12-anchor implementation.
    event_studies(ids, months, arrays, y)
    state = pd.DataFrame(states)
    assert state.groupby(['source', 'month']).customers.sum().eq(len(ids)).all()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    for ax, source in zip(axes, tables):
        z = state[state.source == source].pivot(index='month', columns='state', values='customers').fillna(0)
        (z/len(ids)*100).plot.bar(stacked=True, ax=ax, width=.9)
        ax.set_title(source); ax.set_ylabel('% toàn bộ khách'); ax.legend(fontsize=7, ncol=3)
    publish('reactivation_population_states', fig, state, 'Tái xuất hiện sau im lặng và lần đầu được quan sát',
        'Gap dùng thời điểm UTC thực giữa sự kiện đầu ngày và sự kiện cuối ngày trước đó của cùng khách; '
        'ngưỡng chính ≥30 ngày. Một khách mới được thấy trong tháng được ưu tiên nhóm lần đầu, sau đó tái xuất hiện, log khác, không log. '
        'Tháng đầu nguồn không trọn và ngoài nguồn giữ riêng. Các ngưỡng 60/90 có bộ hình event riêng.',
        'Tách lần đầu được quan sát khỏi hoạt động quay lại sau khoảng nghỉ đủ dài.',
        'Mỗi nguồn–tháng phân nhóm đủ 100.224 khách; các bảng event giữ đủ 12 tháng nhãn và cả hai phía thời gian.',
        'Khoảng im lặng là không thấy log; không chứng minh rời ngân hàng hoặc tracking liên tục. Ngày nguồn có thể đã được ẩn danh.')
    # Exact rotation-mean null: E(overlap) = |A| |B| / days-in-month.
    # This is identical to averaging every cyclic shift, eliminating Monte Carlo noise.
    pair_rows = []
    for sa, sb in [('trx', 'geo'), ('trx', 'dialog'), ('geo', 'dialog')]:
        a, b = daily_sets[sa], daily_sets[sb]
        na = a.groupby(['client_id', 'month']).size().rename('days_a')
        nb = b.groupby(['client_id', 'month']).size().rename('days_b')
        overlap = a.merge(b, on=['client_id', 'day', 'month']).groupby(['client_id', 'month']).size().rename('overlap')
        z = pd.concat([na, nb, overlap], axis=1).fillna(0).reset_index()
        z['expected'] = z.days_a*z.days_b/z.month.dt.days_in_month
        z['excess'] = z.overlap-z.expected
        z['union'] = z.days_a+z.days_b-z.overlap
        for m in months:
            valid = True
            for s in [sa, sb]:
                first = pd.Timestamp(inventory.loc[s, 'first_month']); last = pd.Timestamp(inventory.loc[s, 'last_month'])
                valid &= first <= m <= last and not (bool(inventory.loc[s, 'first_month_partial']) and first == m)
            q = z[z.month == m]
            pair_rows.append(dict(pair=f'{sa} × {sb}', month=str(m.date()), observed_customers=len(ids) if valid else 0,
                unavailable=0 if valid else len(ids), both_active=int(((q.days_a>0)&(q.days_b>0)).sum()) if valid else 0,
                overlap_days=float(q.overlap.sum()/len(ids)) if valid else np.nan,
                expected_days=float(q.expected.sum()/len(ids)) if valid else np.nan,
                excess_days=float(q.excess.sum()/len(ids)) if valid else np.nan,
                jaccard_pooled=float(q.overlap.sum()/q.union.sum()) if valid and q.union.sum() else np.nan))
    sync = pd.DataFrame(pair_rows)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for ax, (pair, q) in zip(axes, sync.groupby('pair')):
        ax.plot(pd.to_datetime(q.month), q.overlap_days, 'o-', label='Đồng hoạt động quan sát')
        ax.plot(pd.to_datetime(q.month), q.expected_days, '--', label='Kỳ vọng dịch vòng chính xác')
        ax.set_title(pair); ax.set_ylabel('Ngày / toàn bộ khách'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    publish('cross_source_daily_synchrony', fig, sync, 'Đồng hoạt động cùng ngày giữa các nguồn',
        'Ngày Moscow; đếm giao ngày từng cặp nguồn trong từng khách–tháng, gồm khách zero. '
        'Kỳ vọng trung bình của TOÀN BỘ phép dịch vòng ngày trong tháng là days_a × days_b / số ngày tháng; '
        'dùng công thức chính xác thay 199 lượt ngẫu nhiên. Đây là kỳ vọng đối chứng, chưa phải kiểm định hay CI. '
        'Tháng không đủ một trong hai nguồn giữ unavailable.',
        'So số ngày đồng hoạt động với mức kỳ vọng từ lượng ngày hoạt động của chính từng khách.',
        'Khoảng cách giữa đường quan sát và kỳ vọng biểu thị đồng bộ vượt mức kỳ vọng dịch vòng. '
        'CSV có excess, Jaccard gộp và n cả hai nguồn; không suy nhân quả hoặc thứ tự trong ngày.',
        'Đối chứng dịch vòng không bảo toàn lịch thứ hoặc xu hướng trong tháng. Nhiễu ngày liên bảng chưa xác minh. '
        'Chưa phân tầng TRX, chưa có CI và chưa tính đồng thời cả ba nguồn.')
    pd.DataFrame(quality).to_csv(OUT / 'daily_validation.csv', index=False)
    old = pd.read_csv(OUT / 'figure_index.csv')
    pd.concat([old, pd.DataFrame(INDEX)]).drop_duplicates('figure', keep='last').to_csv(OUT/'figure_index.csv', index=False)
    status = json.loads((OUT/'execution_status.json').read_text(encoding='utf-8'))
    status['done'] += ['raw_daily_activity', 'active_days_event_studies', 'reactivation_30_60_90', 'pairwise_exact_rotation_mean_synchrony']
    status['pending'] = [s for s in status['pending'] if s not in ['daily_reactivation_30_60_90']]
    status['synchrony_method_change'] = 'Exact expectation across all cyclic shifts replaces 199 Monte Carlo shifts; no p-value claimed'
    status['figures'] = len(pd.read_csv(OUT/'figure_index.csv'))
    (OUT/'execution_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
