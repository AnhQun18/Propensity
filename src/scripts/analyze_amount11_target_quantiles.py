"""Past-only currency11 features vs labels; train-fixed quintiles and client bootstrap."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/propensity_processing_v3'
OUT = ROOT / 'outputs/amount11_target_quantiles_20261116'
LABELS = ['y_any'] + [f'target_{i}' for i in range(1, 5)]
METRICS = [f'trx_{w}d_main_amount_{s}' for w in (30, 84)
           for s in ('sum', 'mean', 'median', 'max')]


def main():
    OUT.mkdir(exist_ok=True)
    assert json.loads((SOURCE / 'manifest.json').read_text())['status'] == 'ready'
    cols = ['client_id', 'cutoff', 'supervised_eligible'] + LABELS + METRICS
    frames = []
    for file in sorted((SOURCE / 'features').rglob('*.parquet')):
        # Month partition names are verified against the stored cutoff below.
        if any(f'month=2022-{m:02d}' in str(file) for m in range(2, 12)):
            frames.append(pq.read_table(file, columns=cols).to_pandas())
    d = pd.concat(frames, ignore_index=True)
    d['month'] = pd.to_datetime(d.cutoff).dt.strftime('%Y-%m')
    assert set(d.month) == {f'2022-{m:02d}' for m in range(2, 12)}
    assert not d.duplicated(['client_id', 'cutoff']).any()
    assert d.supervised_eligible.all() and d[LABELS].isin([0, 1]).all().all()
    assert np.array_equal(d.y_any, d[LABELS[1:]].max(axis=1))
    d['split'] = np.where(d.month < '2022-09', 'train', 'validation')
    rows, monthly, thresholds, effects = [], [], {}, []
    rng = np.random.default_rng(20261116)
    for metric in METRICS:
        values = d[metric]
        assert (values.dropna() > 0).all()
        edges = values[d.split.eq('train')].dropna().quantile([.2,.4,.6,.8]).to_numpy()
        assert len(np.unique(edges)) == 4
        thresholds[metric] = edges.tolist()
        group = pd.Series('no_currency11_amount', index=d.index)
        valid = values.notna()
        group.loc[valid] = ('Q' + (np.searchsorted(edges, values[valid], side='left') + 1).astype(str))
        z = d[['client_id','month','split'] + LABELS].copy()
        z['group'] = group
        z['amount_value'] = values
        for (split, g), a in z.groupby(['split','group'], observed=True):
            item = dict(metric=metric,split=split,group=g,n=len(a),clients=a.client_id.nunique(),
                        amount_min=a.amount_value.min(),amount_median=a.amount_value.median(),amount_max=a.amount_value.max())
            for label in LABELS:
                item[f'{label}_positive'] = int(a[label].sum())
                item[f'{label}_rate_pct'] = 100*a[label].mean()
            rows.append(item)
        for (month, g), a in z.groupby(['month','group'], observed=True):
            monthly.append(dict(metric=metric,month=month,group=g,n=len(a),positive=int(a.y_any.sum()),rate_pct=100*a.y_any.mean()))
        v = z[z.split.eq('validation') & z.group.isin(['Q1','Q5'])]
        # Resample whole clients, not customer-month rows; same weight for all months.
        agg = v.groupby(['client_id','group']).y_any.agg(['sum','size']).unstack(fill_value=0)
        p1,n1,p5,n5 = [agg[(s,g)].to_numpy(dtype=float) for s,g in [('sum','Q1'),('size','Q1'),('sum','Q5'),('size','Q5')]]
        diffs = []
        for _ in range(500):
            weights = rng.poisson(1, size=len(agg))
            diffs.append(100*((weights@p5)/(weights@n5)-(weights@p1)/(weights@n1)))
        effect = dict(metric=metric,validation_q1_rate_pct=100*p1.sum()/n1.sum(),
                      validation_q5_rate_pct=100*p5.sum()/n5.sum(),
                      q5_minus_q1_pp=100*(p5.sum()/n5.sum()-p1.sum()/n1.sum()),
                      lift_q5_over_q1=(p5.sum()/n5.sum())/(p1.sum()/n1.sum()),
                      ci95_low_pp=float(np.quantile(diffs,.025)),ci95_high_pp=float(np.quantile(diffs,.975)))
        effects.append(effect)
        print(json.dumps(effect),flush=True)
    rates = pd.DataFrame(rows)
    rates.to_csv(OUT / 'rates_by_quintile.csv',index=False)
    pd.DataFrame(monthly).to_csv(OUT / 'rates_by_month.csv',index=False)
    pd.DataFrame(effects).to_csv(OUT / 'validation_effects.csv',index=False)
    (OUT / 'thresholds.json').write_text(json.dumps(thresholds,indent=2))
    d.groupby(['split','month']).y_any.agg(['size','sum','mean']).to_csv(OUT / 'cohort_summary.csv')
    fig, axes = plt.subplots(1,2,figsize=(11,4.5))
    for ax, suffix, title in zip(axes,('sum','median'),('Total amount, previous 30 days','Median transaction amount, previous 30 days')):
        metric = f'trx_30d_main_amount_{suffix}'
        for split in ('train','validation'):
            a = rates[(rates.metric==metric)&(rates.split==split)&rates.group.str.startswith('Q')].sort_values('group')
            ax.plot(a.group,a.y_any_rate_pct,marker='o',label=split)
        ax.set(title=title,xlabel='Quintile (train-fixed thresholds)',ylabel='Any-purchase rate (%)')
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'amount11_purchase_rates.png',dpi=170)
    plt.close(fig)
    (OUT / 'methodology.json').write_text(json.dumps(dict(status='ready',rows=len(d),
        source=str(SOURCE),unit='customer-month, not transaction',currency=11,
        labels=LABELS,windows_days=[30,84],train='2022-02 through 2022-08',validation='2022-09 through 2022-11',
        reserved_months_excluded=['2022-12','2023-01'],
        grouping='quintiles from nonmissing train customer-month feature values; ties kept together; frozen in validation; missing separate',
        uncertainty='500 Poisson cluster-bootstrap draws by client; percentile 95% CI of pooled validation Q5-Q1 rate difference; unadjusted exploratory comparisons',
        limitations='Association, not causation. Multiple exploratory metrics; CI not multiplicity adjusted. No control for activity, prior purchase, source availability or other confounding. Monthly tables show stability.',
        raw_modified=False),indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
