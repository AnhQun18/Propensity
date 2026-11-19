"""Full-source descriptive geo cell occupancy; no prediction cutoff is applied."""
from pathlib import Path
import math
import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path('outputs/geo_cell_sizes')
OUT.mkdir(parents=True, exist_ok=True)
con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='4GB'")
con.execute("CREATE VIEW geo AS SELECT * FROM read_parquet('data/MBD-mini/detail/geo/**/*.parquet')")
scope = con.execute('SELECT count(*), count(distinct client_id), min(event_time), max(event_time) FROM geo').fetchone()
print('scope:', scope, flush=True)

def gini(values):
    x = np.sort(np.asarray(values, dtype=float))
    return float(2 * np.dot(np.arange(1, len(x)+1), x) / (len(x)*x.sum()) - (len(x)+1)/len(x))

rows = []
fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=180)
for i, level in enumerate([4, 5, 6]):
    df = con.execute(f'SELECT geohash_{level} cell, count(*) events, count(distinct client_id) customers FROM geo WHERE geohash_{level} IS NOT NULL GROUP BY 1').df()
    df.to_csv(OUT/f'g{level}_cell_counts.csv', index=False)
    df.nlargest(20, 'customers').to_csv(OUT/f'g{level}_top20_customers.csv', index=False)
    for j, metric in enumerate(['events', 'customers']):
        x = df[metric].to_numpy()
        n_top = max(1, math.ceil(len(x)*.01))
        rows.append(dict(level=f'g{level}', metric=metric, cells=len(x), total=int(x.sum()),
            median=float(np.median(x)), p90=float(np.quantile(x,.9)), p99=float(np.quantile(x,.99)),
            maximum=int(x.max()), gini=gini(x), singleton_pct=float((x==1).mean()*100),
            le5_pct=float((x<=5).mean()*100), top1pct_cells=n_top,
            top1pct_share=float(np.sort(x)[-n_top:].sum()/x.sum()*100)))
        # Empirical complementary CDF: retains singletons and avoids bin-width artifacts.
        sizes, freq = np.unique(x, return_counts=True)
        survival = np.cumsum(freq[::-1])[::-1]/len(x)
        ax = axes[j,i]
        ax.loglog(sizes, survival, color=['#2864a0','#208575'][j], lw=1.8)
        ax.set_title(f'g{level} · {len(df):,} observed cells')
        ax.set_xlabel('Events per cell' if metric=='events' else 'Distinct customers per cell')
        ax.set_ylabel('Fraction of cells with size >= x')
        ax.grid(True, which='both', alpha=.15)
        ax.text(.05,.07,f'Gini = {gini(x):.3f}',transform=ax.transAxes)
    print(f'g{level} complete', flush=True)
summary = pd.DataFrame(rows)
summary.to_csv(OUT/'summary.csv', index=False)
fig.suptitle('Geo cell occupancy: empirical CCDF on log-log axes', fontsize=18)
fig.tight_layout(rect=(0,.03,1,.95))
fig.text(.02,.015,'All detail/geo events across 5 folds. Customers are distinct within each cell; a customer may visit multiple cells.', fontsize=10)
fig.savefig(OUT/'cell_size_loglog.png')
plt.close(fig)
(OUT/'scope.txt').write_text(f'Events: {scope[0]}\nDistinct customers: {scope[1]}\nEvent time: {scope[2]} through {scope[3]}\nObserved non-null cells only. No event deduplication. Full-source descriptive EDA, not cutoff-specific model features.\nCustomer sums represent customer-cell memberships, not unique people. Top 1% uses ceil(0.01 * observed cell count), ranked separately per metric.\n', encoding='utf-8')
print(summary.to_string(index=False))
