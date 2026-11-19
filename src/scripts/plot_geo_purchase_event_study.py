"""Plot full-population monthly GEO and DIALOG activity around label months."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/full_population_temporal_eda_20261118'
def plot(source):
    d = pd.read_csv(BASE / 'before_after_monthly_rates.csv', parse_dates=['target_month'])
    d = d[d.source.eq(source)].copy()
    assert d.target_month.nunique() >= 12
    assert (d.observed_customers + d.unavailable_customers == d.total_customers).all()
    assert d.groupby(['target_month', 'lag']).total_customers.sum().eq(100224).all()
    for y, title, name in [(1, 'Nhóm có mua trong tháng mốc', 'geo_event_study_buyers'),
                           (0, 'Nhóm không mua trong tháng mốc', 'geo_event_study_nonbuyers')]:
        fig, ax = plt.subplots(figsize=(14, 4.8))
        q = d[d.purchase.eq(y)]
        for lag, color, label in [(-1, '#2875b9', 'Tháng trước mốc mua'),
                                  (1, '#db6b28', 'Tháng sau mốc mua')]:
            z = q[q.lag.eq(lag)].sort_values('target_month')
            ax.plot(z.target_month, z.rate, 'o-', color=color, label=label, linewidth=2)
            for row in z.itertuples():
                if row.unavailable_customers:
                    ax.annotate('chưa quan sát', (row.target_month, .02), rotation=90,
                                ha='center', va='bottom', fontsize=7, color=color)
        ax.set_title(title, loc='left', weight='bold')
        ax.set_ylabel(f'Tỷ lệ khách có ≥1 sự kiện {source.upper()}')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.2)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                  borderaxespad=0, fontsize=9, frameon=False)
        paired = q.pivot(index='target_month', columns='lag', values='rate').dropna(subset=[-1, 1])
        summary = []
        for lag, label in [(-1, 'Tháng trước'), (1, 'Tháng sau')]:
            rates = paired[lag]
            summary.append(f'{label} (n={len(rates)} tháng)\n'
                           f'  Trung bình: {rates.mean():.1%}\n'
                           f'  Trung vị:    {rates.median():.1%}')
        ax.text(1.02, .68, 'Cùng tháng mốc quan sát đủ hai phía\n\n' + '\n\n'.join(summary), transform=ax.transAxes,
                ha='left', va='top', fontsize=10, linespacing=1.5,
                bbox=dict(facecolor='#f4f7fb', edgecolor='#dbe4ee', boxstyle='round,pad=.6'))
        ax.set_xticks(sorted(d.target_month.unique()))
        ax.set_xticklabels([x.strftime('%m/%Y') for x in sorted(d.target_month.unique())], rotation=40)
        ax.set_xlabel('Tháng nhãn làm mốc')
        fig.suptitle(f'{source.upper()} quanh tháng mua | nhãn 02/2022–01/2023', weight='bold')
        fig.text(.01, .01, f'Mẫu số: khách thuộc nhóm có tháng {source.upper()} quan sát được; không có sự kiện vẫn được tính. '
                 f'{source.upper()} giả định UTC → Moscow. Mua lặp được giữ; không suy nhân quả.', fontsize=8)
        fig.subplots_adjust(left=.08, right=.72, bottom=.22, top=.82)
        path = BASE / (name.replace('geo_', f'{source}_') + '.png')
        fig.savefig(path, dpi=170, bbox_inches='tight')
        fig.savefig(path.with_suffix('.svg'), bbox_inches='tight')
        plt.close(fig)
        print(path.name)

if __name__ == '__main__':
    for source in ('geo', 'dialog'):
        plot(source)
