"""Plot audited any-product label prevalence; no raw or pipeline execution."""
from pathlib import Path
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/propensity_feasibility/any_product'
INK, GREEN, MUTED, CORAL = '#173d38', '#267969', '#647c72', '#c27d5d'


def vi(value, decimals=0):
    return f'{value:,.{decimals}f}'.replace(',', '~').replace('.', ',').replace('~', '.')


def main():
    monthly = pd.read_csv(OUT / 'monthly.csv').sort_values('month').reset_index(drop=True)
    summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    expected = pd.period_range('2022-02', '2023-01', freq='M').astype(str).tolist()
    assert monthly.month.tolist() == expected
    assert monthly[['rows', 'clients', 'labelled']].eq(100224).all().all()
    assert monthly.unknown.eq(0).all()
    assert monthly.positives.between(0, monthly.labelled).all()
    assert monthly.negative.add(monthly.positives).eq(monthly.labelled).all()
    assert monthly.labelled.sum() == summary['rows'] == 1202688
    assert monthly.positives.sum() == summary['positive_rows'] == 12854
    assert np.allclose(monthly.positive_rate, monthly.positives / monthly.labelled)
    rate = monthly.positive_rate.to_numpy() * 100
    overall = monthly.positives.sum() / monthly.labelled.sum() * 100
    assert np.isclose(overall / 100, summary['positive_rate'])
    x = np.arange(len(monthly))
    labels = pd.to_datetime(monthly.month).dt.strftime('%m/%Y')

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 13,
        'text.color': INK, 'axes.labelcolor': MUTED,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'svg.fonttype': 'none',
    })
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    fig.text(.065, .942, 'MBD-MINI / NHÃN MUA BẤT KỲ SẢN PHẨM NÀO',
             fontsize=11, color=MUTED)
    fig.text(.065, .879, 'Biến động tỷ lệ nhãn dương · target_any',
             fontsize=27, fontweight='bold')
    fig.text(.065, .829, 'Ít nhất một trong bốn sản phẩm có nhãn · 100.224 khách/tháng',
             fontsize=15, color=MUTED)
    fig.text(.065, .771,
             f'{vi(summary["positive_rows"])} mẫu dương / {vi(summary["rows"])} khách–tháng'
             f'   ·   Tỷ lệ toàn kỳ: {vi(overall, 4)}%',
             fontsize=15, color=GREEN, fontweight='bold')

    ax = fig.add_axes([.085, .235, .86, .46])
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#e5ece7', linewidth=.9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#d6e2da')
    ax.tick_params(length=0, pad=10)
    ax.axhline(overall, color=CORAL, linestyle=(0, (5, 4)), linewidth=1.6,
               label=f'Toàn kỳ {vi(overall, 4)}%')
    ax.plot(x, rate, color=GREEN, linewidth=2.8, marker='o', markersize=7,
            markeredgecolor='white', markeredgewidth=1.1, zorder=3)
    for pos, value, count in zip(x, rate, monthly.positives):
        ax.annotate(vi(count), (pos, value), xytext=(0, 13),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=12, fontweight='bold', color=INK)
    ax.set_xlim(-.4, 11.4)
    ax.set_ylim(.55, 1.9)
    ax.set_xticks(x, labels, rotation=0, fontsize=11)
    ax.yaxis.set_major_locator(MultipleLocator(.2))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: vi(value, 1) + '%'))
    ax.set_ylabel('Tỷ lệ khách–tháng có target_any = 1', fontsize=12, labelpad=14)
    ax.set_xlabel('Tháng nhãn', fontsize=12, labelpad=15)
    ax.legend(loc='upper right', frameon=False, fontsize=12)

    fig.text(.065, .112,
             'Số cạnh điểm = số mẫu dương trong tháng. Trục Y tuyến tính, phóng từ 0,55% đến 1,90%.',
             fontsize=12, color=MUTED)
    fig.text(.065, .071,
             'target_any = y_any = OR(target_1, target_2, target_3, target_4); mỗi khách–tháng chỉ đếm một lần.',
             fontsize=11, color=MUTED)
    fig.text(.065, .034, 'Nguồn: docs/propensity_feasibility/any_product/monthly.csv · Gộp cả 5 fold',
             fontsize=10, color=MUTED)
    for suffix in ['png', 'svg']:
        path = OUT / f'target_any_by_month.{suffix}'
        fig.savefig(path, dpi=180, facecolor='white')
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
