"""Write an aggregate-only Markdown comparison from a completed validation run."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from propensity_baseline.config import load_baseline_config
from propensity_baseline.data import read_json


def percent(value):
    return '—' if value is None else f'{100*value:.2f}%'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    config=load_baseline_config(args.config)
    out=config.output_dir
    comparison=out/'rule_validation'
    if read_json(comparison/'manifest.json')['status']!='ready':
        raise ValueError('Rule comparison is not ready')
    state=read_json(out/'manifest.json')
    selection=read_json(out/'selection.json')
    records=read_json(comparison/'monthly_comparison.json')
    intervals=read_json(comparison/'paired_comparison.json')
    table=pd.read_csv(comparison/'monthly_comparison.csv')
    names={'multi_recent':'CatBoost trx + geo + dialog: tháng gần nhất',
           'multi_history':'CatBoost trx + geo + dialog: gần nhất + tích lũy cũ'}
    winner=selection['variant']
    reference=table.loc[(table.model==winner)&(table.method=='rule')]
    macro=table.loc[table.method=='model'].groupby('model',sort=True)[
        ['precision_at_1pct','recall_at_1pct','lift_at_1pct','average_precision']].mean()
    lines=['# Baseline đa nguồn so với rule mua tháng trước','',
           f"Ngày báo cáo (UTC): {datetime.now(timezone.utc).date().isoformat()}.",
           '', '**Kết quả validation cuốn chiếu 09–11/2022**, mỗi tháng chỉ train bằng nhãn các tháng trước.',
           'Rule: `dự đoán tháng M = y_any tháng M−1`. So trên cùng khách có nhãn hiện tại và nhãn trước sẵn có.',
           '', f"Mô hình được chọn trong hai baseline: **{winner}**. Test 01/2023 " + ('đã mở.' if state['test_opened'] else '**chưa mở**.'),
           '', '## So sánh cùng top 1%','',
           '| Phương pháp | Precision@1% | Recall@1% | Lift@1% | AP |',
           '| --- | ---: | ---: | ---: | ---: |']
    for variant,row in macro.iterrows():
        lines.append(f"| {names.get(variant,variant)} | {percent(row.precision_at_1pct)} | {percent(row.recall_at_1pct)} | {row.lift_at_1pct:.2f} | {row.average_precision:.4f} |")
    lines.append(f"| Rule mua tháng trước | {percent(reference.precision_at_1pct.mean())} | {percent(reference.recall_at_1pct.mean())} | {reference.lift_at_1pct.mean():.2f} | {reference.average_precision.mean():.4f} |")
    lines += ['', 'Các giá trị là trung bình đều theo tháng. Top 1% dùng K=ceil(0,01×N); ties của rule được phá bằng seed/ID cố định.',
              '', '## Theo tháng: baseline được chọn và rule','',
              '| Tháng | Khách so sánh | Ca mua thực | ML bắt được @1% | Rule bắt được @1% | Recall ML | Recall rule |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    selected=[r for r in records if r['model']==winner and r['status']=='compared']
    for r in selected:
        ml=r['model_metrics']['top_k']['0.01']; rule=r['rule_ranking_metrics']['top_k']['0.01']
        lines.append(f"| {r['month']} | {r['comparison_rows']:,} | {r['model_metrics']['positives']:,} | {ml['hits']} | {rule['hits']} | {percent(ml['recall'])} | {percent(rule['recall'])} |")
    lines += ['', '## Ngân sách tự nhiên của rule','',
              'Rule chọn toàn bộ khách mua tháng trước. ML được chọn đúng cùng số khách, không dùng threshold 0,5 để so lệch ngân sách.',
              '', '| Tháng | Số khách được chọn | ML bắt được | Rule bắt được | Recall ML | Recall rule |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in selected:
        ml=r['model_at_rule_budget']; rule=r['rule_natural_decisions']
        lines.append(f"| {r['month']} | {rule['selected']:,} | {ml['tp']} | {rule['tp']} | {percent(ml['recall'])} | {percent(rule['recall'])} |")
    lines += ['', '## Độ bất định và phạm vi','']
    for interval in intervals:
        limits=interval.get('interval_95')
        if limits:
            lines.append(f"- {interval['variant']} trừ rule, chênh Recall@1%: **{100*interval['estimate']:.2f} điểm phần trăm**, khoảng bootstrap 95% [{100*limits[0]:.2f}; {100*limits[1]:.2f}], {interval['replicates']} lần lấy mẫu theo khách.")
    lines += ['', '- Đây là validation dùng để chọn giữa hai baseline, chưa phải test độc lập. Bootstrap có điều kiện theo mô hình và ba tháng đã quan sát.',
              f"- Tổng số dòng có nhãn bị loại vì thiếu nhãn trước ở baseline được chọn: {sum(r['excluded_unavailable_prior'] for r in selected)}.",
              '- Baseline dùng geo và dialog metadata (count/recency/độ phủ), chưa dùng nội dung dialog embedding.',
              '- Rule là quyết định 0/1, không báo log loss như xác suất đã hiệu chỉnh.',
              '- Nhãn mua tháng trước chỉ dùng cho rule đối chứng, không được đưa vào X của CatBoost.',
              '', '## Tái lập','',
              f"- Config: `configs/propensity_baseline.json`; seed {config.seed}, depth {selection['parameters']['depth']}, {selection['parameters']['iterations']} cây, learning rate {config.learning_rate}.",
              f"- Run: `{out.name}`; nguồn feature: `{config.source_dir.name}`.",
              '- File chi tiết: `rule_validation/monthly_comparison.json`, `monthly_comparison.csv`, `paired_comparison.json` và dự đoán trong output cục bộ.',
              '- Notebook điều phối: `notebooks/05_calendar_baseline.ipynb`.', '']
    path=args.output.resolve()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    print(path)


if __name__=='__main__':
    main()
