"""Publish aggregate-only Markdown/HTML evidence from a completed ablation."""
import argparse
import base64
import html
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from propensity_baseline.data import read_json
from propensity_baseline.runner import sha256_file


NAMES = {
    'previous_month_purchase': 'Rule mua tháng trước',
    'behavior_original': 'Hành vi gốc (153 biến)',
    'behavior': 'Hành vi bỏ biến lịch (141 biến)',
    'purchase': 'Lịch sử mua (15 biến)',
    'combined': 'Hành vi + lịch sử mua (156 biến)',
}
ORDER = list(NAMES)


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |',
                      *['| '+' | '.join(map(str, row))+' |' for row in rows]])


def markdown_html(text, root):
    """Small renderer for our own headings, paragraphs, lists, tables and images."""
    def inline(value):
        value = html.escape(value)
        value = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', value)
        value = re.sub(r'`(.+?)`', r'<code>\1</code>', value)
        return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', value)
    lines, output, i = text.splitlines(), [], 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith('|'):
            group = []
            while i < len(lines) and lines[i].startswith('|'):
                group.append(lines[i]); i += 1
            cells = lambda row: [x.strip() for x in row.strip('|').split('|')]
            output.append('<table><thead><tr>'+''.join('<th>'+inline(x)+'</th>' for x in cells(group[0]))+'</tr></thead><tbody>')
            for row in group[2:]:
                output.append('<tr>'+''.join('<td>'+inline(x)+'</td>' for x in cells(row))+'</tr>')
            output.append('</tbody></table>')
            continue
        match = re.match(r'!\[([^]]*)\]\(([^)]+)\)', line)
        if match:
            image = root/match[2]
            data = base64.b64encode(image.read_bytes()).decode()
            output.append(f'<figure><img alt="{html.escape(match[1])}" src="data:image/png;base64,{data}"></figure>')
        elif line.startswith('#'):
            depth = len(line)-len(line.lstrip('#'))
            output.append(f'<h{depth}>{inline(line[depth:].strip())}</h{depth}>')
        elif line.startswith('- '):
            output.append('<p class="bullet">'+inline(line[2:])+'</p>')
        else:
            output.append('<p>'+inline(line)+'</p>')
        i += 1
    style = '''
    :root{--navy:#142c48;--teal:#087f8c;--muted:#5c6d7d;--line:#dce5eb}
    *{box-sizing:border-box}body{margin:0;background:#edf2f5;color:var(--navy);font:17px/1.65 "Segoe UI",Arial,sans-serif}
    main{max-width:1100px;margin:40px auto;background:white;padding:56px 65px;border-top:9px solid var(--teal)}
    h1{font-size:42px;line-height:1.16;letter-spacing:-1px;margin:0 0 20px}h2{font-size:26px;line-height:1.3;margin:42px 0 16px;border-top:1px solid var(--line);padding-top:24px}
    p{margin:13px 0}.bullet{padding-left:20px;border-left:3px solid var(--teal)}a{color:var(--teal)}code{font-size:.86em;background:#eef5f5;padding:2px 5px}
    table{width:100%;border-collapse:collapse;font-size:14px;margin:22px 0}th{text-align:left;background:var(--navy);color:white;font-weight:600}th,td{padding:12px 10px;border-bottom:1px solid var(--line)}tr:nth-child(even){background:#f4f8fa}
    img{display:block;width:100%;height:auto}figure{margin:24px 0}strong{color:#075e68}footer{font-size:12px;color:var(--muted);border-top:1px solid var(--line);margin-top:40px;padding-top:20px}
    @media print{body{background:white;font-size:11pt}main{margin:0;padding:0;max-width:none;border-top:0}h1{font-size:26pt}h2{font-size:17pt;break-after:avoid}table,figure{break-inside:avoid}a{color:inherit} @page{size:A4;margin:18mm}}
    '''
    return '<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Baseline propensity · 11/11/2026</title><style>'+style+'</style><main>'+''.join(output)+'<footer>MBD-mini · Baseline offline · Tổng hợp từ artifact validation, 11/11/2026</footer></main></html>'


def publish(run, out):
    state = read_json(run/'manifest.json')
    if state['status'] != 'complete' or state['test_opened'] or state['calibration_opened']:
        raise ValueError('Requires a completed, validation-only run')
    for name, expected in state['artifact_sha256'].items():
        if sha256_file(run/name) != expected:
            raise ValueError(f'Run artifact changed: {name}')
    out.mkdir(parents=True, exist_ok=False)
    summary, monthly, comparisons = [read_json(run/name) for name in
                                    ('summary.json','monthly_metrics.json','paired_comparisons.json')]
    by_name = {r['model']:r for r in summary}
    # Confirm exact ranking metrics for the rerun of the historical baseline.
    prior_run = run.parent/'propensity_baseline_geo_dialog_v1'
    old = read_json(prior_run/'validation_metrics.json')
    parity = []
    for current in monthly:
        if current['model'] != 'behavior_original':
            continue
        previous = next(r for r in old if r['model'] == 'multi_history' and r['month'] == current['month'])
        for key in ('average_precision','roc_auc','log_loss','brier'):
            if not np.isclose(current[key], previous[key], rtol=1e-9, atol=1e-12):
                raise ValueError(f'Original baseline rerun differs: {current["month"]} {key}')
        if current['top_k'] != previous['top_k']:
            raise ValueError('Original baseline top-K metrics differ')
        parity.append(dict(month=current['month'], status='matched'))
    if len(parity) != 3:
        raise ValueError('Missing original-model validation rerun')
    pd.DataFrame(summary).to_csv(out/'summary.csv', index=False)
    budget_rows = []
    for row in monthly:
        for budget, values in row['top_k'].items():
            budget_rows.append(dict(model=row['model'], month=row['month'], rows=row['rows'],
                                    positives=row['positives'], budget=float(budget), **values))
    budgets = pd.DataFrame(budget_rows)
    budgets.to_csv(out/'monthly_budget_metrics.csv', index=False)
    write = lambda name, value: (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    write('paired_comparisons.json', comparisons)
    write('monthly_metrics.json', monthly)
    write('feature_contract.json', dict(variants=state['feature_variants'], parameters=state['parameters'],
                                      protocol=state['protocol'], seed=state['config']['seed'],
                                      purchase_rule='Exact calendar lags, historical labels available at each row cutoff only',
                                      caveat='Known history length is an availability feature and can also indicate time. Combined/purchase share the same history features.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.spines.top':False,
                         'axes.spines.right':False, 'axes.spines.left':False})
    colors = ['#a0adb9','#8093a8','#1f526d','#54a8ad','#087f8c']
    fig, ax = plt.subplots(figsize=(11.5,4.5), constrained_layout=True)
    values = [100*by_name[name]['mean_recall_at_1pct'] for name in ORDER]
    ax.barh(range(len(ORDER)), values, color=colors, height=.58)
    ax.set_yticks(range(len(ORDER)), [NAMES[n] for n in ORDER]); ax.invert_yaxis()
    ax.set_xlabel('Recall@1% trung bình ba tháng (%)'); ax.set_xlim(0,max(values)*1.18)
    ax.set_axisbelow(True); ax.grid(axis='x', alpha=.16)
    for i,value in enumerate(values):
        ax.text(value+.12, i, f'{value:.2f}%', va='center', weight='bold', color='#142c48')
    fig.savefig(out/'recall_comparison.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10.5,4.3), constrained_layout=True)
    months = sorted({r['month'] for r in monthly})
    for name,color in zip(ORDER,colors):
        vals = [next(r['top_k']['0.01']['recall']*100 for r in monthly if r['model']==name and r['month']==m) for m in months]
        ax.plot(months,vals,'o-',label=NAMES[name],color=color,linewidth=2)
    ax.set_ylabel('Recall@1% (%)'); ax.grid(axis='y',alpha=.16); ax.legend(fontsize=9,loc='upper left',bbox_to_anchor=(1,1))
    fig.savefig(out/'monthly_recall.png',dpi=180); plt.close(fig)
    pct = lambda value: f'{100*value:.2f}%'.replace('.',',')
    pp = lambda value: f'{100*value:+.2f}'.replace('.',',')
    best = max(summary,key=lambda r:r['mean_recall_at_1pct'])
    combined_delta = next(c for c in comparisons if c['variant']=='combined' and c['reference']=='purchase')
    ci = combined_delta['interval_95']
    if ci and ci[0] > 0:
        verdict = 'Khoảng bootstrap nằm trên 0: kết hợp cải thiện Recall@1% trong các tháng đã quan sát.'
    elif ci and ci[1] < 0:
        verdict = 'Khoảng bootstrap nằm dưới 0: kết hợp có Recall@1% thấp hơn mô hình lịch sử mua trong các tháng đã quan sát.'
    else:
        verdict = 'Khoảng bootstrap còn chứa 0: chưa có bằng chứng rõ rằng kết hợp tốt hơn mô hình lịch sử mua trên Recall@1%.'
    summary_table = table(['Phương pháp','Biến','Recall@1%','AP','ROC-AUC','Logloss'],
                          [[NAMES[n],by_name[n]['predictors'],pct(by_name[n]['mean_recall_at_1pct']),
                            f"{by_name[n]['mean_ap']:.4f}",f"{by_name[n]['mean_roc_auc']:.4f}",
                            f"{by_name[n]['mean_log_loss']:.5f}" if by_name[n]['mean_log_loss'] is not None else 'Không áp dụng'] for n in ORDER])
    monthly_table = table(['Tháng','Dương','Rule','Hành vi gốc','Hành vi bỏ biến lịch','Lịch sử mua','Kết hợp'],
                          [[m, next(r['positives'] for r in monthly if r['month']==m),
                            *[pct(next(r['top_k']['0.01']['recall'] for r in monthly if r['month']==m and r['model']==n)) for n in ORDER]] for m in months])
    comparison_table = table(['So sánh Recall@1%','Chênh lệch (đpt)','Khoảng bootstrap 95% (đpt)'],
                             [[NAMES[c['variant']]+' − '+NAMES[c['reference']],pp(c['estimate']),
                               '['+pp(c['interval_95'][0])+'; '+pp(c['interval_95'][1])+']'] for c in comparisons])
    report = f'''# Baseline propensity: kết quả và định hướng hoàn thiện

Cập nhật ngày **11/11/2026**. Phạm vi: nghiên cứu offline trên MBD-mini, dự đoán mua ít nhất một trong bốn sản phẩm có nhãn trong tháng tiếp theo. Kết quả công bố thuộc validation, test 01/2023 vẫn giữ kín.

## 1. Kết quả chính

Đã chạy **12 lần fit CatBoost trên dữ liệu thật**, gồm bốn bộ biến và ba fold thời gian, với một cấu hình cố định. Phương án có Recall@1% trung bình cao nhất trong lần chạy này là **{NAMES[best['model']]}: {pct(best['mean_recall_at_1pct'])}**.

Mô hình kết hợp đạt **{pct(by_name['combined']['mean_recall_at_1pct'])}**, so với **{pct(by_name['purchase']['mean_recall_at_1pct'])}** của mô hình lịch sử mua và **{pct(by_name['behavior']['mean_recall_at_1pct'])}** của mô hình hành vi bỏ biến độ phủ theo lịch. {verdict}

![So sánh Recall tại ngân sách 1%](recall_comparison.png)

{summary_table}

Các chỉ số là trung bình đều của 09–11/2022, trên cùng 100.224 khách mỗi tháng. Mỗi phương pháp chọn đúng **1.003 khách/tháng**. AP là Average Precision. Logloss dùng xác suất gốc của model từng fold, chưa calibration. Rule chỉ có quyết định 0/1 nên không so Logloss như một mô hình xác suất.

## 2. Bài toán và quần thể

Mỗi mẫu là một cặp khách hàng–tháng. Target `y_any=1` khi có ít nhất một trong bốn target bằng 1, bằng 0 khi cả bốn bằng 0, còn lại giữ unknown. Dữ liệu có **100.224 khách × 12 tháng = 1.202.688 dòng**, với **12.854 dòng dương, khoảng 1,07%**. Các dòng cùng khách qua nhiều tháng có phụ thuộc, không phải những mẫu độc lập.

Tại đầu tháng M, chỉ tổng hợp sự kiện trước cutoff để dự đoán mua trong tháng M. Nhãn lịch sử chỉ trở thành predictor khi đã khả dụng tại cutoff của chính dòng đó. Không đưa ID, fold hoặc target cùng tháng vào X. Khách thiếu geo/dialog vẫn nằm trong quần thể đánh giá.

Mục đích là xếp hạng danh sách khách theo ngân sách mô phỏng. Chênh số người mua trong danh sách không đo số lượt mua do chiến dịch tạo thêm, doanh thu hay ROI.

## 3. Thiết kế đánh giá đã khóa

{table(['Fold','Nhãn train','Validation','Dòng train'],[['1','02–08/2022','09/2022','701.568'],['2','02–09/2022','10/2022','801.792'],['3','02–10/2022','11/2022','902.016']])}

Cùng khách có thể xuất hiện ở các tháng train và validation, phù hợp kịch bản khách hiện hữu. Chia năm fold theo client phục vụ câu hỏi khách chưa thấy hoặc đối chiếu benchmark riêng, không thay thế đánh giá tháng tương lai.

Giữ **UTC và giả định độ trễ nhãn 0**, chưa xác nhận SLA của nguồn. Tháng 12/2022 dành cho calibration trong protocol cũ. Lần ablation này không đọc feature/nhãn calibration hoặc test, không chọn lại hay ghi đè model cuối đã lưu. Chỉ đọc metadata nguồn chứa phạm vi toàn bộ tháng.

## 4. Các đối chứng thực sự đã chạy

{table(['Mã','Đầu vào','Số biến','Mục đích'],[['B gốc','Hành vi tháng gần nhất và lịch sử cũ','153','Tái lập kết quả đã công bố'],['B','B gốc bỏ overlap-days và history-truncated của từng nguồn/cửa sổ','141','Kiểm tra ảnh hưởng của biến độ phủ theo lịch'],['H','Lịch sử nhãn mua đã khả dụng','15','Đối chứng mạnh hơn rule một lag'],['B+H','B kết hợp H','156','Đo giá trị tăng thêm của hành vi khi đã có lịch sử mua'],['R','Mua tháng trước','1','Rule đơn giản, cùng ngân sách']])}

H gồm `y_any` của ba tháng lịch liền trước, bốn target của tháng trước, số tháng có nhãn/số tháng dương/tỷ lệ trong ba tháng và toàn lịch sử, số tháng từ lần mua gần nhất và cờ có nhãn tháng trước. Thiếu tháng không được thay bằng tháng gần nhất còn lại. Không có lịch sử được giữ NaN cho các đại lượng chưa xác định. Số tháng lịch sử có nhãn là biến độ phủ và cũng có thể gợi thời gian; H và B+H dùng cùng tập biến này.

Việc bỏ 12 biến ở B chỉ kiểm soát các biến độ phủ nguồn theo lịch được chỉ định. Thí nghiệm chưa chứng minh đã loại mọi khả năng suy ra thời gian từ các biến khác, và chưa phải ablation riêng trx/geo/dialog.

CatBoost CPU: **depth 4, 200 cây, learning rate 0,04, L2 leaf 5, seed 20261110, 2 threads**. Giữ phân bố tự nhiên, không class weight hoặc resampling. Không tuning theo kết quả mới, không early stopping. Có lưu model từng fold để tái kiểm tra.

## 5. Độ ổn định và bất định

![Recall từng tháng validation](monthly_recall.png)

{monthly_table}

{comparison_table}

Khoảng phân vị từ **500 lần paired bootstrap theo client**, lấy cùng khách ở mọi tháng và tính lại top-K trong quần thể resample. Các khoảng có điều kiện theo những model và tháng đã quan sát, chưa bao quát drift ở tháng tương lai, biến động khi train lại hoặc toàn bộ quá trình lựa chọn. Validation đã được dùng nhiều lần cho phát triển nên các so sánh mang tính khám phá. Không có bằng chứng test độc lập trong báo cáo này.

Các bảng CSV đi kèm công bố Precision/Recall/Lift tại ngân sách 0,5%, 1% và 5%. Không gộp thứ hạng giữa các tháng.

## 6. Định hướng khoảng 160 biến được điều chỉnh thế nào

Giữ hướng bảng phẳng và GBDT. Đề xuất ban đầu cộng đúng **trx 110 + dialog 38 + geo 12 = 160 biến**. Thêm dialog cho B1 tạo 148 biến, thêm CoLES 64 chiều tạo 224 biến, giữ 256 chiều tạo 416 biến. Các con số này là thiết kế, khác schema 141/153/156 biến thực sự đã chạy.

- Amount được tổng hợp theo currency. Mã 11 là nhóm chính theo quy ước dự án, chưa xác định tên tiền hay đơn vị gốc. Không cộng amount khác currency thành tổng tiền có nghĩa kinh tế.
- Xu hướng ưu tiên rate theo độ dài cửa sổ hoặc hai khoảng không chồng lấn. Khóa cách xử lý mẫu số 0, cửa sổ bị cắt và ít sự kiện.
- Geo dùng thống kê ba mã geohash đã có. Chưa suy khoảng cách địa lý từ các số mã. Ô mới được so với lịch sử trước cửa sổ đang xét.
- Count 0 nghĩa là không có log trong phạm vi được đếm. Mean, std, recency và embedding không xác định giữ NaN. Không tự gán nguyên nhân thiếu nguồn.
- Vocabulary, top-K, PCA và encoder chỉ học trên dữ liệu được phép của fold và cutoff. Không fit PCA bằng embedding thiếu đã điền 0. Benchmark khách chưa thấy còn cần loại ID holdout khỏi fit biểu diễn.
- Không đặt ngưỡng bắt buộc 60–80 biến hay nén CoLES chỉ từ số dòng dương. Chọn bằng đối chứng validation, độ ổn định và chi phí. Gain phục vụ chẩn đoán, chưa đủ để quyết định bỏ nhóm biến tương quan.
- Tháng trong năm là thông tin biết trước nhưng một năm nhãn chưa đủ kiểm chứng mùa vụ. Baseline hiện chưa thêm biến này. Tiếp tục loại `trans_count` và `diff_trans_date` nguồn cho đến khi xác minh công thức và thời điểm tạo.

## 7. Phần đã hoàn tất và giới hạn còn lại

**Đã hoàn tất:** sửa trainer để tái dựng protocol từ cấu hình preparation và từ chối mọi sai khác; thêm kiểm thử tháng tương lai, train trùng validation, timezone và thời điểm nhãn khả dụng; triển khai feature lịch sử mua có kiểm soát cutoff; chạy 12 model và rule; đối chiếu tái lập baseline gốc khớp ba tháng; xuất metric và bootstrap; giữ artifact cũ và test kín.

**Chưa thực hiện:** cửa sổ mới 7/14/30/90 ngày, ablation từng nguồn, dialog PCA, CoLES, tuning XGBoost/LightGBM, backtest calibration nhiều giai đoạn, inference vận hành và test 01/2023. Đây là thí nghiệm tiếp theo, không được trình bày như kết quả đã đạt.

Ưu tiên tiếp theo là xác nhận giá trị tăng thêm của từng nguồn trên cùng nền lịch sử mua. Sau đó thử cửa sổ ngắn và dialog embedding. CoLES là bước nâng cấp sau khi các đối chứng này ổn định. Cần khóa lựa chọn và kiểm tra quy trình calibration trước khi đánh giá test độc lập.

## 8. Tái lập và kiểm chứng

Chạy từ root bằng Python ≥3.11 và các extra baseline, plots: `.\\.venv\\Scripts\\python.exe -u -B -X utf8 scripts/run_purchase_ablation.py --config configs/propensity_ablation.json`. Output cấu hình phải chưa tồn tại. Dùng config mới và output mới khi chạy lại, không ghi đè run cũ.

Xuất báo cáo bằng `scripts/publish_baseline_ablation.py --run outputs/propensity_ablation_20261111 --output deliverables/baseline-final`. Script xác minh hash artifact và metric tái lập trước khi công bố. Git chỉ chứa báo cáo, biểu đồ và bảng tổng hợp. Dự đoán theo khách, model từng fold và dữ liệu gốc nằm trong outputs/data đã được ignore.

Nguồn dự án: [báo cáo baseline gốc](../../BASELINE_REPORT.md), [thiết kế baseline](../../docs/propensity_baseline_design.md), [cấu hình ablation](../../configs/propensity_ablation.json), [mã tạo lịch sử mua](../../src/propensity_baseline/ablation.py). Nguồn phương pháp: [bài MBD](https://arxiv.org/html/2409.17587v1), [cross-validation theo thời gian](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-of-time-series-data), [tránh rò rỉ tiền xử lý](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage), [tham số XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html).
'''
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    (out/'report.html').write_text(markdown_html(report,out),encoding='utf-8')
    write('verification.json',dict(run_status=state['status'],test_opened=False,calibration_opened=False,
                                   original_baseline_parity=parity,source_fingerprint=state['source_fingerprint'],
                                   code_sha256=state['code_sha256'],parameters=state['parameters'],
                                   elapsed_seconds=state['elapsed_seconds'],environment=state['environment'],
                                   aggregate_artifact_sha256={p.name:sha256_file(p) for p in out.iterdir() if p.is_file()},
                                   raw_prediction_and_model_files_published=False))
    print(json.dumps(dict(output=str(out),best=best,combined_vs_purchase=combined_delta),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('outputs/propensity_ablation_20261111'))
    parser.add_argument('--output',type=Path,default=Path('deliverables/baseline-final'))
    options = parser.parse_args()
    publish(options.run, options.output)
