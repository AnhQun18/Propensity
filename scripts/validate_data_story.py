"""Validate published aggregates against counts, model contracts, and outline coverage."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables/propensity-data-story'
A=OUT/'analysis'
def csv(name):return pd.read_csv(A/(name+'.csv'))
checks={}
def checked(name,condition):
    if not condition:raise ValueError(name)
    checks[name]=True

d=csv('feature_deciles')
checked('decile_population_conservation',d.groupby('feature').n.sum().eq(701568).all())
checked('decile_positive_conservation',d.groupby('feature').positives.sum().eq(8672).all())
for feature,g in d[d.band.gt(0)].groupby('feature'):
    g=g.sort_values('band')
    checked('ties_not_split_'+feature,all(a<b for a,b in zip(g.upper_value.iloc[:-1],g.lower_value.iloc[1:])))
co=csv('target_cohort_drift')
checked('cohort_population',co.groupby('month').n.sum().eq(100224).all())
checked('fixed_cohort_membership_counts',co.groupby('cohort').n.nunique().eq(1).all())
target=pd.read_csv(ROOT/'docs/propensity_feasibility/any_product/monthly.csv').set_index('month')
checked('cohort_labels_match_target',all(v==target.loc[m,'positives'] for m,v in co.groupby('month').positives.sum().items()))
cat=csv('category_lift')
checked('category_share_conservation',abs(cat.event_share_pct.sum()-100)<1e-8)
checked('category_unique_customer_month_presence',cat.present_rows.le(701568).all() and cat.positives.le(8672).all())
checked('category_rates',np.allclose(cat.rate_pct,100*cat.positives/cat.present_rows))
geo=csv('geo_quality_proxy').iloc[0]
checked('geo_order_pair_conservation',int(geo.pairs)==int(geo.rows-geo.clients))
checked('geo_change_proxy_counts',0<=geo.changes_within60s<=geo.pairs_within60s<=geo.pairs)

manifest=json.loads((A/'model_manifest.json').read_text(encoding='utf-8'))
checked('experiment_code_hash_matches',manifest['code_sha256']==hashlib.sha256((ROOT/'scripts/run_data_story_models.py').read_bytes()).hexdigest())
checked('no_holdout_opened',not manifest['test_opened'] and not manifest['calibration_opened'])
checked('complete_21_fits',manifest['status']=='complete' and len(manifest['fits'])==21)
checked('window_matched_predictor_count',all(r['predictors']==31 for r in manifest['fits'] if r['variant'].startswith('window_')))
expected={'without_trx':59,'without_geo':127,'without_dialog':141}
checked('source_drop_predictor_contract',all(r['predictors']==expected[r['variant']] for r in manifest['fits'] if r['variant'] in expected))
for fold in manifest['protocol']['rolling_validation']:
    checked('strict_train_before_'+fold['validate'],all(m<fold['validate'] for m in fold['train']))
s=csv('source_ablation_monthly')
checked('source_fold_set',len(s)==12 and set(s.month)=={'2022-09','2022-10','2022-11'})
old=pd.read_csv(ROOT/'deliverables/baseline-final/summary.csv').set_index('model')
summary=csv('source_ablation_summary').set_index('model')
checked('reference_ap_matches_existing_full',abs(summary.loc['combined','ap']-old.loc['combined','mean_ap'])<1e-12)
checked('reference_recall_matches_existing_full',abs(summary.loc['combined','recall1']-old.loc['combined','mean_recall_at_1pct'])<1e-12)
checked('delta_ap_correct',np.allclose(summary.ap_loss_vs_full,summary.loc['combined','ap']-summary.ap))
wm=csv('window_sensitivity_monthly')
checked('window_fold_set',len(wm)==12 and wm.groupby('model').size().eq(3).all())
error=csv('error_segments')
for (month,model),g in error.groupby(['month','model']):
    partition=g[g.segment.isin(['bought_lag1','not_bought_lag1'])]
    checked('error_partition_'+month+'_'+model,int(partition.n.sum())==100224 and int(partition.selected.sum())==1003 and int(partition.positives.sum())==int(target.loc[month,'positives']))
    if model=='combined':
        checked('error_tp_matches_full_'+month,int(partition.hits.sum())==int(s[(s.month==month)&s.model.eq('combined')].hits1.iloc[0]))
ledger=json.loads((OUT/'coverage_ledger.json').read_text(encoding='utf-8'))
checked('all_26_requirements_located',len(ledger)==26 and all(v['main_slides'] or v['detailed_slides'] or v['appendix_slides'] for v in ledger))
for file,count in [('outline.json',19),('detailed_outline.json',23)]:
    outline=json.loads((OUT/file).read_text(encoding='utf-8'))
    checked(file+'_narrative_count',sum(not s['appendix'] for s in outline)==count)
    checked(file+'_definitions',all(s['meaning'] and s['method'] and s['source'] for s in outline))
hashes={str(p.relative_to(OUT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in A.iterdir() if p.is_file()}
report=dict(status='passed',checks=checks,aggregate_sha256=hashes)
(OUT/'content_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def f(v,n=4):return f'{float(v):.{n}f}'.replace('.',',')
md=['# Phân tích bổ sung sau rà soát deck','',
 'Tất cả thí nghiệm dưới đây chỉ dùng validation 09–11/2022, cùng 100.224 ID/tháng. Test 01/2023 và calibration không mở. Cấu hình cố định depth 4, 200 cây, learning rate 0,04, L2 5, seed 20261110.',
 '', '## Bỏ từng nguồn trên cùng nền lịch sử mua', '',
 '9 fit mới, full B+H 156 biến tái sử dụng model đã lưu sau xác minh hash. ΔAP = AP full − AP reduced; đóng góp có điều kiện theo các nguồn còn lại, không cộng thành tổng.', '',
 '| Phương án | AP macro | Recall@1% macro | ΔAP | Giảm recall (đpt) |', '|---|---:|---:|---:|---:|']
for model,row in summary.iterrows():
 md.append(f'| {model} | {f(row.ap)} | {f(row.recall1*100,2)}% | {f(row.ap_loss_vs_full,6)} | {f(row.recall_loss_vs_full*100,2)} |')
md+=['','Geo có chênh nhỏ và đổi dấu ở tháng 10; chưa có kiểm định độ ổn định trên nhiều seed hoặc tháng độc lập. Trx và dialog có ΔAP dương ở cả ba tháng. Không suy nhân quả từ ablation này.',
 '', '## Cùng bộ biến, khác độ dài cửa sổ', '',
 '12 fit mới, cùng 16 thống kê hành vi + 15 biến H. Chỉ thay giá trị cửa sổ; cutoff, cohort, split và cấu hình giữ nguyên. Không so trực tiếp kết quả bộ 31 biến này thành tác dụng cửa sổ cho full 156 biến.', '',
 '| Cửa sổ | AP macro | Recall@1% macro |', '|---|---:|---:|']
for _,row in csv('window_sensitivity_summary').iterrows():md.append(f'| {row.model} | {f(row.ap)} | {f(row.recall1*100,2)}% |')
md+=['','Nhãn tháng không xác định được nhãn mua trong 30 ngày sau một ngày T bất kỳ. Có thể tạo feature tại T; đổi horizon cần nhãn tương thích và backtest riêng.',
 '', '## Lỗi của B+H mới nhất', '',
 '| Nhóm | Dương | Được chọn | TP | FP | Recall nhóm | Precision nhóm |', '|---|---:|---:|---:|---:|---:|---:|']
for key,g in error[error.model.eq('combined')].groupby('segment'):
 p,k,tp=int(g.positives.sum()),int(g.selected.sum()),int(g.hits.sum())
 md.append(f'| {key} | {p} | {k} | {tp} | {k-tp} | {f(100*tp/p,2)}% | {f(100*tp/k,2) if k else "—"}% |')
md+=['','Các nhóm có chồng lấn. Top-K được chọn trên toàn cohort rồi mới phân nhóm; gộp TP/dương qua ba tháng, khác macro đều tháng. Thiếu dialog là trạng thái log trước cutoff, không tự chứng minh lỗi thu thập.',
 '', '## Giới hạn dữ liệu cần giữ đúng', '',
 '- Nhãn nhị phân chỉ đo tháng dương, không đếm số đơn hoặc số lần mua trong tháng.',
 '- Không có text, duration hoặc độ dài hội thoại; embedding 768D chưa được dùng trong các thí nghiệm này.',
 '- Mã geo ẩn danh không cho phép suy quãng đường thật. Đổi mã nhanh là proxy để kiểm tra, chưa phải chứng cứ nhiễu GPS.',
 '- Khoảng trx trước 02/2022 chưa có target là lịch sử; SLA công bố nhãn thực tế chưa xác minh.',
 '', 'Chi tiết số liệu: CSV trong analysis/. Đối chiếu đề cương: coverage_ledger.md.']
(OUT/'ANALYSIS.md').write_text('\n'.join(md)+'\n',encoding='utf-8',newline='\n')
print('PASS:',len(checks),'data/model/coverage checks')
