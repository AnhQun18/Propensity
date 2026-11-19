"""Descriptive Spearman correlations on the fixed historical EDA cohort only."""
from pathlib import Path
import json
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]; O=ROOT/'outputs/spearman_heatmap'
protocol=json.loads((ROOT/'configs/eda_analysis.json').read_text()); months=protocol['analysis_months']
features={'trx_events':'TRX: số giao dịch','trx_active_days':'TRX: ngày active','trx_amount_currency11':'TRX: tổng amount mã 11','trx_recency_days':'TRX: recency','geo_events':'GEO: số log','geo_active_days':'GEO: ngày active','geo_unique_regions':'GEO: số mã cấp 6','geo_dominant_share':'GEO: tỷ trọng vùng chính','geo_entropy':'GEO: entropy vùng','geo_recency_days':'GEO: recency','dialog_events':'Dialog: số hội thoại','dialog_active_days':'Dialog: ngày active','dialog_recency_days':'Dialog: recency'}
a=pd.read_parquet(ROOT/'outputs/dataset_eda/customer_month_snapshots.parquet',columns=['client_id','outcome_month',*features]); a=a[a.outcome_month.dt.strftime('%Y-%m').isin(months)].copy(); assert len(a)==100224*len(months)
raw=a[list(features)].copy(); observed=raw.copy()
for src in ['trx','geo','dialog']:
 cols=[f for f in features if f.startswith(src+'_')]; observed.loc[a[src+'_events'].eq(0),cols]=np.nan
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':'#193454'})
for tag,data,subtitle in [('observed',observed,'Chỉ tính đặc trưng nguồn khi có log tháng T−1; loại ảnh hưởng của các số 0 do thiếu nguồn'),('including_zero',raw,'Giữ số 0 của count/ngày active/amount khi không có log; các giá trị thiếu khác giữ nguyên')]:
 corr=data.corr(method='spearman',min_periods=100); available=data.notna().astype(np.int64); pairs=available.T.dot(available)
 corr.to_csv(O/f'spearman_{tag}.csv'); pairs.to_csv(O/f'pair_counts_{tag}.csv')
 f,ax=plt.subplots(figsize=(14,12),dpi=220); f.subplots_adjust(left=.23,right=.91,bottom=.25,top=.84)
 im=ax.imshow(corr.to_numpy(),vmin=-1,vmax=1,cmap='RdBu_r'); ax.set_xticks(range(len(features)),features.values(),rotation=48,ha='right',fontsize=10); ax.set_yticks(range(len(features)),features.values(),fontsize=10)
 for i in range(len(features)):
  for j in range(len(features)):
   val=corr.iloc[i,j]
   if pd.notna(val): ax.text(j,i,f'{val:.2f}',ha='center',va='center',fontsize=8.5,color='white' if abs(val)>.6 else '#20354e')
 ax.set_xticks(np.arange(-.5,len(features),1),minor=True); ax.set_yticks(np.arange(-.5,len(features),1),minor=True); ax.grid(which='minor',color='white',lw=.5); ax.tick_params(which='minor',bottom=False,left=False)
 cb=f.colorbar(im,ax=ax,fraction=.035,pad=.03); cb.set_label('Spearman ρ')
 f.text(.055,.955,'Tương quan Spearman giữa các đặc trưng hành vi',fontsize=22,fontweight='bold'); f.text(.055,.91,f'Cohort 02–08/2022 · {len(a):,} khách–tháng · Chỉ dùng các tháng đã chọn'.replace(f'{len(a):,}',f'{len(a):,}'.replace(',','.')),fontsize=12,color='#62738a'); f.text(.055,.875,subtitle,fontsize=10.5,color='#62738a')
 f.text(.055,.065,'Count, active days, amount, đa dạng GEO: tháng T−1; recency: lịch sử trước đầu tháng T. Không đưa target hoặc mã ID vào ma trận.',fontsize=10,color='#62738a'); f.text(.055,.03,'Tính theo từng cặp đủ dữ liệu; số quan sát từng cặp được xuất CSV. Đây là tương quan mô tả, không chứng minh quan hệ nhân quả.',fontsize=10,color='#62738a'); f.savefig(O/f'spearman_{tag}.png'); plt.close(f)
 if tag=='observed':
  upper=corr.where(np.triu(np.ones(corr.shape),1).astype(bool)).stack(); best=upper.abs().sort_values(ascending=False).head(8); print('Top observed correlations:'); print(upper.loc[best.index].to_string()); print('range n',pairs.to_numpy().min(),pairs.to_numpy().max())
(O/'method.json').write_text(json.dumps(dict(train_months=months,rows=len(a),clients=a.client_id.nunique(),features=features,correlation='Spearman, pairwise complete, min_periods=100',observed='Mask all source features when source count in T-1 equals zero; includes recency masking for this version',tests='No holdout loaded for analysis; valid coefficients bounded [-1,1], symmetric matrix',note='Repeated customer months; descriptive coefficients only, no iid p-values'),indent=2,ensure_ascii=False),encoding='utf-8')
