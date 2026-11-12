"""Check whether source trans_count equals cumulative positive target labels."""
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; O=ROOT/'outputs/trans_count_target_history'; O.mkdir(exist_ok=True)
a=pd.read_parquet(ROOT/'outputs/dataset_eda/customer_month_snapshots.parquet',columns=['client_id','outcome_month','trans_count','y_any',*[f'target_{i}' for i in range(1,5)]]).sort_values(['client_id','outcome_month']).reset_index(drop=True)
assert not a.duplicated(['client_id','outcome_month']).any()
a['all_product_positives']=a[[f'target_{i}' for i in range(1,5)]].sum(axis=1)
rows=[]; monthly=[]
for col in ['y_any','all_product_positives',*[f'target_{i}' for i in range(1,5)]]:
 cum=a.groupby('client_id')[col].cumsum()
 for mode,v in [('prior_only',cum-a[col]),('including_current_diagnostic',cum)]:
  name=col+'_'+mode; same=a.trans_count.eq(v); a[name]=v
  for cohort,mask in [('all',np.ones(len(a),dtype=bool)),('exclude_first_observed_month',a.outcome_month.gt(a.outcome_month.min()).to_numpy()),('source_positive',a.trans_count.gt(0).to_numpy())]:
   rows.append(dict(candidate=name,cohort=cohort,rows=int(mask.sum()),exact=int(same[mask].sum()),match_pct=float(same[mask].mean()*100),candidate_max=float(v[mask].max()),source_max=int(a.loc[mask,'trans_count'].max()),mean_absolute_error=float((a.loc[mask,'trans_count']-v[mask]).abs().mean())))
  for month,idx in a.groupby('outcome_month').groups.items(): monthly.append(dict(candidate=name,month=str(month.date()),rows=len(idx),exact=int(same.loc[idx].sum()),match_pct=float(same.loc[idx].mean()*100)))
# Count distinct products ever positive before cutoff (rather than positive months).
prior_flags=[]
for i in range(1,5):
 col=f'target_{i}'; prior=(a.groupby('client_id')[col].cumsum()-a[col]).gt(0); prior_flags.append(prior.astype(int))
v=sum(prior_flags); a['distinct_products_prior']=v; same=a.trans_count.eq(v)
rows.append(dict(candidate='distinct_products_prior',cohort='all',rows=len(a),exact=int(same.sum()),match_pct=float(same.mean()*100),candidate_max=int(v.max()),source_max=int(a.trans_count.max()),mean_absolute_error=float((a.trans_count-v).abs().mean())))
r=pd.DataFrame(rows); r.to_csv(O/'candidate_summary.csv',index=False); pd.DataFrame(monthly).to_csv(O/'candidate_by_month.csv',index=False)
a.loc[a.trans_count.gt(a.all_product_positives_including_current_diagnostic)].head(30).to_csv(O/'counterexamples.csv',index=False)
summary={'rows':len(a),'clients':a.client_id.nunique(),'observed_label_months':a.outcome_month.nunique(),'source_exceeds_all_observed_product_positives_to_current':int(a.trans_count.gt(a.all_product_positives_including_current_diagnostic).sum()),'first_month_positive_trans_count':int(a.loc[a.outcome_month.eq(a.outcome_month.min()),'trans_count'].gt(0).sum()),'caveat':'Targets are binary monthly indicators, not purchase transaction counts. Earlier purchase history before the first label month is unobserved; cannot reject a definition using unavailable history.'}
import json
(O/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print(r[r.cohort=='all'][['candidate','exact','match_pct','candidate_max']].to_string(index=False)); print(summary)
