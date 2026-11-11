"""Validation-only source ablation and matched window sensitivity for the deck."""
from pathlib import Path
import sys,json,time
import numpy as np
import pandas as pd
import duckdb

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from propensity_baseline.config import load_baseline_config
from propensity_baseline.runner import source_contract,read_months,fit_model,predict_frame,monthly_metrics,sha256_file
from propensity_baseline.ablation import purchase_features,feature_variants,PURCHASE_FEATURES
from propensity_baseline.metrics import ranked_indices

cfg=load_baseline_config(ROOT/'configs/propensity_ablation.json')
manifest,schema,protocol,inventory=source_contract(cfg)
old=ROOT/'outputs/propensity_ablation_20261111'
oldmanifest=json.loads((old/'manifest.json').read_text())
assert oldmanifest['status']=='complete' and not oldmanifest['test_opened']
for name,digest in oldmanifest['artifact_sha256'].items():
 assert sha256_file(old/name)==digest, name

run=ROOT/'outputs/data_story_models_v2'
run.mkdir(exist_ok=True)
checkpoint=json.loads((run/'manifest.json').read_text()) if (run/'manifest.json').exists() else None
if checkpoint and checkpoint['status']=='complete':
 raise ValueError('Completed run is immutable; use a new output directory')
pub=ROOT/'deliverables/propensity-data-story/analysis'
pub.mkdir(exist_ok=True)
started=time.monotonic()
state=dict(status='running',test_opened=False,calibration_opened=False,
config=cfg.to_dict(),protocol=protocol,source_inventory=inventory,
code_sha256=sha256_file(Path(__file__)),fits=checkpoint['fits'] if checkpoint else [],
 design='Exploratory fixed-hyperparameter validation comparisons; no tuning or test selection')
def statewrite():
 (run/'manifest.json').write_text(json.dumps(state,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
statewrite()

variants=feature_variants(schema)
allmonths=sorted(set(m for f in protocol['rolling_validation'] for m in [*f['train'],f['validate']]))
assert not set(allmonths)&{protocol['calibration'],protocol['test']}
frame=read_months(cfg,allmonths,variants['behavior'],schema,manifest)
history=frame[['client_id','month','y_any',*[f'target_{i}' for i in range(1,5)]]]
frame=pd.concat([frame,purchase_features(frame,history,protocol)],axis=1)
assert len(frame)==1002240
existing=pd.read_parquet(old/'validation_predictions.parquet')
records=[]
importance=[]
predictions=[]

def summarize(records):
 return pd.DataFrame([dict(model=r['model'],month=r['month'],ap=r['average_precision'],
 recall1=r['top_k']['0.01']['recall'],precision1=r['top_k']['0.01']['precision'],
 hits1=r['top_k']['0.01']['hits']) for r in records])

for fold in protocol['rolling_validation']:
 train=frame.loc[frame.month.isin(fold['train'])]
 valid=frame.loc[frame.month.eq(fold['validate'])]
 for source in ['trx','geo','dialog']:
  name='without_'+source
  features=[f for f in variants['combined'] if not f.startswith(source+'_')]
  print('SOURCE',fold['validate'],name,len(features),flush=True)
  from catboost import CatBoostClassifier
  model_file=run/f"{fold['validate']}_{name}.cbm"
  model=CatBoostClassifier().load_model(str(model_file)) if model_file.exists() else fit_model(cfg,train,features,dict(depth=4,iterations=200))
  pred=predict_frame(cfg,model,valid,features,name)
  predictions.append(pred)
  records.extend(monthly_metrics(pred,cfg.seed))
  model.save_model(str(run/f"{fold['validate']}_{name}.cbm"))
  fit_record=dict(month=fold['validate'],variant=name,predictors=len(features))
  if fit_record not in state['fits']:state['fits'].append(fit_record)
  statewrite()
 # Importance is descriptive, obtained from existing full fold models.
 from catboost import CatBoostClassifier
 model=CatBoostClassifier().load_model(str(old/f"model_{fold['validate']}_combined.cbm"))
 for f,v in zip(model.feature_names_,model.get_feature_importance()):
  importance.append(dict(month=fold['validate'],feature=f,source=f.split('_')[0],importance=v))
 del train,valid,model

combined=existing.loc[existing.model.eq('combined')]
records.extend(monthly_metrics(combined,cfg.seed))
source_metrics=summarize(records)
source_metrics.to_csv(pub/'source_ablation_monthly.csv',index=False,lineterminator='\n')
source_summary=source_metrics.groupby('model')[['ap','recall1','precision1','hits1']].mean()
source_summary['ap_loss_vs_full']=source_summary.loc['combined','ap']-source_summary.ap
source_summary['recall_loss_vs_full']=source_summary.loc['combined','recall1']-source_summary.recall1
source_summary.reset_index().to_csv(pub/'source_ablation_summary.csv',index=False,lineterminator='\n')
pd.DataFrame(importance).groupby('source').importance.sum().div(3).rename('mean_importance').to_csv(pub/'importance_by_source.csv',lineterminator='\n')

# Errors for the CURRENT B+H model vs same-budget rule; cohorts use past only.
segments=[];overlaps=[]
rule=existing.loc[existing.model.eq('previous_month_purchase')]
for month,g in combined.groupby('month'):
 r=rule.loc[rule.month.eq(month)].set_index('client_id').loc[g.client_id].reset_index()
 assert np.array_equal(g.y_any.to_numpy(),r.y_any.to_numpy())
 meta=frame.loc[frame.month.eq(month)].set_index('client_id').loc[g.client_id]
 a=np.zeros(len(g),bool); b=a.copy()
 a[ranked_indices(g.client_id,g.score,cfg.seed)[:1003]]=True
 b[ranked_indices(r.client_id,r.score,cfg.seed)[:1003]]=True
 y=g.y_any.to_numpy()==1
 overlaps.append(dict(month=month,common_hits=int((a&b&y).sum()),combined_only_hits=int((a&~b&y).sum()),rule_only_hits=int((~a&b&y).sum()),missed_by_both=int((~a&~b&y).sum()),overlap_selected=int((a&b).sum())))
 groups={
  'bought_lag1':meta.purchase_any_lag1.eq(1).to_numpy(),
  'not_bought_lag1':meta.purchase_any_lag1.eq(0).to_numpy(),
  'no_purchase_history':meta.purchase_positive_months_history.eq(0).to_numpy(),
  'prior_purchase_history':meta.purchase_positive_months_history.gt(0).to_numpy(),
  'no_dialog_history':meta.dialog_ever_seen_source.eq(0).to_numpy(),
  'has_dialog_history':meta.dialog_ever_seen_source.eq(1).to_numpy(),
  'no_geo_history':meta.geo_ever_seen_source.eq(0).to_numpy(),
  'has_geo_history':meta.geo_ever_seen_source.eq(1).to_numpy(),
 }
 for name,mask in groups.items():
  for model,selected in [('combined',a),('rule',b)]:
   n=int(mask.sum());p=int((mask&y).sum());tp=int((mask&selected&y).sum());k=int((mask&selected).sum())
   segments.append(dict(month=month,segment=name,model=model,n=n,positives=p,selected=k,hits=tp,misses=p-tp,recall=tp/p if p else np.nan,precision=tp/k if k else np.nan))
pd.DataFrame(segments).to_csv(pub/'error_segments.csv',index=False,lineterminator='\n')
pd.DataFrame(overlaps).to_csv(pub/'error_overlap.csv',index=False,lineterminator='\n')

# Matched 31-predictor experiment: one behavioral window (16 summaries) + same H (15).
# Keep identical feature names/types/count and cutoffs; only window values change.
suffixes={s:['has_event','log1p_event_count','active_days','days_since_last_event'] for s in ['trx','geo','dialog']}
suffixes['trx']+=['log1p_main_amount_abs_sum','event_type_nunique']
suffixes['geo']+=['geohash_6_nunique','top_geohash_6_share']
generic=[f'{s}_{f}' for s,ss in suffixes.items() for f in ss]
common=[*generic,*PURCHASE_FEATURES]
window_records=[]
for tag in ['recent','30d','84d','all']:
 window=frame[['client_id','month','y_any','cold_start','trx_ever_seen_source','geo_ever_seen_source','dialog_ever_seen_source',*PURCHASE_FEATURES]].copy()
 selected=[f'{s}_{tag}_{f}' for s,ss in suffixes.items() for f in ss]
 with duckdb.connect() as con:
  parts=[]
  root=cfg.source_dir if tag=='recent' else ROOT/'outputs/propensity_data_v1'
  for month in allmonths:
   file=str(root/f'features/month={month}/part.parquet').replace('\\','/')
   part=con.execute('SELECT client_id,'+','.join(selected)+f" FROM read_parquet('{file}',hive_partitioning=false)").df()
   part['month']=month;parts.append(part)
 values=pd.concat(parts,ignore_index=True).rename(columns=dict(zip(selected,generic)))
 window=window.merge(values,on=['client_id','month'],validate='one_to_one')
 del values,parts
 for fold in protocol['rolling_validation']:
  print('WINDOW',tag,fold['validate'],len(common),flush=True)
  train=window.loc[window.month.isin(fold['train'])];valid=window.loc[window.month.eq(fold['validate'])]
  model=fit_model(cfg,train,common,dict(depth=4,iterations=200))
  pred=predict_frame(cfg,model,valid,common,'window_'+tag)
  window_records.extend(monthly_metrics(pred,cfg.seed))
  model.save_model(str(run/f"{fold['validate']}_window_{tag}.cbm"))
  state['fits'].append(dict(month=fold['validate'],variant='window_'+tag,predictors=len(common)))
  statewrite()
  del train,valid,model
 del window
wm=summarize(window_records)
wm.to_csv(pub/'window_sensitivity_monthly.csv',index=False,lineterminator='\n')
wm.groupby('model')[['ap','recall1','precision1']].mean().reset_index().to_csv(pub/'window_sensitivity_summary.csv',index=False,lineterminator='\n')
pd.concat(predictions).to_parquet(run/'source_predictions.parquet',index=False)
assert source_contract(cfg)[3]==inventory
state.update(status='complete',elapsed_seconds=time.monotonic()-started,
 artifact_sha256={p.name:sha256_file(p) for p in run.iterdir() if p.suffix=='.cbm'},
 source_summary=source_summary.reset_index().to_dict(orient='records'))
statewrite()
public={k:state[k] for k in ['status','test_opened','calibration_opened','design','fits','elapsed_seconds','code_sha256']}
public['parameters']=dict(depth=4,iterations=200,learning_rate=.04,l2_leaf_reg=5,seed=cfg.seed)
public['source_features']=variants['combined']
public['window_features']=common
public['protocol']=protocol
(pub/'model_manifest.json').write_text(json.dumps(public,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(source_summary,flush=True)
print(wm.groupby('model')[['ap','recall1']].mean(),flush=True)
