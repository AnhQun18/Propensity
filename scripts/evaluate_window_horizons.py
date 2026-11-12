"""Re-evaluate saved window models on rolling validation only; no training/test access.
Run: .venv/Scripts/python.exe scripts/evaluate_window_horizons.py
"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import duckdb
from catboost import CatBoostClassifier
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,auc,brier_score_loss
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from propensity_baseline.config import load_baseline_config
from propensity_baseline.runner import source_contract,read_months,numeric_x,sha256_file
from propensity_baseline.ablation import purchase_features,PURCHASE_FEATURES
OUT=ROOT/'outputs/window_horizon_evaluation'; OUT.mkdir(exist_ok=True)
cfg=load_baseline_config(ROOT/'configs/propensity_ablation.json')
manifest,schema,protocol,_=source_contract(cfg)
saved=ROOT/'outputs/data_story_models_v2'
model_manifest=json.loads((saved/'manifest.json').read_text())
assert model_manifest['status']=='complete'
folds=protocol['rolling_validation']; months=sorted({m for f in folds for m in [*f['train'],f['validate']]})
assert not set(months)&{protocol['test'],protocol['calibration']}
# Read metadata plus one permitted predictor to reuse the label-contract checks.
frame=read_months(cfg,months,[schema['predictors'][0]],schema,manifest)
history=frame[['client_id','month','y_any',*[f'target_{i}' for i in range(1,5)]]]
frame=pd.concat([frame,purchase_features(frame,history,protocol)],axis=1)
suffixes={s:['has_event','log1p_event_count','active_days','days_since_last_event'] for s in ['trx','geo','dialog']}
suffixes['trx']+=['log1p_main_amount_abs_sum','event_type_nunique']
suffixes['geo']+=['geohash_6_nunique','top_geohash_6_share']
generic=[f'{s}_{f}' for s,ss in suffixes.items() for f in ss]; common=[*generic,*PURCHASE_FEATURES]
expected=pd.read_csv(ROOT/'deliverables/propensity-data-story/analysis/window_sensitivity_monthly.csv')
records=[]; predictions=[]; hashes={}
for tag in ['recent','30d','84d','all']:
 for fold in folds:
  month=fold['validate']; print(tag,month,flush=True)
  root=cfg.source_dir if tag=='recent' else ROOT/'outputs/propensity_data_v1'
  selected=[f'{s}_{tag}_{f}' for s,ss in suffixes.items() for f in ss]
  file=root/f'features/month={month}/part.parquet'
  with duckdb.connect() as con:
   part=con.execute('SELECT client_id,'+','.join(selected)+" FROM read_parquet(?,hive_partitioning=false)",[str(file)]).df()
  part=part.rename(columns=dict(zip(selected,generic)))
  valid=frame.loc[frame.month.eq(month),['client_id','month','y_any',*PURCHASE_FEATURES]].merge(part,on='client_id',validate='one_to_one')
  model_path=saved/f'{month}_window_{tag}.cbm'; digest=sha256_file(model_path)
  assert model_manifest['artifact_sha256'][model_path.name]==digest
  hashes[model_path.name]=digest
  model=CatBoostClassifier(); model.load_model(str(model_path)); assert model.feature_names_==common
  y=valid.y_any.to_numpy(dtype=int); score=model.predict(numeric_x(valid,common),prediction_type='RawFormulaVal',thread_count=cfg.threads); p=expit(score)
  precision,recall,_=precision_recall_curve(y,score)
  ap=average_precision_score(y,score)
  prior=float(frame.loc[frame.month.isin(fold['train']),'y_any'].mean())
  baseline=brier_score_loss(y,np.full(len(y),prior)); brier=brier_score_loss(y,p)
  previous=float(expected.loc[(expected.model=='window_'+tag)&(expected.month==month),'ap'].iloc[0]); assert np.isclose(ap,previous,rtol=1e-9,atol=1e-12),(ap,previous)
  records.append(dict(window=tag,cutoff=month+'-01',n=len(y),positives=int(y.sum()),prevalence=y.mean(),roc_auc=roc_auc_score(y,score),average_precision=ap,pr_auc_trapezoid=auc(recall,precision),brier=brier,training_prevalence=prior,brier_train_prior_baseline=baseline,brier_skill=1-brier/baseline))
  predictions.append(valid[['client_id','month','y_any']].assign(window=tag,probability=p,score=score))
  pd.DataFrame(records).to_csv(OUT/'metrics_by_cutoff.csv',index=False)
metrics=pd.DataFrame(records)
summary=metrics.groupby('window')[['roc_auc','average_precision','pr_auc_trapezoid','brier','brier_skill']].agg(['mean','std']); summary.columns=['_'.join(c) for c in summary.columns]; summary=summary.reset_index().sort_values('average_precision_mean',ascending=False); summary.to_csv(OUT/'window_summary.csv',index=False)
pd.concat(predictions,ignore_index=True).to_parquet(OUT/'validation_predictions.parquet',index=False)
(OUT/'manifest.json').write_text(json.dumps(dict(protocol=protocol,model_hashes=hashes,code_sha256=sha256_file(Path(__file__)),selection_metric='macro mean validation average precision',probabilities='Raw saved model probabilities; no calibration fitted; Brier evaluates these probabilities.',windows=dict(recent='Previous calendar month',**{'30d':'30 days','84d':'84 days','all':'All available behavior before cutoff'}),design='Same 16 behavioral summaries + 15 purchase-history predictors; saved fixed-parameter models. No calibration/test opened.'),ensure_ascii=False,indent=2),encoding='utf-8')
print(summary.to_string(index=False))
