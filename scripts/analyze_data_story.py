"""Additional descriptive evidence for the data-story deck; train labels only."""
from pathlib import Path
import json
import sys
import duckdb

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables/propensity-data-story/analysis'
OUT.mkdir(exist_ok=True)
c=duckdb.connect()
c.execute("SET memory_limit='2GB'")
c.execute('SET threads=2')

def files(root,months,leaf='part.parquet'):
    return [str(ROOT/root/f'month={m}'/leaf).replace('\\','/') for m in months]

months=[f'2022-{m:02d}' for m in range(2,9)]
allmonths=[f'2022-{m:02d}' for m in range(2,12)]
c.read_parquet(files(Path('outputs/propensity_data_v1/features'),months),hive_partitioning=True).create_view('tr')
c.read_parquet(files(Path('outputs/propensity_data_v1/features'),allmonths),hive_partitioning=True).create_view('tv')

checks={}
def save(name,sql):
    frame=c.execute(sql).df()
    frame.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8',lineterminator='\n')
    checks[name]=len(frame)
    return frame

# Fixed, pre-label-period cohort membership; it does not inspect a future last event.
save('target_cohort_drift',"""
WITH cohort AS (
 SELECT client_id,CASE WHEN trx_all_has_event=0 THEN 'no_trx'
 WHEN trx_all_days_since_first_recorded_event>=180 THEN 'history_180d'
 ELSE 'history_under180d' END cohort FROM tv WHERE month='2022-02')
SELECT t.month,c.cohort,count(*) n,sum(y_any) positives,avg(y_any)*100 rate_pct
FROM tv t JOIN cohort c USING(client_id) GROUP BY 1,2 ORDER BY 1,2
""")

save('sparsity_monthly',"""SELECT month,count(*) n,
count(*) FILTER(WHERE trx_30d_has_event=0) silent_30d,
count(*) FILTER(WHERE trx_84d_has_event=0) silent_84d,
count(*) FILTER(WHERE trx_all_has_event=0) never_observed,
avg(y_any) FILTER(WHERE trx_84d_has_event=0)*100 silent84_rate_pct
FROM tv GROUP BY 1 ORDER BY 1""")

features={'R':'trx_30d_days_since_last_event','F':'trx_30d_event_count',
 'M':'trx_30d_main_amount_abs_sum','geo_diversity':'geo_30d_geohash_6_nunique',
 'geo_concentration':'geo_30d_top_geohash_6_share',
 'dialog_count':'dialog_30d_event_count','dialog_recency':'dialog_30d_days_since_last_event'}
parts=[]
for label,feature in features.items():
    # Aggregate ties before deciling. Equal feature values always share a band.
    # Bands refer to cumulative rank midpoints, so some are absent with many zeros.
    parts.append(f"""WITH freq AS (SELECT {feature} v,count(*) n FROM tr WHERE {feature} IS NOT NULL GROUP BY 1),
    cut AS (SELECT v,least(10,1+floor(10*(sum(n) OVER(ORDER BY v)-n/2.0)/sum(n) OVER()))::INT band FROM freq)
    SELECT '{label}' feature,coalesce(band,0) band,min(t.{feature}) lower_value,max(t.{feature}) upper_value,
    count(*) n,sum(y_any) positives,avg(y_any)*100 rate_pct,avg(y_any)/(SELECT avg(y_any) FROM tr) lift
    FROM tr t LEFT JOIN cut ON t.{feature}=cut.v GROUP BY 2""")
deciles=[]
for q in parts:deciles.append(c.execute(q).df())
import pandas as pd
pd.concat(deciles).sort_values(['feature','band']).to_csv(OUT/'feature_deciles.csv',index=False,lineterminator='\n')
checks['feature_deciles']=sum(map(len,deciles))

cats=files(Path('outputs/propensity_data_v1/categories'),months,'30d.parquet')
c.read_parquet(cats,hive_partitioning=True).create_view('cat')
save('category_lift',"""WITH a AS (SELECT category,sum(event_count) event_exposures,
count(*) present_rows,sum(y_any) positives,avg(y_any) present_rate
FROM cat JOIN tr USING(client_id,month) WHERE attribute='event_type' AND category<>'__MISSING__'
GROUP BY 1), total AS (SELECT count(*) n,sum(y_any) p,avg(y_any) prevalence FROM tr)
SELECT a.*,present_rate*100 rate_pct,present_rate/prevalence lift,
(p-positives)::DOUBLE/(n-present_rows)*100 absent_rate_pct,
event_exposures::DOUBLE/sum(event_exposures) OVER()*100 event_share_pct
FROM a CROSS JOIN total ORDER BY event_exposures DESC""")

# Metadata hour can be derived from the timestamp. No duration/text length exists.
# Read only the normalized cache, and only events strictly inside each train window.
cache=str(ROOT/'outputs/propensity_processing_v3/work.duckdb').replace('\\','/')
c.execute(f"ATTACH '{cache}' AS cache (READ_ONLY)")
save('dialog_metadata',"""
WITH a AS (SELECT t.client_id,t.month,t.y_any,
count(d.event_time) n_dialog,
count(d.event_time) FILTER(WHERE extract(hour FROM d.event_time)<6) night_dialog,
count(d.event_time) FILTER(WHERE extract(isodow FROM d.event_time)>=6) weekend_dialog,
epoch(max(t.cutoff)-max(d.event_time))/86400.0 recency
FROM tr t LEFT JOIN cache.dialog d ON t.client_id=d.client_id
AND d.event_time<t.cutoff AND d.event_time>=t.cutoff-INTERVAL 30 DAY GROUP BY 1,2,3),
long AS (
SELECT y_any,'frequency' feature,CASE WHEN n_dialog=0 THEN '0' WHEN n_dialog=1 THEN '1' ELSE '2+' END band FROM a
UNION ALL SELECT y_any,'night_00_06',CASE WHEN n_dialog=0 THEN 'no_dialog' WHEN night_dialog>0 THEN 'yes' ELSE 'no' END FROM a
UNION ALL SELECT y_any,'weekend',CASE WHEN n_dialog=0 THEN 'no_dialog' WHEN weekend_dialog>0 THEN 'yes' ELSE 'no' END FROM a
UNION ALL SELECT y_any,'recency',CASE WHEN recency IS NULL THEN 'missing' WHEN recency<=7 THEN '0_7d' ELSE '7_30d' END FROM a)
SELECT feature,band,count(*) n,sum(y_any) positives,avg(y_any)*100 rate_pct,
avg(y_any)/(SELECT avg(y_any) FROM tr) lift FROM long GROUP BY 1,2 ORDER BY 1,2
""")

# A stable customer sample measures code transitions as a proxy, not physical distance.
# A one-to-many event join would multiply rows; count category exposures separately above.
save('geo_quality_proxy',"""
WITH e AS (SELECT g.*,lag(event_time) OVER w prior_time,lag(geohash_6) OVER w prior_code
FROM cache.geo g WHERE event_time<'2022-09-01' AND hash(client_id)%100=0
WINDOW w AS (PARTITION BY client_id ORDER BY event_time,source_file,source_row))
SELECT count(*) rows,count(DISTINCT client_id) clients,
count(*) FILTER(WHERE prior_time IS NOT NULL) pairs,
count(*) FILTER(WHERE prior_time=event_time AND prior_code<>geohash_6) simultaneous_code_conflicts,
count(*) FILTER(WHERE epoch(event_time-prior_time)>0 AND epoch(event_time-prior_time)<=60) pairs_within60s,
count(*) FILTER(WHERE epoch(event_time-prior_time)>0 AND epoch(event_time-prior_time)<=60 AND prior_code<>geohash_6) changes_within60s
FROM e
""")

save('leakage_audit',"""SELECT 'trx' source,count(*) total_before_sep,
count(*) FILTER(WHERE event_time<'2022-02-01') before_first_label,
min(event_time) first_event,max(event_time) last_event
FROM cache.trx WHERE event_time<'2022-09-01'""")

assert c.execute('SELECT count(*),sum(y_any) FROM tr').fetchone()==(701568,8672)
assert all(f.groupby('feature').n.sum().eq(701568).all() for f in deciles)
co=pd.read_csv(OUT/'target_cohort_drift.csv')
assert co.groupby('month').n.sum().eq(100224).all()
cat=pd.read_csv(OUT/'category_lift.csv')
assert abs(cat.event_share_pct.sum()-100)<1e-6
(OUT/'descriptive_manifest.json').write_text(json.dumps(dict(status='complete',duckdb_version=duckdb.__version__,scope='Train 02–08/2022; cohort drift 02–11/2022; no calibration/test features opened',
 timezone='UTC assumption, not publisher-confirmed',tie_policy='Equal feature values share midpoint-rank decile; 0=missing',
 geo_sample='hash(client_id)%100=0 using DuckDB; observed code transitions do not establish location errors',
 checks=checks),ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(checks)
