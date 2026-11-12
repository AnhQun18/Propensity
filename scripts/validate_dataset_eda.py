"""Independent cross-table and temporal validation of the new EDA package."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import duckdb
from analyze_dataset_eda import safe_rate,wilson

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables/mbd-mini-dataset-eda'
T=OUT/'tables'
checks={}
def read(name):return pd.read_csv(T/(name+'.csv'))
def check(name,condition):
    assert condition,name
    checks[name]=True

def main():
    check('zero_denominator_is_unknown',safe_rate(0,0) is None and np.isnan(wilson(0,0)[0]))
    check('observed_zero_stays_zero',safe_rate(0,10)==0 and wilson(0,10)[1]>0)
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    n=manifest['checks']['clients'];rows=n*12
    monthly=read('target_monthly');rates=read('rates_cluster_ci');frequency=read('target_frequency')
    check('monthly_labels_complete',len(monthly)==60 and monthly.n_total.eq(n).all() and monthly.n_null.eq(0).all())
    check('rate_denominators',(rates.n_positive+rates.n_negative).eq(rates.n_total).all())
    check('rate_points',np.allclose(rates.rate,rates.n_positive/rates.n_total))
    estimated=rates[(rates.n_positive>0)&rates.valid_replicates.eq(2000)]
    check('rate_ci_bounds',estimated.ci_low.between(0,1).all() and estimated.ci_high.between(0,1).all() and estimated.ci_low.le(estimated.ci_high).all())
    if 'ci_status' in rates:
        zero=rates[rates.n_positive==0]
        check('zero_positive_interval_not_overstated',zero.ci_low.isna().all() and zero.ci_high.isna().all())
    check('bootstrap_contract',rates.replicates.eq(2000).all() and rates.seed.eq(42).all() and rates.valid_replicates.between(1,2000).all())
    if 'ci_status' in rates:
        unstable=rates[rates.valid_replicates<2000]
        check('unstable_bootstrap_flagged',unstable.ci_low.isna().all() and unstable.ci_high.isna().all())
    check('small_sample_flags',rates.small_positive.eq(rates.n_positive.lt(30)).all())
    for target in monthly.target.unique():
        m=monthly[monthly.target==target];p=rates[(rates.kind=='prevalence')&(rates.target==target)].iloc[0]
        f=frequency[frequency.target==target]
        check(target+'_positive_reconciled',int(m.n_positive.sum())==int(p.n_positive)==int((f.positive_months*f.customers).sum()))
        check(target+'_frequency_population',f.customers.sum()==n)
    co=read('buyer_composition_monthly')
    grouped=co.groupby(['target','month'])
    check('buyer_composition_totals',np.allclose(grouped.share.sum(),1) and (grouped.buyers.sum()==grouped.all_current_buyers.first()).all())
    check('first_month_is_unknown',co[(co.month=='2022-02-01')&(co.buyer_group=='history_unknown')].share.eq(1).all())
    check('later_unknown_zero',co[(co.month!='2022-02-01')&(co.buyer_group=='history_unknown')].buyers.eq(0).all())
    transitions=read('target_transitions_monthly')
    check('transition_pairs_complete',transitions.groupby(['target','month']).n_total.sum().eq(n).all() and len(transitions)==110)
    bins=read('feature_group_definitions')
    check('feature_groups_cover_all_rows',bins.groupby('feature').rows.sum().eq(rows).all())
    overlap=read('cross_modal_full_history').set_index('mask')
    check('overlap_population',overlap.clients.sum()==n and overlap.loc[0,'clients']==468 and overlap.loc[7,'clients']==35424)
    audit=read('temporal_event_audit')
    check('cutoff_all_sources',audit.boundary_check.all() and (pd.to_datetime(audit.max_used_event)<pd.to_datetime(audit.cutoff)).all())
    c=duckdb.connect(':memory:')
    cache_path=(ROOT/'outputs/dataset_eda/analysis.duckdb').as_posix().replace("'","''")
    c.execute(f"ATTACH '{cache_path}' AS cache (READ_ONLY)")
    snapshot=(ROOT/'outputs/dataset_eda/customer_month_snapshots.parquet').as_posix()
    c.read_parquet(snapshot).create_view('snap')
    check('snapshot_population',c.sql('select count(*) from snap').fetchall()[0][0]==rows)
    check('snapshot_keys_unique',c.sql('select count(*)-count(distinct (cid,outcome_month)) from snap').fetchall()[0][0]==0)
    check('binary_label_domain',c.sql('select count(*) from snap where target_1 not in (0,1) or target_2 not in (0,1) or target_3 not in (0,1) or target_4 not in (0,1) or y_any not in (0,1)').fetchall()[0][0]==0)
    for source in ['trx','dialog','geo']:
        # Re-derive recency by a relational join against all monthly last timestamps.
        error=c.sql(f"""WITH past AS (SELECT s.cid,s.outcome_month,max(e.last_time) last_event
           FROM (SELECT DISTINCT cid,client_id,outcome_month FROM snap) s
           LEFT JOIN cache.eda_{source}_month e ON e.client_id=s.client_id AND e.event_month<s.outcome_month GROUP BY 1,2)
           SELECT count(*) FROM snap s JOIN past p USING(cid,outcome_month)
           WHERE (s.{source}_recency_days IS NULL)<>(p.last_event IS NULL)
           OR abs(s.{source}_recency_days-epoch(s.outcome_month-p.last_event)/86400.)>0.000001""").fetchall()[0][0]
        check(source+'_recency_full_history',error==0)
        mismatch=c.sql(f"""SELECT count(*) FROM snap s LEFT JOIN cache.eda_{source}_month e ON s.client_id=e.client_id
          AND e.event_month=s.outcome_month-INTERVAL '1 month'
          WHERE s.{source}_events<>coalesce(e.events,0)""").fetchall()[0][0]
        check(source+'_calendar_month_counts',mismatch==0)
        q=audit[audit.source==source]
        check(source+'_snapshot_vs_audit',int(c.sql(f'SELECT sum({source}_events) FROM snap').fetchall()[0][0])==int(q.window_events.sum()))
    check('geo_transition_rate_bounds',c.sql('select count(*) from snap where geo_transition_rate<0 or geo_transition_rate>1').fetchall()[0][0]==0)
    check('geo_no_pair_is_unknown',c.sql('select count(*) from snap where geo_valid_pairs=0 and geo_transition_rate is not null').fetchall()[0][0]==0)
    for source in ['trx','dialog']:
        check(source+'_singleton_gap_unknown',c.sql(f'select count(*) from snap where {source}_events<2 and {source}_gap_days is not null').fetchall()[0][0]==0)
    check('zero_count_vs_unknown_recency',c.sql('select count(*) from snap where trx_events=0 and trx_recency_days is null').fetchall()[0][0]>0)
    c.close()
    result=dict(pass_all=True,checks=checks,total_checks=len(checks))
    (OUT/'data_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
