"""Validate the generated full-history report, cached aggregates and linked sample."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re

import duckdb
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def validate(output,sample,cache):
    checks=[]
    def check(name,ok,detail=None):
        checks.append(dict(check=name,passed=bool(ok),detail=detail))
        if not ok:raise AssertionError(name+': '+str(detail))
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    meta=json.loads((output/'metadata.json').read_text(encoding='utf-8'))
    frames={}
    for item in manifest['tables']:
        path=output/item['path']
        with path.open(encoding='utf-8-sig',newline='') as handle:
            reader=csv.reader(handle);columns=next(reader);rows=sum(1 for _ in reader)
        check('shape:'+item['path'],rows==item['rows'] and columns==item['columns'])
        check('aggregate_only:'+item['path'],'client_id' not in columns)
    for path in (output/'tables/review').glob('*.csv'):
        frames[path.stem]=pd.read_csv(path)
    expected={p.relative_to(output).as_posix() for p in (output/'tables').rglob('*.csv')}
    check('manifest_exact_tables',expected=={i['path'] for i in manifest['tables']})
    expected_figures={p.relative_to(output).as_posix() for p in (output/'figures').glob('*.png')}
    check('manifest_exact_figures',expected_figures==set(manifest['figures']))
    for relative in manifest['figures']:
        with Image.open(output/relative) as image:
            image.verify()
        check('png:'+relative,True)
    report=(output/'report.md').read_text(encoding='utf-8')
    for relative in re.findall(r'\]\(([^)]+)\)',report):
        if not relative.startswith(('https://','http://')):
            check('link:'+relative,(output/relative).exists())
    check('obsolete_log_chart_removed',not (output/'figures/amount_log_histogram.png').exists())
    schema=frames['source_schema'].drop_duplicates('source').set_index('source')
    counts={s:int(schema.loc[s,'rows']) for s in ['trx','geo','dialog']}
    check('amount_all_rows',int(frames['amount_full'].finite_n.iloc[0])==counts['trx'])
    check('amount_all_bins',int(frames['amount_raw_bins'].n.sum())==counts['trx'])
    check('amount_currency_total',int(frames['amount_by_currency'].finite_n.sum())==counts['trx'])
    check('amount_outlier_currency_total',int(frames['amount_outliers_currency'].finite_n.sum())==counts['trx'])
    for s,n in counts.items():
        check('monthly_count:'+s,int(frames['monthly_coverage'].query('source==@s').events.sum())==n)
        row=frames['events_per_customer'].query("source==@s and population=='all_customers'").iloc[0]
        check('customer_moments:'+s,np.isclose(row['mean']*row.rows,n))
    check('behavior_combinations_partition',int(frames['behavior_combinations'].customers.sum())==int(schema.loc['clients','rows']))
    zeros=frames['duplicates'].query("kind in ['full_row','unique_key']")
    check('exact_and_key_duplicates',zeros.duplicate_n.eq(0).all(),int(zeros.duplicate_n.sum()))
    missing=frames['null_target_context']
    check('diff_null_zero_context',missing.query("trans_count_group!='zero'").missing_diff.sum()==0)
    check('diff_null_all_zero',missing.query("trans_count_group=='zero'").missing_diff.sum()==missing.query("trans_count_group=='zero'").rows.sum())
    targets=frames['target_monthly'];base=pd.read_csv(output/'tables/descriptive.targets/target_prevalence.csv')
    for row in base.itertuples():
        check('target_total:'+row.target,int(targets.query('target==@row.target').positive_n.sum())==row.positive_n)
    selection=pd.read_parquet(sample/'selection.parquet')
    sample_ids=set(selection.client_id)
    check('sample_unique',len(sample_ids)==len(selection)==meta['review']['sample_customers'])
    sample_meta=json.loads((sample/'metadata.json').read_text(encoding='utf-8'))
    check('sample_fingerprint',sample_meta['source_fingerprint']==meta['review']['source_fingerprint'])
    allocation=frames['sampling_strata']
    check('allocation_total',int(allocation.sample_n.sum())==len(selection))
    check('allocation_no_oversample',allocation.sample_n.le(allocation.population_n).all())
    check('hamilton_error_at_most_one',(allocation.sample_n-allocation.expected_n).abs().le(1.00000001).all())
    grid=pd.read_parquet(sample/'customer_month.parquet')
    check('grid_unique',not grid.duplicated(['client_id','month']).any())
    check('grid_keeps_all_months',len(grid)==len(selection)*frames['monthly_coverage'].month.nunique())
    with duckdb.connect(str(cache/'review.duckdb'),read_only=True) as db:
        for item in sample_meta['tables']:
            frame=pd.read_parquet(sample/item['path'])
            check('sample_rows:'+item['source'],len(frame)==item['rows'])
            check('sample_links:'+item['source'],frame.client_id.isin(sample_ids).all())
            if item['source'] in counts:
                s=item['source']
                cache_counts=db.execute(f'SELECT client_id,sum(events)::BIGINT events FROM cache_month_{s} GROUP BY client_id').df().set_index('client_id').events
                actual=frame.groupby('client_id').size().reindex(selection.client_id,fill_value=0)
                expected=cache_counts.reindex(selection.client_id,fill_value=0)
                check('sample_preserves_full_history:'+s,np.array_equal(actual.to_numpy(),expected.to_numpy()))
                check('grid_counts:'+s,int(grid[s+'_raw_events'].sum())==len(frame))
                expected_null=(grid.month<'2022-01')&grid[s+'_raw_events'].eq(0) if s!='trx' else pd.Series(False,index=grid.index)
                check('grid_null_policy:'+s,grid[s+'_events'].isna().equals(expected_null))
    inventory=json.loads((cache/'source_inventory.json').read_text(encoding='utf-8'))
    digest=hashlib.sha256(json.dumps(inventory['files'],sort_keys=True).encode()).hexdigest()
    check('cache_inventory_fingerprint',digest==inventory['fingerprint']==meta['review']['source_fingerprint'])
    check('github_file_sizes',all(p.stat().st_size<100_000_000 for d in [output,sample] for p in d.rglob('*') if p.is_file()))
    result=dict(status='passed',created_at=datetime.now(timezone.utc).isoformat(),checks_passed=len(checks),
                tables=len(manifest['tables']),figures=len(manifest['figures']),population=int(schema.loc['clients','rows']),
                event_rows=counts,sample_customers=len(selection),source_fingerprint=digest,checks=checks)
    (ROOT/'docs/mbd_mini_eda_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'docs/mbd_mini_eda')
    parser.add_argument('--sample-dir',type=Path,default=ROOT/'sampledata/mbd_mini')
    parser.add_argument('--cache-dir',type=Path,default=ROOT/'outputs/mbd_review_cache')
    args=parser.parse_args();validate(args.output_dir,args.sample_dir,args.cache_dir)
