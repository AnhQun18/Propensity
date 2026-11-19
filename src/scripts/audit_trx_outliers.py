"""Descriptive full-history outlier audit; flags are not deletion rules."""
from pathlib import Path
import json
import hashlib
import duckdb
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trx_outliers_20261116'


def main():
    OUT.mkdir(exist_ok=True)
    files = sorted((ROOT / 'data/MBD-mini/detail/trx').rglob('*.parquet'))
    before = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    fingerprint = hashlib.sha256(json.dumps(before).encode()).hexdigest()
    c = duckdb.connect()
    c.execute("SET threads=1")
    c.execute("SET preserve_insertion_order=false")
    c.execute("SET memory_limit='2GB'")
    c.execute(f"SET temp_directory='{(OUT / 'tmp').as_posix()}'")
    c.read_parquet([p.as_posix() for p in files], hive_partitioning=False).create_view('raw')

    def export(name, sql):
        d = c.execute(sql).df()
        d.to_csv(OUT / f'{name}.csv', index=False)
        print(name, len(d), flush=True)
        return d

    # Robust reference within customer, currency and transaction subtype.
    print('Building customer behaviour references', flush=True)
    cache = OUT / 'buckets'
    cache_manifest = OUT / 'bucket_inventory.json'
    if cache.exists() and (not cache_manifest.exists() or
            json.loads(cache_manifest.read_text())['fingerprint'] != fingerprint):
        raise RuntimeError('Intermediate cache is incomplete or stale; choose a fresh output directory')
    if not cache.exists():
        c.execute(f"COPY (SELECT *,hash(client_id)%32 bucket FROM raw) TO '{cache.as_posix()}' (FORMAT PARQUET, PARTITION_BY(bucket))")
        cache_manifest.write_text(json.dumps(dict(fingerprint=fingerprint)),encoding='utf-8')
    ref_sql = '''SELECT client_id,currency,event_type,event_subtype,
        count(*) n,median(ln(amount::DOUBLE)) med,median(amount::DOUBLE) median_amount
        FROM part WHERE amount>0 AND isfinite(amount) GROUP BY ALL'''
    scale_sql = '''SELECT r.*,median(abs(ln(t.amount::DOUBLE)-r.med)) mad
        FROM part t JOIN local_ref r ON t.client_id=r.client_id
        AND t.currency IS NOT DISTINCT FROM r.currency
        AND t.event_type=r.event_type AND t.event_subtype=r.event_subtype
        WHERE r.n>=20 AND t.amount>0 AND isfinite(t.amount) GROUP BY ALL'''
    flag_sql = '''SELECT t.*,s.n reference_n,s.median_amount,
        t.amount::DOUBLE/s.median_amount ratio_to_median,
        abs(ln(t.amount::DOUBLE)-s.med)/greatest(1.4826*s.mad,ln(2)) robust_score
        FROM part t JOIN local_scales s ON t.client_id=s.client_id
        AND t.currency IS NOT DISTINCT FROM s.currency
        AND t.event_type=s.event_type AND t.event_subtype=s.event_subtype
        WHERE s.n>=20 AND t.amount>0 AND isfinite(t.amount)
        AND abs(ln(t.amount::DOUBLE)-s.med)>6*greatest(1.4826*s.mad,ln(2))'''
    repeat_sql = '''SELECT client_id,event_time,currency,amount,
        count(*) n,count(DISTINCT struct_pack(et:=event_type,es:=event_subtype,
        s11:=src_type11,s12:=src_type12,d11:=dst_type11,d12:=dst_type12,
        s21:=src_type21,s22:=src_type22,s31:=src_type31,s32:=src_type32)) variants
        FROM part GROUP BY ALL HAVING count(*)>1'''
    profile_sql = '''SELECT client_id,count(*) events,
        count(DISTINCT cast(event_time AS DATE)) active_days,
        sum(amount::DOUBLE) total_amount,median(amount::DOUBLE) median_amount,
        max(amount::DOUBLE) max_amount FROM part WHERE currency=11 GROUP BY client_id'''
    for i, folder in enumerate(sorted(cache.glob('bucket=*'))):
        c.read_parquet([p.as_posix() for p in folder.glob('*.parquet')],hive_partitioning=False).create_view('part',replace=True)
        c.execute('CREATE OR REPLACE TEMP TABLE local_ref AS '+ref_sql)
        c.execute('CREATE OR REPLACE TEMP TABLE local_scales AS '+scale_sql)
        for name, sql in [('ref','SELECT * FROM local_ref'),('flagged',flag_sql),('repeats',repeat_sql),('profiles',profile_sql)]:
            c.execute((f'CREATE TEMP TABLE {name} AS ' if i==0 else f'INSERT INTO {name} ')+sql)
        print(f'Completed customer bucket {i+1}/32',flush=True)
    export('within_customer_summary','''SELECT currency,count(*) flagged_events,
        count(DISTINCT client_id) clients FROM flagged GROUP BY currency''')
    export('within_customer_top500','SELECT * FROM flagged ORDER BY robust_score DESC LIMIT 500')
    coverage = export('reference_coverage','''SELECT sum(n) total_events,
        sum(n) FILTER(WHERE n>=20) eligible_events,count(*) reference_groups,
        count(*) FILTER(WHERE n>=20) eligible_groups FROM ref''')
    assert int(coverage.iloc[0]['total_events']) == c.execute('SELECT count(*) FROM raw').fetchall()[0][0]

    # Compare customers on the same currency; all currency-11 customers are kept.
    customers = export('customer_profiles','SELECT * FROM profiles')
    assert customers.client_id.is_unique
    cols = ['events','active_days','total_amount','median_amount','max_amount']
    x = RobustScaler().fit_transform(np.log1p(customers[cols]))
    customers['cluster'] = KMeans(n_clusters=5,n_init=10,random_state=20261116).fit_predict(x)
    customers.to_csv(OUT / 'customer_clusters_currency11.csv',index=False)
    clusters = customers.groupby('cluster').agg(clients=('client_id','size'),
        events=('events','sum'),median_events=('events','median'),
        median_total_amount=('total_amount','median'),median_ticket=('median_amount','median'))
    clusters.to_csv(OUT / 'cluster_summary_currency11.csv')
    print(clusters.to_string(),flush=True)

    print('Checking timestamp repetitions and power-of-ten signatures',flush=True)
    export('same_time_amount_summary','''SELECT currency,count(*) groups,sum(n) involved_rows,
        sum(n-1) extra_rows,sum(n-1) FILTER(WHERE variants=1) identical_extra_rows,
        count(*) FILTER(WHERE variants>1) different_metadata_groups FROM repeats GROUP BY currency''')
    export('same_time_amount_top500','SELECT * FROM repeats ORDER BY n DESC LIMIT 500')
    # 0.01% relative tolerance accommodates float32 representation; not proof of error.
    export('power10_by_currency','''SELECT currency,count(*) total,
        count(*) FILTER(WHERE abs(log10(amount::DOUBLE)-round(log10(amount::DOUBLE)))<0.00004343) near_power10,
        min(amount) min_amount,max(amount) max_amount FROM raw GROUP BY currency''')
    export('within_customer_power10_ratios','''SELECT currency,
        round(log10(ratio_to_median)) exponent,count(*) events,count(DISTINCT client_id) clients
        FROM flagged WHERE abs(log10(ratio_to_median)-round(log10(ratio_to_median)))<0.00004343
        GROUP BY ALL ORDER BY events DESC''')
    export('schema_value_checks','''SELECT count(*) rows,
        count(*) FILTER(WHERE amount IS NULL) missing_amount,
        count(*) FILTER(WHERE NOT isfinite(amount)) nonfinite_amount,
        count(*) FILTER(WHERE amount<=0) nonpositive_amount,
        count(*) FILTER(WHERE client_id IS NULL) missing_client,
        count(*) FILTER(WHERE event_time IS NULL) missing_time,
        min(event_time) first_event,max(event_time) last_event FROM raw''')
    assert before == [(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in files]
    (OUT / 'methodology.json').write_text(json.dumps(dict(
        status='ready',files=len(files),raw_modified=False,
        within_customer='client/currency/event_type/subtype; n>=20; absolute log deviation >6*max(1.4826*MAD,ln(2)); both tails',
        clustering='currency11 only; KMeans k=5 on RobustScaler(log1p(events,active_days,total,median,max)); exploratory, not validated segments',
        power10='amount or within-customer ratio within about 0.01% of 10^k; signature only, not error proof',
        limitations='Full-history descriptive references include future observations. Fit train-only or past-only references for modeling. Float32 anonymous amounts; no monetary unit mapping. Typed schema blocks textual contamination but numeric contamination needs source reconciliation.',
        coverage=coverage.to_dict(orient='records')),indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
