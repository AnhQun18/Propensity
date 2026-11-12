"""Reproducible descriptive MBD-mini analysis for the 40-slide HTML deck.

Raw files and existing caches are read only. Individual records stay in outputs/.
Only aggregate tables are published in deliverables/mbd-mini-dataset-eda/tables/.
"""
from pathlib import Path
import hashlib
import json
import time
import shutil
import sys
import numpy as np
import pandas as pd
import duckdb
import pyarrow.parquet as pq
from scipy import sparse
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/MBD-mini'
OUT = ROOT / 'deliverables/mbd-mini-dataset-eda'
WORK = ROOT / 'outputs/dataset_eda'
TABLES = OUT / 'tables'
TARGETS = ['target_1', 'target_2', 'target_3', 'target_4', 'y_any']
SEED, REPS = 42, 2000


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str,
                               allow_nan=False) + '\n', encoding='utf-8')


def wilson(k, n):
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    z = norm.ppf(.975)
    d = 1 + z*z / np.maximum(n, 1)
    center = (p + z*z/(2*np.maximum(n, 1))) / d
    half = z * np.sqrt(p*(1-p)/np.maximum(n, 1) + z*z/(4*np.maximum(n, 1)**2))/d
    return np.where(n > 0, np.maximum(0, center-half), np.nan), np.where(n > 0, np.minimum(1, center+half), np.nan)


def safe_rate(k, n):
    return float(k / n) if n else None


class Analysis:
    def __init__(self):
        TABLES.mkdir(parents=True, exist_ok=True)
        WORK.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.c = duckdb.connect(str(WORK / 'analysis.duckdb'))
        self.c.execute("SET memory_limit='1200MB'")
        self.c.execute('SET threads=2')
        self.c.execute('SET preserve_insertion_order=false')
        self.c.execute('SET temp_directory=?', [str(WORK / 'spill')])
        self.inventory = []
        self.schema = []
        for name, rel in {'clients':'client_split','targets':'targets','trx':'detail/trx',
                          'geo':'detail/geo','dialog':'detail/dialog', 'ptls_trx':'ptls/trx',
                          'ptls_geo':'ptls/geo','ptls_dialog':'ptls/dialog'}.items():
            files = sorted((DATA/rel).rglob('*.parquet'))
            for p in files:
                pf = pq.ParquetFile(p)
                stat = p.stat()
                self.inventory.append(dict(source=name,path=p.relative_to(DATA).as_posix(),
                    bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,rows=pf.metadata.num_rows,
                    schema=str(pf.schema_arrow.remove_metadata())))
            arrow = pq.ParquetFile(files[0]).schema_arrow
            for col in arrow:
                self.schema.append(dict(source=name,column=col.name,dtype=str(col.type)))
            self.schema.append(dict(source=name,column='fold',dtype='Hive partition integer'))
            self.c.read_parquet([p.as_posix() for p in files],hive_partitioning=True).create_view('raw_'+name, replace=True)
        self.fingerprint = hashlib.sha256(json.dumps(self.inventory,sort_keys=True).encode()).hexdigest()
        old = duckdb.connect(str(ROOT/'outputs/mbd_review_cache/review.duckdb'), read_only=True)
        assert old.execute("SELECT value FROM cache_meta WHERE key='source_fingerprint'").fetchall()[0][0] == self.fingerprint, 'Existing EDA input fingerprint changed'
        old.close()
        self.c.execute('CREATE TABLE IF NOT EXISTS run_meta (fingerprint VARCHAR)')
        saved_meta = self.c.execute('SELECT * FROM run_meta').fetchall()
        prev = saved_meta[0] if saved_meta else None
        if prev:
            assert prev[0] == self.fingerprint, 'Use a fresh work directory for changed inputs'
        else:
            self.c.execute('INSERT INTO run_meta VALUES (?)',[self.fingerprint])
        self.published = {}
        self.contributions = []
        self.metric_records = []
        self.checks = {}

    def save(self, name, frame):
        frame = pd.DataFrame(frame)
        path = TABLES / (name+'.csv')
        frame.to_csv(path,index=False,lineterminator='\n')
        self.published[name] = dict(rows=len(frame),columns=list(frame.columns),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return frame

    def query(self,name,sql):
        table='eda_'+name
        if not self.c.execute('SELECT count(*) FROM information_schema.tables WHERE table_name=?',[table]).fetchall()[0][0]:
            print('Compute:',name,flush=True)
            if name in ['trx_gap','dialog_gap','geo_spatial','geo_moves']:
                source='geo' if name.startswith('geo') else name.split('_')[0]
                partial=[]
                for fold in range(5):
                    child=table+'_fold'+str(fold)
                    exists=self.c.execute('SELECT count(*) FROM information_schema.tables WHERE table_name=?',[child]).fetchall()[0][0]
                    if not exists:
                        print('  fold',fold,flush=True)
                        folded=sql.replace('FROM raw_'+source, f'FROM (SELECT * FROM raw_{source} WHERE fold={fold})')
                        self.c.execute(f'CREATE TABLE {child} AS {folded}')
                    partial.append('SELECT * FROM '+child)
                self.c.execute(f'CREATE TABLE {table} AS '+ ' UNION ALL '.join(partial))
            else:
                self.c.execute(f'CREATE TABLE {table} AS {sql}')
        else:
            print('Reuse:',name,flush=True)
        return self.c.execute(f'SELECT * FROM {table}').df()

    def add_metric(self, kind, name, target, ids, y, group='All'):
        ids=np.asarray(ids,dtype=np.int32)
        y=np.asarray(y,dtype=np.float32)
        counts=sparse.coo_matrix((np.ones(len(ids),dtype=np.float32),(ids,np.zeros(len(ids),dtype=int))),shape=(self.nclients,1)).tocsr()
        positives=sparse.coo_matrix((y,(ids,np.zeros(len(ids),dtype=int))),shape=(self.nclients,1)).tocsr()
        self.contributions.extend([counts,positives])
        n,k=len(y),int(y.sum())
        record=dict(metric_id=len(self.metric_records),kind=kind,metric=name,target=target,group=str(group),
                    n_total=n,n_positive=k,n_negative=n-k,rate=safe_rate(k,n),small_positive=k<30)
        self.metric_records.append(record)
        return record['metric_id']

    def labels(self):
        self.clients=self.c.sql('SELECT client_id,fold FROM raw_clients ORDER BY client_id').df()
        self.nclients=len(self.clients)
        self.clients['cid']=np.arange(self.nclients)
        self.c.register('client_index',self.clients)
        self.lab=self.c.sql("""SELECT i.cid,t.*,date_trunc('month',cast(mon AS DATE)) AS outcome_month,
          CASE WHEN target_1=1 OR target_2=1 OR target_3=1 OR target_4=1 THEN 1
          WHEN target_1=0 AND target_2=0 AND target_3=0 AND target_4=0 THEN 0 END AS y_any
          FROM raw_targets t JOIN client_index i USING(client_id) ORDER BY i.cid,mon"""
        ).df()
        self.lab['outcome_month']=pd.to_datetime(self.lab.outcome_month)
        assert self.lab.shape[0]==1202688 and self.nclients==100224
        assert not self.lab.duplicated(['client_id','mon']).any()
        assert self.lab[TARGETS].isna().sum().sum()==0
        assert self.lab[TARGETS].isin([0,1]).all().all()
        assert self.lab.groupby('cid').size().eq(12).all()
        assert self.lab.y_any.sum()==12854
        self.checks.update(label_rows=len(self.lab),clients=self.nclients,label_nulls=0,
                           label_duplicate_keys=0,y_any_positive=12854)
        schema=[]
        for col in self.lab.columns:
            if col in ['cid','outcome_month']:continue
            schema.append(dict(column=col,dtype=str(self.lab[col].dtype),null_count=int(self.lab[col].isna().sum()),
                null_rate=float(self.lab[col].isna().mean()),unique_count=int(self.lab[col].nunique())))
        self.save('target_schema',schema)
        self.save('target_null_patterns',[dict(pattern='0000 (0 = observed)',rows=len(self.lab),rate=1.)])
        monthly=[];frequencies=[];trans=[];composition=[];pairmetrics=[]
        for target in TARGETS:
            self.add_metric('prevalence',target,target,self.lab.cid,self.lab[target])
            mat=self.lab[target].to_numpy().reshape(self.nclients,12)
            for j, month in enumerate(sorted(self.lab.outcome_month.unique())):
                k=int(mat[:,j].sum());lo,hi=wilson(k,self.nclients)
                monthly.append(dict(target=target,month=str(month)[:10],n_total=self.nclients,
                    n_positive=k,n_negative=self.nclients-k,n_null=0,rate=k/self.nclients,
                    ci_low=float(lo),ci_high=float(hi),small_positive=k<30))
                prior=(mat[:,:j]==1).any(axis=1) if j else np.zeros(self.nclients,dtype=bool)
                buyer=mat[:,j]==1
                cats={'returning':buyer & prior,'first_observed':buyer & ~prior if j else np.zeros_like(buyer),
                      'history_unknown':np.zeros_like(buyer) if j else buyer}
                assert sum(int(mask.sum()) for mask in cats.values())==k
                for category,mask in cats.items():
                    clo,chi=wilson(int(mask.sum()),k)
                    composition.append(dict(target=target,month=str(month)[:10],buyer_group=category,
                        buyers=int(mask.sum()),all_current_buyers=k,share=safe_rate(mask.sum(),k),
                        ci_low=float(clo),ci_high=float(chi),small_positive=int(mask.sum())<30))
                if j:
                    for p in [0,1]:
                        mask=mat[:,j-1]==p
                        nk=int(mask.sum());pk=int(mat[mask,j].sum());tl,th=wilson(pk,nk)
                        trans.append(dict(target=target,month=str(month)[:10],previous=p,n_total=nk,
                            n_positive=pk,n_negative=nk-pk,rate=pk/nk,ci_low=float(tl),ci_high=float(th),small_positive=pk<30))
            for count,n in zip(*np.unique(mat.sum(axis=1),return_counts=True)):
                frequencies.append(dict(target=target,positive_months=int(count),customers=int(n)))
            prior=mat[:,:-1].ravel();cur=mat[:,1:].ravel();ids=np.repeat(np.arange(self.nclients),11)
            for p in [0,1]:
                self.add_metric('transition','previous_'+str(p),target,ids[prior==p],cur[prior==p])
            priorpositive=np.cumsum(mat,axis=1)-mat>0
            for label,mask in [('returning',priorpositive),('first_observed',~priorpositive & (np.arange(12)>0)),
                               ('history_unknown',np.broadcast_to(np.arange(12)==0,mat.shape))]:
                buyer=mat.ravel()==1
                self.add_metric('buyer_composition',label,target,np.repeat(np.arange(self.nclients),12)[buyer],mask.ravel()[buyer])
        self.save('target_monthly',monthly);self.save('target_frequency',frequencies)
        self.save('target_transitions_monthly',trans);self.save('buyer_composition_monthly',composition)
        self.save('raw_schema',self.schema)

    def events(self):
        monthly=[];summary=[];quality=[];alignment=[]
        for source in ['trx','dialog','geo']:
            # Scan only needed columns. Frequencies include every original record.
            extra=",sum(CASE WHEN currency=11 THEN amount ELSE 0 END) amount_currency11,sum(CASE WHEN currency=11 THEN 1 ELSE 0 END) currency11_events" if source=='trx' else ''
            frame=self.query(source+'_month',f"""SELECT client_id,date_trunc('month',event_time) event_month,
             count(*) events,count(distinct cast(event_time AS DATE)) active_days,min(event_time) first_time,max(event_time) last_time
             {extra} FROM raw_{source} GROUP BY 1,2""")
            frame['event_month']=pd.to_datetime(frame.event_month)
            frame=frame.merge(self.clients[['client_id','cid']],on='client_id',validate='many_to_one')
            self.checks[source+'_rows']=int(frame.events.sum())
            overview=self.c.execute(f'SELECT min(event_time),max(event_time),count(*) FROM raw_{source}').fetchall()[0]
            summary.append(dict(source=source,rows=int(overview[2]),clients=frame.cid.nunique(),
                first_event=str(overview[0]),last_event=str(overview[1])))
            for month,g in frame.groupby('event_month'):
                monthly.append(dict(source=source,month=str(month)[:10],events=int(g.events.sum()),
                    active_clients=g.cid.nunique(),events_per_active_client=float(g.events.mean()),
                    median=float(g.events.median()),p95=float(g.events.quantile(.95)),
                    log_coverage=g.cid.nunique()/self.nclients))
            # Full-calendar T-1 counts + cumulative recency; no 30-day approximation.
            snap=frame.copy();snap['outcome_month']=snap.event_month+pd.offsets.MonthBegin(1)
            cols=['cid','outcome_month','events','active_days']+(['amount_currency11','currency11_events'] if source=='trx' else [])
            self.lab=self.lab.merge(snap[cols].rename(columns={x:source+'_'+x for x in cols if x not in ['cid','outcome_month']}),
                on=['cid','outcome_month'],how='left',validate='one_to_one')
            for col in cols[2:]:self.lab[source+'_'+col]=self.lab[source+'_'+col].fillna(0)
            history=np.full(self.nclients,np.datetime64('NaT','ns'),dtype='datetime64[ns]')
            firstcut=pd.Timestamp('2022-02-01')
            earlier=frame[frame.event_month<firstcut-pd.offsets.MonthBegin(1)].groupby('cid').last_time.max()
            history[earlier.index]=earlier.to_numpy(dtype='datetime64[ns]')
            recency=np.full((self.nclients,12),np.nan)
            for j,month in enumerate(sorted(self.lab.outcome_month.unique())):
                prev=pd.Timestamp(month)-pd.offsets.MonthBegin(1)
                up=frame[frame.event_month==prev]
                assert up.empty or (up.last_time<pd.Timestamp(month)).all()
                alignment.append(dict(source=source,outcome_month=str(month)[:10],window_start=str(prev)[:10],
                    cutoff=str(month)[:10],window_events=int(up.events.sum()),
                    cumulative_events=int(frame.loc[frame.event_month<pd.Timestamp(month),'events'].sum()),
                    excluded_at_or_after_cutoff=int(frame.loc[frame.event_month>=pd.Timestamp(month),'events'].sum()),
                    max_used_event=str(up.last_time.max()),boundary_check=True))
                history[up.cid]=up.last_time.to_numpy(dtype='datetime64[ns]')
                recency[:,j]=(np.datetime64(month,'ns')-history)/np.timedelta64(1,'D')
            self.lab[source+'_recency_days']=recency.ravel()
            # Full-history per-customer distributions preserve the zero-log population.
            counts=frame.groupby('cid').events.sum().reindex(range(self.nclients),fill_value=0).to_numpy()
            qs=np.quantile(counts,[0,.25,.5,.75,.9,.95,.99,1])
            quality.append(dict(source=source,clients=self.nclients,zero_log=int((counts==0).sum()),
                mean=float(counts.mean()),minimum=qs[0],p25=qs[1],median=qs[2],p75=qs[3],p90=qs[4],p95=qs[5],p99=qs[6],maximum=qs[7],
                top1_share=float(np.sort(counts)[-int(np.ceil(self.nclients*.01)):].sum()/counts.sum()),
                top5_share=float(np.sort(counts)[-int(np.ceil(self.nclients*.05)):].sum()/counts.sum())))
        self.save('source_overview',summary);self.save('source_monthly',monthly);self.save('customer_activity',quality)
        assert self.checks['trx_rows']==38764930 and self.checks['geo_rows']==66295724 and self.checks['dialog_rows']==217205
        self.save('temporal_event_audit',alignment)
        self.save('temporal_alignment',[dict(outcome_month=str(m)[:10],cutoff=str(m)[:10]+' 00:00:00 UTC',
            window_start=str(pd.Timestamp(m)-pd.offsets.MonthBegin(1))[:10],window_end_exclusive=str(m)[:10],
            timezone_status='UTC research convention; original label ETL timezone unconfirmed') for m in sorted(self.lab.outcome_month.unique())])
        # Auxiliary features: exact monthly gaps and region-code entropy (not distances).
        for source in ['trx','dialog']:
            gap=self.query(source+'_gap',f"""WITH e AS (SELECT client_id,
               date_trunc('month',event_time)+INTERVAL '1 month' outcome_month,
               list_sort(list(event_time)) times FROM raw_{source} GROUP BY 1,2 HAVING count(*)>1)
             SELECT client_id,outcome_month,
              list_median(list_transform(range(2,len(times)+1),lambda i: epoch(times[i]-times[i-1])/86400.)) median_gap_days,
              len(times)-1 gap_pairs FROM e""")
            self.lab=self.lab.merge(gap.rename(columns={'median_gap_days':source+'_gap_days','gap_pairs':source+'_gap_pairs'}),on=['client_id','outcome_month'],how='left',validate='one_to_one')
        geo=self.query('geo_spatial',"""WITH r AS (SELECT client_id,date_trunc('month',event_time)+INTERVAL '1 month' outcome_month,
             geohash_6,count(*) n FROM raw_geo GROUP BY 1,2,3), p AS (SELECT *,sum(n) OVER(PARTITION BY client_id,outcome_month) total FROM r)
             SELECT client_id,outcome_month,count(*) unique_regions,max(n)/max(total)::DOUBLE dominant_share,
             -sum(n/total::DOUBLE*ln(n/total::DOUBLE)) entropy FROM p GROUP BY 1,2""")
        self.lab=self.lab.merge(geo.rename(columns={x:'geo_'+x for x in ['unique_regions','dominant_share','entropy']}),on=['client_id','outcome_month'],how='left',validate='one_to_one')
        # A timestamp with conflicting codes is ambiguous; do not invent their order.
        moves=self.query('geo_moves',"""WITH t AS (SELECT client_id,event_time,
          CASE WHEN min(geohash_6)=max(geohash_6) THEN min(geohash_6) END region
          FROM raw_geo GROUP BY 1,2), l AS (
          SELECT client_id,date_trunc('month',event_time)+INTERVAL '1 month' outcome_month,
          list_sort(list(struct_pack(t:=event_time,r:=region))) events,
          count(*) FILTER(WHERE region IS NULL) ambiguous_timestamps FROM t GROUP BY 1,2)
          SELECT client_id,outcome_month,
          coalesce(list_sum(list_transform(range(2,len(events)+1),lambda i: CASE WHEN events[i].r IS NOT NULL AND events[i-1].r IS NOT NULL THEN 1 ELSE 0 END)),0) valid_pairs,
          coalesce(list_sum(list_transform(range(2,len(events)+1),lambda i: CASE WHEN events[i].r<>events[i-1].r THEN 1 ELSE 0 END)),0) transitions,
          ambiguous_timestamps FROM l""")
        self.lab=self.lab.merge(moves.rename(columns={x:'geo_'+x for x in ['valid_pairs','transitions','ambiguous_timestamps']}),on=['client_id','outcome_month'],how='left',validate='one_to_one')
        self.lab['geo_transition_rate']=self.lab.geo_transitions/self.lab.geo_valid_pairs.replace(0,np.nan)
        assert np.array_equal(self.lab.cid.to_numpy(),np.repeat(np.arange(self.nclients),12))
        self.lab.to_parquet(WORK/'customer_month_snapshots.parquet',index=False)
        self.checks['snapshot_rows']=len(self.lab)
        # Embeddings: vector norm and zero/invalid checks, without fitting a classifier.
        emb=self.query('embedding',"""WITH e AS (SELECT client_id,event_time,len(embedding) dimension,
          list_sum(list_transform(embedding,lambda x: CASE WHEN x IS NULL OR NOT isfinite(x) THEN 1 ELSE 0 END)) bad_elements,
          sqrt(list_sum(list_transform(embedding,lambda x: cast(x AS DOUBLE)*x))) norm FROM raw_dialog)
          SELECT count(*) rows,count(*) FILTER(WHERE dimension IS NULL) null_vectors,
          count(*) FILTER(WHERE dimension<>768) wrong_dimension,
          sum(bad_elements) bad_elements,count(*) FILTER(WHERE norm=0) zero_vectors,
          min(norm) minimum,quantile_cont(norm,.25) p25,median(norm) median,quantile_cont(norm,.75) p75,max(norm) maximum FROM e""")
        self.save('embedding_quality',emb)
        # Category choices come from event popularity, independently of labels.
        cat=self.query('trx_categories',"SELECT event_type,count(*) events,count(distinct client_id) clients FROM raw_trx GROUP BY 1 ORDER BY events DESC")
        self.save('trx_categories',cat)
        top=list(cat.event_type.head(5).astype(int))
        presence=self.query('trx_category_presence',f"""SELECT DISTINCT client_id,date_trunc('month',event_time)+INTERVAL '1 month' outcome_month,event_type
              FROM raw_trx WHERE event_type IN ({','.join(map(str,top))})""")
        for code in top:
            selected=self.lab[['cid','client_id','outcome_month']].merge(presence[presence.event_type==code],on=['client_id','outcome_month'],how='left')
            self.lab['category_'+str(code)]=selected.event_type.notna().to_numpy()
        self.lab.to_parquet(WORK/'customer_month_snapshots.parquet',index=False)
        self.source_summary=pd.DataFrame(summary)

    def comparisons(self):
        features={
          'trx_events':('TRX count T-1',[0,5,20,100]),
          'trx_recency_days':('TRX recency (days)',[3,10,30,90]),
          'trx_amount_currency11':('TRX amount code 11 T-1',[0,10000,100000,1000000]),
          'dialog_events':('DIALOG count T-1',[0,1,2,5]),
          'dialog_recency_days':('DIALOG recency (days)',[7,30,90,180]),
          'geo_unique_regions':('GEO distinct code 6 T-1',[1,5,20,100]),
          'geo_dominant_share':('GEO dominant share T-1',[.25,.5,.75,.95]),
          'geo_entropy':('GEO entropy T-1 (nats)',[.5,1,2,3]),
          'geo_transition_rate':('GEO code-transition rate T-1',[.1,.25,.5,.75]),
          'trx_gap_days':('TRX median within-month gap (days)',[.01,.1,1,7]),
          'dialog_gap_days':('DIALOG median within-month gap (days)',[1,7,14,30]),
        }
        summaries=[];bins=[]
        for col,(name,cuts) in features.items():
            values=self.lab[col].to_numpy(dtype=float)
            codes=np.searchsorted(cuts,values,side='left');codes[~np.isfinite(values)]=-1
            for b in sorted(set(codes)):
                mask=codes==b
                group='Unknown / no usable event' if b==-1 else ('<= '+str(cuts[b]) if b==0 else ('> '+str(cuts[-1]) if b==len(cuts) else '('+str(cuts[b-1])+', '+str(cuts[b])+']'))
                bins.append(dict(feature=col,group=group,rows=int(mask.sum()),lower=None if b<=0 else cuts[b-1],upper=None if b<0 or b==len(cuts) else cuts[b]))
                for target in TARGETS:
                    self.add_metric('feature',col,target,self.lab.loc[mask,'cid'],self.lab.loc[mask,target],group)
            for target in TARGETS:
                for y,g in self.lab.groupby(target):
                    vals=g[col].dropna()
                    summaries.append(dict(feature=col,target=target,label=int(y),n_total=len(g),n_observed=len(vals),
                        n_missing=int(g[col].isna().sum()),median=float(vals.median()),p25=float(vals.quantile(.25)),
                        p75=float(vals.quantile(.75)),p95=float(vals.quantile(.95))))
        # Same observation-density strata prevent conflating volume and region diversity.
        for lower,upper in [(1,10),(11,100),(101,1000),(1001,float('inf'))]:
            mask=self.lab.geo_events.between(lower,upper)
            for y,g in self.lab[mask].groupby('y_any'):
                summaries.append(dict(feature='geo_unique_regions_exposure_'+str(lower),target='y_any',label=int(y),
                    n_total=len(g),n_observed=len(g),n_missing=0,median=float(g.geo_unique_regions.median()),
                    p25=float(g.geo_unique_regions.quantile(.25)),p75=float(g.geo_unique_regions.quantile(.75)),p95=float(g.geo_unique_regions.quantile(.95))))
        for col in [x for x in self.lab if x.startswith('category_')]:
            mask=self.lab[col]
            for target in TARGETS:self.add_metric('category',col,target,self.lab.loc[mask,'cid'],self.lab.loc[mask,target],'present in T-1')
        self.save('feature_group_definitions',bins)
        self.save('feature_target_summary',summaries)
        # Coverage patterns use disjoint groups including the no-source population.
        masks=sum((self.lab[s+'_events']>0).astype(int)*(1<<i) for i,s in enumerate(['trx','dialog','geo']))
        temp=self.lab[['outcome_month','cid']].assign(mask=masks)
        self.save('cross_modal_monthly',temp.groupby(['outcome_month','mask']).size().reset_index(name='customer_months'))
        full=self.lab.groupby('cid')[[s+'_events' for s in ['trx','dialog','geo']]].max()
        # For full history, monthly aggregation files also include pre-label events.
        fullmask=np.zeros(self.nclients,dtype=np.int8)
        for i,s in enumerate(['trx','dialog','geo']):
            active=self.c.execute(f'SELECT DISTINCT i.cid FROM eda_{s}_month e JOIN client_index i USING(client_id)').fetchnumpy()['cid']
            fullmask[active] |= 1<<i
        self.save('cross_modal_full_history',pd.Series(fullmask).value_counts().sort_index().rename_axis('mask').reset_index(name='clients'))

    def uncertainty(self):
        print('Cluster bootstrap:',len(self.metric_records),'rates;',REPS,'replicates',flush=True)
        contributions=sparse.hstack(self.contributions,format='csr',dtype=np.float32)
        rng=np.random.default_rng(SEED)
        probs=np.full(self.nclients,1/self.nclients)
        draws=np.empty((REPS,contributions.shape[1]),dtype=np.float64)
        for start in range(0,REPS,32):
            take=min(32,REPS-start)
            weights=rng.multinomial(self.nclients,probs,size=take).astype(np.float32)
            draws[start:start+take]=weights @ contributions
            if start%320==0:print('  bootstrap',start,'/',REPS,flush=True)
        den,num=draws[:,0::2],draws[:,1::2]
        rates=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
        baseline={r['target']:r['metric_id'] for r in self.metric_records if r['kind']=='prevalence'}
        records=[]
        for r in self.metric_records:
            idx=r['metric_id'];dist=rates[:,idx];valid=np.isfinite(dist)
            r.update(ci_low=float(np.nanquantile(dist,.025)),ci_high=float(np.nanquantile(dist,.975)),
                     ci_method='client cluster bootstrap',replicates=REPS,seed=SEED,valid_replicates=int(valid.sum()))
            if r['kind'] in ['feature','category']:
                b=baseline[r['target']]
                lift=np.divide(dist,rates[:,b],out=np.full(REPS,np.nan),where=rates[:,b]>0)
                r.update(baseline=self.metric_records[b]['rate'],lift=r['rate']/self.metric_records[b]['rate'],
                         lift_ci_low=float(np.nanquantile(lift,.025)),lift_ci_high=float(np.nanquantile(lift,.975)),
                         lift_valid_replicates=int(np.isfinite(lift).sum()))
            records.append(r)
        self.save('rates_cluster_ci',records)
        transition=[]
        for target in TARGETS:
            a=next(r for r in records if r['kind']=='transition' and r['target']==target and r['metric']=='previous_1')
            b=next(r for r in records if r['kind']=='transition' and r['target']==target and r['metric']=='previous_0')
            ratio=np.divide(rates[:,a['metric_id']],rates[:,b['metric_id']],out=np.full(REPS,np.nan),where=rates[:,b['metric_id']]>0)
            diff=rates[:,a['metric_id']]-rates[:,b['metric_id']]
            transition.append(dict(target=target,repeat_rate=a['rate'],nonbuyer_rate=b['rate'],
                difference=a['rate']-b['rate'],difference_ci_low=float(np.nanquantile(diff,.025)),difference_ci_high=float(np.nanquantile(diff,.975)),
                relative_rate=a['rate']/b['rate'],ratio_ci_low=float(np.nanquantile(ratio,.025)),ratio_ci_high=float(np.nanquantile(ratio,.975))))
        self.save('target_transition_comparison',transition)
        self.checks['bootstrap_replicates']=REPS
        self.checks['bootstrap_finite_rate_fraction']=float(np.isfinite(rates).mean())

    def reuse(self):
        sources={
          'raw_missing':'docs/mbd_mini_eda/tables/review/missing_values.csv',
          'raw_duplicates':'docs/mbd_mini_eda/tables/review/duplicates.csv',
          'sequence_reconciliation':'docs/mbd_mini_eda/tables/review/sequence_reconciliation.csv',
          'amount_currency':'deliverables/eda/tables/amount_by_currency.csv',
          'amount_outliers':'deliverables/eda/tables/amount_outliers_by_currency.csv',
        }
        self.reused=[]
        for name,rel in sources.items():
            p=ROOT/rel
            self.save(name,pd.read_csv(p))
            self.reused.append(dict(table=name,source=rel,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                verification='raw input fingerprint matches full-history review; updated log1p outlier table'))

    def finish(self):
        self.checks['raw_inventory_unchanged']=all((DATA/r['path']).stat().st_size==r['bytes'] and (DATA/r['path']).stat().st_mtime_ns==r['mtime_ns'] for r in self.inventory)
        assert self.checks['raw_inventory_unchanged']
        dump(WORK/'input_inventory.json',self.inventory)
        dump(OUT/'manifest.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
            dataset='ai-lab/MBD-mini',scope='all 12 label months, full detail event history; calendar T-1 comparisons',
            fingerprint=self.fingerprint,fingerprint_kind='Parquet relative paths, bytes, mtimes, row counts and schemas; not raw content hash',
            timezone='UTC research convention; original label ETL timezone unconfirmed',
            mapping='T = calendar month containing mon',window='[start(T-1), start(T))',
            bootstrap=dict(unit='client_id',replicates=REPS,seed=SEED,method='nonparametric multinomial cluster bootstrap; shared draws for group and baseline'),
            model_fits=0,raw_modified=False,individual_data='outputs/dataset_eda/customer_month_snapshots.parquet (local only)',
            tables=self.published,reused=self.reused,checks=self.checks,duration_seconds=round(time.time()-self.started,1)))
        self.c.close()
        print('DONE',json.dumps(self.checks),flush=True)


if __name__=='__main__':
    a=Analysis();a.labels();a.events();a.comparisons();a.uncertainty();a.reuse();a.finish()
