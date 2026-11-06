"""Full-history MBD-mini review, persistent query cache and linked stratified sample.

Run from the repository: python -u -X utf8 src/review_mbd_eda.py
No event is dropped for a missing non-key attribute or an outlier flag.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deliverables/mbd_mini_eda_fixes"))
from complete_mbd_statistics import normalized_expression, qi, sql_literal
from mbd_mini_eda.statistics import numeric_summary

SOURCES = {"clients": "client_split", "targets": "targets", "trx": "detail/trx",
           "geo": "detail/geo", "dialog": "detail/dialog", "ptls_trx": "ptls/trx",
           "ptls_geo": "ptls/geo", "ptls_dialog": "ptls/dialog"}
EVENTS = ("trx", "geo", "dialog")
COLORS = ("#257568", "#c8794b", "#7864a6", "#326fb0")
VERSION = 1


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")


def source_manifest(root):
    records, schemas = [], {}
    for source, relative in SOURCES.items():
        files = sorted((root / relative).rglob("*.parquet"))
        if not files:
            raise ValueError(f"No parquet files: {relative}")
        schema = None
        for path in files:
            reader = pq.ParquetFile(path)
            current = reader.schema_arrow.remove_metadata()
            if schema is not None and not current.equals(schema):
                raise ValueError(f"Schema changed: {path}")
            schema = current
            stat = path.stat()
            records.append(dict(source=source, path=path.relative_to(root).as_posix(),
                                bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                rows=reader.metadata.num_rows, schema=str(current)))
        schemas[source] = schema
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return records, schemas, fingerprint


class QueryCache:
    """Cache final query results, including expensive exact duplicate checks.

    Results and query digests commit together. Source inventory changes refuse
    reuse rather than silently returning stale statistics. No raw table copy.
    """
    def __init__(self, directory, fingerprint, memory="3GB", threads=2):
        directory.mkdir(parents=True, exist_ok=True)
        self.db = duckdb.connect(str(directory / "review.duckdb"))
        self.db.execute(f"SET memory_limit={sql_literal(memory)}")
        self.db.execute(f"SET threads={int(threads)}")
        self.db.execute(f"SET temp_directory={sql_literal((directory / 'spill').as_posix())}")
        self.db.execute("SET preserve_insertion_order=false")
        self.db.execute("CREATE TABLE IF NOT EXISTS cache_meta(key VARCHAR PRIMARY KEY, value VARCHAR)")
        saved = self.db.execute("SELECT value FROM cache_meta WHERE key='source_fingerprint'").fetchone()
        if saved and saved[0] != fingerprint:
            self.db.close()
            raise ValueError("Raw inventory changed; choose a new --cache-dir. Existing cache retained.")
        self.db.execute("INSERT OR REPLACE INTO cache_meta VALUES ('source_fingerprint',?)", [fingerprint])
        self.hits, self.computed = [], []

    def get(self, name, sql):
        digest = hashlib.sha256((str(VERSION) + sql).encode()).hexdigest()
        key = "query:" + name
        saved = self.db.execute("SELECT value FROM cache_meta WHERE key=?", [key]).fetchone()
        if saved and saved[0] == digest:
            self.hits.append(name)
            print(f"Cache: {name}", flush=True)
        else:
            print(f"Scan: {name}", flush=True)
            started = time.perf_counter()
            self.db.execute("BEGIN")
            try:
                self.db.execute(f"CREATE OR REPLACE TABLE {qi('cache_'+name)} AS {sql}")
                self.db.execute("INSERT OR REPLACE INTO cache_meta VALUES (?,?)", [key, digest])
                self.db.execute("COMMIT")
            except Exception:
                self.db.execute("ROLLBACK")
                raise
            self.computed.append(name)
            print(f"  {time.perf_counter()-started:.1f}s", flush=True)
        return self.db.execute(f"SELECT * FROM {qi('cache_'+name)}").df()


def open_views(cache, data_root):
    for source, relative in SOURCES.items():
        paths = [p.as_posix() for p in sorted((data_root / relative).rglob("*.parquet"))]
        cache.db.read_parquet(paths, hive_partitioning=True).create_view("src_" + source, replace=True)


def scan(cache, schemas):
    tables = {}
    missing, duplicate, sequence = [], [], []
    for source, schema in schemas.items():
        view = "src_" + source
        columns = list(schema.names) + (["fold"] if "fold" not in schema.names else [])
        expr = ["count(*) rows"]
        expr += [f"count(*) FILTER(WHERE {qi(c)} IS NULL) AS {qi(c)}" for c in columns]
        stats = cache.get("missing_"+source, f"SELECT {','.join(expr)} FROM {view}").iloc[0]
        for c in columns:
            missing.append(dict(source=source, column=c, rows=int(stats['rows']),
                                missing_n=int(stats[c]), missing_rate=float(stats[c]/stats['rows'])))
        keys = ["client_id", "mon"] if source == "targets" else ["client_id"]
        if source not in EVENTS:
            k = ",".join(map(qi, keys))
            d = cache.get("key_dups_"+source,
                f"SELECT COALESCE(sum(n-1),0)::BIGINT duplicate_n,count(*)::BIGINT duplicate_groups "
                f"FROM (SELECT {k},count(*) n FROM {view} GROUP BY ALL HAVING n>1)").iloc[0]
            duplicate.append(dict(source=source, kind="unique_key", duplicate_n=int(d.duplicate_n),
                                  duplicate_groups=int(d.duplicate_groups), analyzed_rows=int(stats['rows']), scope="full"))
        # ptls rows have a declared customer key, checked above. Full-row
        # equality on detail includes all scalar/list fields and partition fold.
        if source in ("clients", "targets", *EVENTS):
            types = {f.name: f.type for f in schema}
            expressions = [normalized_expression(qi(c), types[c])+" AS "+qi(c) if c in types else qi(c)
                           for c in columns]
            d = cache.get("full_dups_"+source,
                f"SELECT COALESCE(sum(n-1),0)::BIGINT duplicate_n,count(*)::BIGINT duplicate_groups "
                f"FROM (SELECT {','.join(expressions)},count(*) n FROM {view} GROUP BY ALL HAVING n>1)").iloc[0]
            duplicate.append(dict(source=source, kind="full_row", duplicate_n=int(d.duplicate_n),
                                  duplicate_groups=int(d.duplicate_groups), analyzed_rows=int(stats['rows']), scope="full"))
        if source in EVENTS:
            tables["month_"+source] = cache.get("month_"+source,
                f"SELECT s.client_id,strftime(s.event_time,'%Y-%m') AS \"month\",count(*)::BIGINT events "
                f"FROM {view} s JOIN src_clients c USING(client_id) WHERE s.event_time IS NOT NULL GROUP BY ALL")
            tables["daily_"+source] = cache.get("daily_"+source,
                f"SELECT CAST(event_time AS DATE) AS \"day\",count(*)::BIGINT events FROM {view} s "
                "JOIN src_clients c USING(client_id) WHERE event_time IS NOT NULL GROUP BY ALL ORDER BY \"day\"")
            tables["time_"+source] = cache.get("time_"+source,
                f"SELECT min(event_time) first_event,max(event_time) last_event FROM {view}")
            d = cache.get("time_dups_"+source,
                f"SELECT COALESCE(sum(n-1),0)::BIGINT duplicate_n,count(*)::BIGINT duplicate_groups "
                f"FROM (SELECT client_id,event_time,count(*) n FROM {view} GROUP BY ALL HAVING n>1)").iloc[0]
            duplicate.append(dict(source=source, kind="same_client_time_candidate", duplicate_n=int(d.duplicate_n),
                                  duplicate_groups=int(d.duplicate_groups), analyzed_rows=int(stats['rows']), scope="full"))
        if source.startswith("ptls_"):
            for field in schema:
                if not pa.types.is_list(field.type):
                    continue
                c = qi(field.name)
                expr = (f"SELECT count(*) rows,COALESCE(sum(array_length({c})),0)::BIGINT elements,"
                        f"COALESCE(sum(array_length({c})-list_count({c})),0)::BIGINT null_elements,"
                        f"count(*) FILTER(WHERE array_length({c}) IS DISTINCT FROM array_length(event_time))::BIGINT length_mismatch_rows "
                        f"FROM {view}")
                r = cache.get("sequence_"+source+"_"+field.name, expr).iloc[0].to_dict()
                sequence.append(dict(source=source, column=field.name, **r))
    tables['missing_values'] = pd.DataFrame(missing)
    tables['duplicates'] = pd.DataFrame(duplicate)
    tables['sequence_quality'] = pd.DataFrame(sequence)
    reconciliation=[]
    for source in EVENTS:
        reconciliation.append(cache.get('sequence_reconciliation_'+source,f"""WITH d AS (
            SELECT client_id,sum(events)::BIGINT events FROM cache_month_{source} GROUP BY client_id),
            p AS (SELECT client_id,array_length(event_time) events FROM src_ptls_{source})
            SELECT '{source}' source,sum(coalesce(d.events,0))::BIGINT detail_events,
            sum(coalesce(p.events,0))::BIGINT ptls_events,max(d.events) max_detail_events,
            max(p.events) max_sequence_events,count(*) FILTER(WHERE p.events<d.events)::BIGINT customers_shorter_sequence,
            count(*) FILTER(WHERE p.events>d.events)::BIGINT customers_longer_sequence,
            count(*) FILTER(WHERE (p.events IS NULL)!=(d.events IS NULL))::BIGINT customer_presence_mismatch
            FROM src_clients c LEFT JOIN d USING(client_id) LEFT JOIN p USING(client_id)"""))
    tables['sequence_reconciliation']=pd.concat(reconciliation,ignore_index=True)
    tables['embedding_element_quality']=cache.get('embedding_element_quality',"""SELECT 'detail/dialog' source,
        count(*)::BIGINT rows,count(*) FILTER(WHERE embedding IS NULL)::BIGINT null_vectors,
        count(*) FILTER(WHERE array_length(embedding)<>768)::BIGINT non_768_vectors,
        sum(array_length(list_filter(embedding,lambda x: x IS NULL)))::BIGINT null_scalar_elements,
        sum(array_length(list_filter(embedding,lambda x: x IS NOT NULL AND NOT isfinite(x))))::BIGINT nonfinite_scalar_elements
        FROM src_dialog UNION ALL SELECT 'ptls/dialog',count(*),
        sum(array_length(list_filter(embedding,lambda v: v IS NULL))),
        sum(array_length(list_filter(embedding,lambda v: array_length(v)<>768))),
        sum(list_sum(list_transform(embedding,lambda v: array_length(list_filter(v,lambda x: x IS NULL))))),
        sum(list_sum(list_transform(embedding,lambda v: array_length(list_filter(v,lambda x: x IS NOT NULL AND NOT isfinite(x))))))
        FROM src_ptls_dialog""")
    tables['null_target_context'] = cache.get("null_target_context", """SELECT
        CASE WHEN trans_count=0 THEN 'zero' WHEN trans_count=1 THEN 'one'
             WHEN trans_count>1 THEN 'two_or_more' ELSE 'negative_or_null' END trans_count_group,
        count(*)::BIGINT rows,count(*) FILTER(WHERE diff_trans_date IS NULL)::BIGINT missing_diff,
        min(trans_count) min_trans_count,max(trans_count) max_trans_count
        FROM src_targets GROUP BY ALL ORDER BY trans_count_group""")
    pairs = []
    for left, right in (("src_type11","src_type12"),("dst_type11","dst_type12"),
                        ("src_type21","src_type22"),("src_type31","src_type32")):
        frame = cache.get("null_pair_"+left, f"""SELECT {sql_literal(left+' / '+right)} pair,
            count(*) FILTER(WHERE {left} IS NULL AND {right} IS NULL)::BIGINT both_null,
            count(*) FILTER(WHERE ({left} IS NULL) != ({right} IS NULL))::BIGINT only_one_null,
            count(*)::BIGINT rows FROM src_trx""")
        pairs.append(frame)
    tables['null_pairs'] = pd.concat(pairs, ignore_index=True)
    tables['null_by_event_type'] = cache.get('null_by_event_type', """SELECT event_type,
        count(*)::BIGINT rows,count(*) FILTER(WHERE currency IS NULL)::BIGINT null_currency,
        count(*) FILTER(WHERE src_type11 IS NULL)::BIGINT null_src11,
        count(*) FILTER(WHERE dst_type11 IS NULL)::BIGINT null_dst11,
        count(*) FILTER(WHERE src_type21 IS NULL)::BIGINT null_src21,
        count(*) FILTER(WHERE src_type31 IS NULL)::BIGINT null_src31
        FROM src_trx GROUP BY ALL ORDER BY event_type""")
    tables['amount_by_currency'] = cache.get('amount_by_currency', """SELECT
        CAST(currency AS VARCHAR) currency,count(*)::BIGINT rows,
        count(*) FILTER(WHERE isfinite(amount))::BIGINT finite_n,
        avg(amount::DOUBLE) FILTER(WHERE isfinite(amount)) mean,
        min(amount::DOUBLE) FILTER(WHERE isfinite(amount)) min,max(amount::DOUBLE) FILTER(WHERE isfinite(amount)) max,
        quantile_cont(amount::DOUBLE,[0.25,0.5,0.75,0.95,0.99]) FILTER(WHERE isfinite(amount)) quantiles
        FROM src_trx GROUP BY currency ORDER BY currency""")
    tables['amount_outliers_currency'] = cache.get('amount_outliers_currency', """WITH q AS (
        SELECT currency,quantiles[1] q1,quantiles[3] q3 FROM cache_amount_by_currency), a AS (
        SELECT t.*,q1-1.5*(q3-q1) lower_fence,q3+1.5*(q3-q1) upper_fence
        FROM src_trx t JOIN q ON t.currency IS NOT DISTINCT FROM q.currency::INTEGER)
        SELECT CAST(currency AS VARCHAR) currency,count(*)::BIGINT finite_n,
        min(lower_fence) lower_fence,min(upper_fence) upper_fence,
        count(*) FILTER(WHERE amount<lower_fence OR amount>upper_fence)::BIGINT outlier_n,
        count(*) FILTER(WHERE amount<=0)::BIGINT nonpositive_n,max(amount::DOUBLE) max
        FROM a WHERE isfinite(amount) GROUP BY currency ORDER BY currency""")
    # Original values, explicit unequal-width bins: bar heights are counts,
    # never density or a transformed x axis. All tails are retained.
    edges = [-np.inf,0,1000,5000,10000,25000,50000,100000,250000,500000,1e6,5e6,1e7,1e8,1e9,np.inf]
    cases = " ".join(f"WHEN amount<{b:.17g} THEN {i}" for i,b in enumerate(edges[1:-1]))
    hist = cache.get('amount_raw_bins', f"SELECT CASE {cases} ELSE {len(edges)-2} END bin,"
                     "count(*)::BIGINT n FROM src_trx WHERE isfinite(amount) GROUP BY ALL ORDER BY bin")
    hist = hist.set_index('bin').reindex(range(len(edges)-1), fill_value=0).reset_index()
    hist['left'], hist['right'] = edges[:-1], edges[1:]
    tables['amount_raw_bins'] = hist
    # Global quantiles must be computed globally, not averaged over currencies.
    summary = cache.get('amount_full', """SELECT count(*)::BIGINT finite_n,avg(amount::DOUBLE) mean,
        stddev_samp(amount::DOUBLE) std,min(amount::DOUBLE) min,max(amount::DOUBLE) max,
        quantile_cont(amount::DOUBLE,[0.25,0.5,0.75,0.95,0.99]) quantiles
        FROM src_trx WHERE isfinite(amount)""")
    q99 = float(summary.iloc[0].quantiles[-1])
    tables['amount_full'] = summary
    tables['amount_zoom'] = cache.get('amount_zoom', f"""SELECT
        least(29, floor(amount::DOUBLE/{q99:.17g}*30)::INTEGER) bin,count(*)::BIGINT n
        FROM src_trx WHERE isfinite(amount) AND amount>=0 AND amount<={q99:.17g}
        GROUP BY ALL ORDER BY bin""")
    tables['amount_zoom']['left'] = tables['amount_zoom'].bin*q99/30
    tables['amount_zoom']['right'] = (tables['amount_zoom'].bin+1)*q99/30
    return tables


def make_cube(clients, tables):
    first = min(tables['month_'+s].month.min() for s in EVENTS)
    last = max(tables['month_'+s].month.max() for s in EVENTS)
    months = pd.period_range(first, last, freq='M').astype(str).to_numpy()
    ids = pd.Index(clients.client_id)
    cube = np.zeros((len(ids),len(months),3),dtype=np.uint32)
    for i,s in enumerate(EVENTS):
        frame = tables['month_'+s]
        if frame.events.max() > np.iinfo(np.uint32).max:
            raise OverflowError('Customer-month count exceeds uint32')
        cube[ids.get_indexer(frame.client_id),pd.Index(months).get_indexer(frame.month),i] = frame.events
    return cube, months


def activity_bands(totals):
    positive = totals[totals>0]
    cuts = np.unique(np.quantile(positive,[.25,.5,.75,.95,.99])) if len(positive) else np.array([])
    labels = np.searchsorted(cuts,totals,side='left')+1
    labels[totals==0] = 0
    return labels, cuts


def purposeful_groups(totals, size=8):
    """Disjoint actual people nearest population statistics; stable ID ties."""
    used, groups = set(), {}
    index = np.arange(len(totals))
    criteria = [("Nhiều nhất", np.lexsort((index,-totals.astype(float)))),
                ("Gần trung bình", np.lexsort((index,np.abs(totals-totals.mean())))),
                ("Gần trung vị", np.lexsort((index,np.abs(totals-np.median(totals))))),
                ("Ít nhất có log", index[totals>0][np.argsort(totals[totals>0],kind='stable')]),
                ("Không có log", index[totals==0])]
    for name,order in criteria:
        chosen = [int(i) for i in order if i not in used][:size]
        groups[name] = chosen
        used.update(chosen)
    return groups


def nullable_months(cube, months):
    """Preserve every positive count; unavailable early geo/dialog cells are null.

    For Jan 2022 onward zero means no recorded log, not certified inactivity.
    An early customer-month without geo/dialog stays unknown, even if other
    customers have logs in that month. Raw arrays remain lossless in cache.
    """
    values = cube.astype(float)
    for i in (1,2):
        early = months < '2022-01'
        part = values[:,early,i]
        part[part==0] = np.nan
        values[:,early,i] = part
    return values


def stratified_sample(clients, totals, masks, size, seed):
    if not 1 <= size <= len(clients):
        raise ValueError('sample-size must be 1..population')
    bands,cuts = activity_bands(totals)
    work = clients[['client_id','fold']].copy()
    work['activity_band'], work['source_mask'] = bands,masks
    keys = ['fold','source_mask','activity_band']
    work['stratum'] = work[keys].astype(str).agg(':'.join,axis=1)
    strata = work.groupby('stratum',sort=True).size().rename('population_n').reset_index()
    expected = strata.population_n.to_numpy()*size/len(work)
    allocation = np.floor(expected).astype(int)
    remainder = size-int(allocation.sum())
    # Hamilton allocation: exact sample size, no replacement or forced rare strata.
    order = np.lexsort((np.arange(len(strata)),-(expected-allocation)))
    allocation[order[:remainder]] += 1
    strata['sample_n'],strata['expected_n'] = allocation,expected
    rng = np.random.default_rng(seed)
    parts = []
    for row in strata.itertuples(index=False):
        group = work[work.stratum.eq(row.stratum)]
        if row.sample_n:
            chosen = rng.choice(group.index.to_numpy(),int(row.sample_n),replace=False)
            part = work.loc[chosen].copy()
            part['sampling_weight'] = row.population_n/row.sample_n
            parts.append(part)
    sample = pd.concat(parts).sort_values('client_id').reset_index(drop=True)
    assert len(sample)==size and not sample.client_id.duplicated().any()
    strata['population_rate'] = strata.population_n/len(work)
    strata['sample_rate'] = strata.sample_n/size
    strata['difference_pp'] = (strata.sample_rate-strata.population_rate)*100
    return sample,strata,cuts


def build_sample(cache, sample, directory, seed, fingerprint):
    directory.mkdir(parents=True,exist_ok=True)
    cache.db.register('selected_clients',sample[['client_id']])
    sample.to_parquet(directory/'selection.parquet',index=False)
    inventory=[]
    for source,relative in SOURCES.items():
        path=directory/(source+'.parquet')
        print('Sample: '+source,flush=True)
        cache.db.execute(f"COPY (SELECT s.* FROM src_{source} s JOIN selected_clients c USING(client_id)) "
                         f"TO {sql_literal(path.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)")
        rows=pq.ParquetFile(path).metadata.num_rows
        inventory.append(dict(source=source,rows=rows,path=path.name))
    save_json(directory/'metadata.json',dict(seed=seed,sample_customers=len(sample),
        source_fingerprint=fingerprint,method='proportional_stratified_without_replacement_Hamilton',
        strata=['fold','full_history_source_mask','full_history_activity_band'],
        scope='full_history_linked_customer_sample_for_EDA_only',tables=inventory,
        null_policy='source raw fields retained; customer_month.parquet distinguishes unknown early geo/dialog',
        leakage_note='Full-history strata are for EDA; recreate selection using only pre-cutoff data for modelling.'))
    return inventory


def run(args):
    started=time.perf_counter()
    for path in (args.output_dir,args.cache_dir,args.sample_dir):
        if path.resolve().is_relative_to(args.data_root.resolve()):
            raise ValueError('Outputs/cache/sample must be outside raw data')
    records,schemas,fingerprint=source_manifest(args.data_root)
    cache=QueryCache(args.cache_dir,fingerprint,args.memory_limit,args.threads)
    save_json(args.cache_dir/'source_inventory.json',dict(fingerprint=fingerprint,files=records))
    try:
        open_views(cache,args.data_root)
        clients=cache.get('clients','SELECT * FROM src_clients ORDER BY client_id')
        if clients.client_id.isna().any() or clients.client_id.duplicated().any():
            raise ValueError('client_split key null/duplicate')
        tables=scan(cache,schemas)
        cube,months=make_cube(clients,tables)
        unknown=np.zeros(cube.shape[:2],dtype=np.uint8)
        for i in (1,2):
            unknown |= (((cube[:,:,i]==0)&(months<'2022-01')[None,:]).astype(np.uint8)<<i)
        np.savez_compressed(args.cache_dir/'source_month_counts.npz',counts=cube,months=months,
                            unknown_source_mask=unknown,client_ids=clients.client_id.to_numpy(dtype=str),
                            source_fingerprint=fingerprint)
        totals=cube.sum(axis=1,dtype=np.uint64)
        union=totals.sum(axis=1)
        masks=sum(((totals[:,i]>0).astype(np.uint8)<<i) for i in range(3))
        sample,strata,cuts=stratified_sample(clients,union,masks,min(args.sample_size,len(clients)),args.seed)
        tables['sampling_strata']=strata
        inventory=build_sample(cache,sample,args.sample_dir,args.seed,fingerprint)
        from mbd_eda_review_report import write_review
        write_review(args,cache,clients,cube,months,tables,sample,inventory,records,schemas,fingerprint,cuts)
        save_json(args.cache_dir/'last_run.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
            fingerprint=fingerprint,cache_hits=cache.hits,computed=cache.computed,
            elapsed_seconds=round(time.perf_counter()-started,2)))
        print(f"Review complete: {args.output_dir/'report.md'}",flush=True)
    finally:
        cache.db.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=ROOT/'data/MBD-mini')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'docs/mbd_mini_eda')
    parser.add_argument('--cache-dir',type=Path,default=ROOT/'outputs/mbd_review_cache')
    parser.add_argument('--sample-dir',type=Path,default=ROOT/'sampledata/mbd_mini')
    parser.add_argument('--sample-size',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--memory-limit',default='3GB')
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    if args.sample_size<1 or args.seed<0 or args.threads<1:
        parser.error('sample-size/threads must be positive, seed nonnegative')
    run(args)


if __name__=='__main__':
    main()
