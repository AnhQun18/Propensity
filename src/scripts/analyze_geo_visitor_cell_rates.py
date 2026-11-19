"""All-visitor cell rates; preserve client dependence in label permutations."""
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from analyze_geo_spatial_purchase_moscow import OUT,CUTOFF,SEED,wilson


def main():
    d=pd.read_parquet(OUT/'customer_snapshot.parquet').reset_index(drop=True)
    ids=d[['client_id']].copy();ids['customer_index']=np.arange(len(d))
    cache=OUT/'visitor_memberships.parquet'
    if cache.exists():
        v=pd.read_parquet(cache)
    else:
        c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='3GB'")
        c.register('ids',ids)
        parts=[]
        for fold in range(5):
            parts.append(c.execute(f"SELECT DISTINCT i.customer_index,g.geohash_4 g4,g.geohash_5 g5,g.geohash_6 g6 FROM read_parquet('data/MBD-mini/detail/geo/fold={fold}/*.parquet') g JOIN ids i USING(client_id) WHERE g.event_time<TIMESTAMP '{CUTOFF}'").df())
            print('Visitor memberships fold',fold,flush=True)
        v=pd.concat(parts,ignore_index=True)
        v.to_parquet(cache,index=False)
    assert v.customer_index.nunique()==d.has_geo.sum()
    edges={};codes={};maps={};raw={}
    for level,parent in [('g4',None),('g5','g4'),('g6','g5')]:
        e=v[['customer_index']+([parent] if parent else [])+[level]].drop_duplicates()
        keys,index=pd.factorize(e[level],sort=True)
        edges[level]=e.customer_index.to_numpy(dtype=int);codes[level]=keys
        maps[level]=pd.DataFrame({'cell':index,'n':np.bincount(keys)})
        if parent:
            assert e.groupby(level)[parent].nunique().max()==1
            maps[level]['parent_cell']=maps[level].cell.map(e.drop_duplicates(level).set_index(level)[parent])
        else:maps[level]['parent_cell']=np.nan
        for h in [1,2,3]:
            r=maps[level].copy()
            r['buyers']=np.bincount(keys,weights=d[f'y_{h}'].to_numpy()[edges[level]]).astype(int)
            r['rate']=r.buyers/r.n
            r['wilson_low'],r['wilson_high']=wilson(r.buyers,r.n)
            r['horizon']=h;r['level']=level
            raw[(h,level)]=r
    smooth=[];support=[]
    for h in [1,2,3]:
        for level in ['g4','g5','g6']:
            r=raw[(h,level)]
            support.append(dict(horizon=h,level=level,cells=len(r),cells_n_ge100=int(r.n.ge(100).sum()),median_n=r.n.median(),cells_n_lt10=int(r.n.lt(10).sum())))
        for strength in [20,100,500]:
            previous=None
            for level in ['g4','g5','g6']:
                r=raw[(h,level)].copy()
                r['prior_mean']=d[f'y_{h}'].mean() if previous is None else r.parent_cell.map(previous.set_index('cell').smoothed)
                r['smoothed']=(r.buyers+strength*r.prior_mean)/(r.n+strength)
                r['data_weight']=r.n/(r.n+strength);r['strength']=strength
                smooth.append(r);previous=r
    pd.concat(smooth,ignore_index=True).to_parquet(OUT/'visitor_hierarchical_cell_rates.parquet',index=False)
    display=pd.concat([raw[(h,level)] for h in [1,2,3] for level in ['g4','g5']],ignore_index=True)
    display['meets_display_n100']=display.n.ge(100)
    display.to_csv(OUT/'visitor_g4_g5_wilson.csv',index=False)
    pd.DataFrame(support).to_csv(OUT/'visitor_cell_support.csv',index=False)
    # All edges of a customer use the same permuted label. Preserve TRX-stratum totals.
    mask=d.has_geo.to_numpy();blocks=[np.asarray(ix) for ix in d[mask].groupby('trx_stratum').groups.values()]
    tests=[];B=499
    for h in [1,2,3]:
        y=d[f'y_{h}'].to_numpy(dtype=float)
        p=np.zeros(len(d))
        for ix in blocks:p[ix]=y[ix].mean()
        specs={}
        for level in ['g4','g5']:
            code=codes[level];ci=edges[level]
            expected=np.bincount(code,weights=p[ci])
            variance=np.bincount(code,weights=p[ci]*(1-p[ci]));valid=variance>0
            def statistic(y,code=code,ci=ci,expected=expected,variance=variance,valid=valid):
                obs=np.bincount(code,weights=y[ci],minlength=len(expected))
                return np.sum((obs[valid]-expected[valid])**2/variance[valid])
            specs[level]=(statistic,float(statistic(y)),[])
        rng=np.random.default_rng(SEED+500+h)
        for _ in range(B):
            yp=y.copy()
            for ix in blocks:yp[ix]=rng.permutation(y[ix])
            for fun,observed,null in specs.values():null.append(float(fun(yp)))
        for level,(fun,observed,null) in specs.items():
            null=np.asarray(null)
            tests.append(dict(horizon=h,level=level,condition='within_trx_strata_all_visitors',customers=int(mask.sum()),cells=len(maps[level]),
                memberships=len(edges[level]),statistic=observed,null_mean=null.mean(),null_p95=np.quantile(null,.95),
                permutation_p=(1+int((null>=observed).sum()))/(B+1),permutations=B))
        print('Visitor permutations H=',h,flush=True)
    t=pd.DataFrame(tests)
    # Holm correction across all planned modal and visitor tests (18 comparisons).
    modal=pd.read_csv(OUT/'global_cell_permutation.csv')
    combined=pd.concat([modal.drop(columns='holm_p'),t],ignore_index=True)
    order=np.argsort(combined.permutation_p.to_numpy());adjusted=np.maximum.accumulate(combined.permutation_p.to_numpy()[order]*(len(combined)-np.arange(len(combined))))
    combined['holm_p']=0.0;combined.loc[order,'holm_p']=np.minimum(adjusted,1)
    combined.to_csv(OUT/'global_cell_permutation_all_definitions.csv',index=False)
    t=combined.tail(len(t));t.to_csv(OUT/'visitor_global_permutation.csv',index=False)
    for level in ['g4','g5','g6']:
        assert (raw[(3,level)].buyers<=raw[(3,level)].n).all()
        assert raw[(3,level)].n.sum()==len(edges[level])
    print(pd.DataFrame(support).query('horizon==3').to_string(index=False))
    print(t.to_string(index=False))


if __name__=='__main__':main()
