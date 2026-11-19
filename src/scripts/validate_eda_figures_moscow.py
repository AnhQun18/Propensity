"""Validate produced figure bundles and rebuild the navigational index."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/eda_figures_moscow'


def main():
    bundles=[]
    for png in sorted(OUT.glob('*.png')):
        md=OUT/f'{png.stem}.md'
        assert md.exists(),str(md)
        title=md.read_text(encoding='utf-8').splitlines()[0].removeprefix('# ')
        bundles.append(dict(figure=png.stem,title=title,data=f'{png.stem}.csv',explanation=md.name))
    index=pd.DataFrame(bundles).sort_values('figure')
    assert index.figure.is_unique
    checks={}
    for r in index.itertuples():
        for ext in ['png','svg','csv','md']:
            f=OUT/f'{r.figure}.{ext}'
            assert f.exists() and f.stat().st_size>0, str(f)
        d=pd.read_csv(OUT/f'{r.figure}.csv')
        assert len(d)>0
        text=(OUT/f'{r.figure}.md').read_text(encoding='utf-8')
        # Make machine scalar repr human readable in older already-rendered bundles.
        text=re.sub(r'np\.float64\(([-+0-9.eE]+)\)',lambda m:f'{float(m[1]):.4g}',text)
        text=text.replace(f']({r.figure}.png)',f'](<{(OUT/f"{r.figure}.png").as_posix()}>)')
        assets=f'[CSV](<{(OUT/f"{r.figure}.csv").as_posix()}>) · [SVG](<{(OUT/f"{r.figure}.svg").as_posix()}>)'
        if assets not in text:
            text=text.replace('## Hình thể hiện gì',assets+'\n\n## Hình thể hiện gì')
        (OUT/f'{r.figure}.md').write_text(text,encoding='utf-8')
        assert all(s in text for s in ['Cách dựng','insight','Giới hạn'])
    checks['all_bundles_have_png_svg_csv_explanation']=True
    for metric in ['trx','geo','dialog','amount11']:
        d=pd.read_csv(OUT/f'event_{metric}.csv')
        assert d.anchor.nunique()==12 and len(d)==12*7*2
        assert (d.observed+d.unavailable==d.population).all()
        assert d.groupby(['anchor','lag']).population.sum().eq(100224).all()
        z=pd.read_csv(OUT/f'composition_{metric}.csv')
        assert z.groupby('month').customers.sum().eq(100224).all()
    checks['all_12_anchors_full_population_and_unavailable_reconcile']=True
    sync=pd.read_csv(OUT/'cross_source_daily_synchrony.csv')
    assert (sync.observed_customers+sync.unavailable==100224).all()
    checks['synchrony_population_reconciles']=True
    paired=pd.read_csv(OUT/'paired_by_anchor.csv')
    assert paired.groupby(['metric','purchase']).anchor.nunique().eq(12).all()
    assert (paired.paired<=paired.population).all()
    checks['paired_panel_keeps_all_label_months']=True
    daily=pd.read_csv(OUT/'daily_validation.csv')
    assert (daily.raw_events==daily.daily_events).all()
    checks['raw_daily_event_totals_reconcile']=True
    groups=pd.read_csv(OUT/'reactivation_population_states.csv')
    assert groups.groupby(['source','month']).customers.sum().eq(100224).all()
    checks['reactivation_states_exhaust_population']=True
    for source,total in [('trx',38764930),('geo',66295724),('dialog',217205)]:
        q=pd.read_parquet(OUT/f'{source}_raw_interarrival.parquet')
        assert int(q.events.sum())==total
        assert (q.gaps==q.events-1).all()
    checks['raw_interarrival_events_and_pairs_reconcile']=True
    for column,total in [('geohash_4',66295724),('geohash_5',66295724),('geohash_6',66295724),('event_type',38764930)]:
        q=pd.read_parquet(OUT/f'{column}_monthly_profiles.parquet')
        assert int(q.events.sum())==total
        assert q.new_cell_event_share.between(0,1).all()
        assert not q.duplicated(['client_id','month']).any()
    checks['monthly_categorical_profiles_reconcile_raw']=True
    strata=pd.read_csv(OUT/'fixed_past_trx_strata_event_study.csv')
    assert strata.groupby(['source','anchor','lag']).population.sum().eq(100224).all()
    observation_month=pd.to_datetime(strata.anchor).dt.to_period('M')+strata.lag
    partial=(strata.source=='geo')&(observation_month==pd.Period('2021-12'))
    assert strata.loc[partial,'observed'].eq(0).all()
    checks['fixed_strata_exhaust_population_and_preserve_partial_month_unknown']=True
    index.to_csv(OUT/'figure_index.csv',index=False)
    text='# Danh mục hình EDA\n\nMỗi liên kết mở hình và giải thích riêng; CSV/SVG cùng tên gốc.\n\n'
    text+='\n'.join(f"- [{r.title}](<{(OUT/r.explanation).as_posix()}>)" for r in index.itertuples())
    (OUT/'INDEX.md').write_text(text+'\n',encoding='utf-8')
    inventory=[]
    for p in sorted((ROOT/'data/MBD-mini').rglob('*.parquet')):
        s=p.stat();inventory.append({'path':p.relative_to(ROOT).as_posix(),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns})
    previous=json.loads((ROOT/'outputs/mbd_review_cache/source_inventory.json').read_text(encoding='utf-8'))
    current={p['path'].removeprefix('data/MBD-mini/'):p for p in inventory}
    assert set(current)=={p['path'] for p in previous['files']}
    for p in previous['files']:
        assert current[p['path']]['bytes']==p['bytes'] and current[p['path']]['mtime_ns']==p['mtime_ns'],p['path']
    checks['raw_inventory_matches_previous_validated_cache']=True
    scripts=sorted(set((ROOT/'scripts').glob('extend_eda_*.py')) |
                   {ROOT/'src/scripts/complete_eda_figures_moscow.py', Path(__file__).resolve()})
    manifest={'timezone':'Europe/Moscow','raw_naive_assumption':'UTC; ETL unconfirmed',
        'customers':100224,'label_months':12,'no_horizons':True,'no_product_breakdown':True,
        'figures':len(index),'raw_inventory':inventory,
        'scripts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},
        'note':'Inventory is paths/size/mtime, not raw byte checksums. Source raw not modified.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    status=json.loads((OUT/'execution_status.json').read_text(encoding='utf-8'))
    status['done']=sorted(set(status['done']))
    status['figures']=len(index)
    status['validation']='See validation.json; raw daily totals checked; bootstrap applies to paired monthly count/amount only.'
    status['script_sha256_current']=manifest['scripts']
    status['pending']=[
        'Word2Vec/cell2vec: gensim unavailable; installation via configured index returned no matching distribution',
        'ADI-CV2 and Kaplan-Meier with repeated-event customer uncertainty',
        'Three-source joint synchrony and earliest-source reactivation sequence',
        'GEO transition/IDF/rare-pair/node-articulation temporal stability and pre/post cell rate smoothing',
        'Dialog near-duplicate kNN, within-client cosine and customer-balanced geometry',
        'Time-stable bag/type/cell embeddings and cluster bootstrap beyond fold stability',
        'Pre/post concentration/amount composition with complete fixed activity AND history-length strata',
        'PSI/MMD and bootstrap CIs for extended raw features beyond paired core counts/amount',
        'Complete checkbox-by-checkbox acceptance audit against original three lists']
    (OUT/'execution_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    # Contact sheets are QA only, outside the user figure catalog.
    qa=OUT/'qa';qa.mkdir(exist_ok=True)
    for start in range(0,len(index),12):
        fig,axes=plt.subplots(4,3,figsize=(18,16))
        for ax in axes.flat:ax.axis('off')
        for ax,r in zip(axes.flat,list(index.itertuples())[start:start+12]):
            ax.imshow(plt.imread(OUT/f'{r.figure}.png'));ax.set_title(r.figure,fontsize=8)
        fig.tight_layout();fig.savefig(qa/f'contact_{start//12+1:02d}.png',dpi=90);plt.close(fig)
    print(json.dumps({'figures':len(index),'checks':checks,'complete_plan':False},ensure_ascii=False))


if __name__=='__main__':main()
