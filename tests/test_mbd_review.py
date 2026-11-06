"""Review contracts: full history, unknown vs zero, exact/cache/sample integrity."""
import argparse
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from review_mbd_eda import (QueryCache,nullable_months,purposeful_groups,stratified_sample,
                            source_manifest,open_views,scan,make_cube,build_sample)


class ReviewTests(unittest.TestCase):
    def test_early_missing_is_null_and_positive_old_logs_survive(self):
        cube=np.array([[[3,0,0],[2,1,0],[5,0,0]],[[0,0,1],[0,0,0],[0,0,0]]],dtype=np.uint32)
        original=cube.copy()
        values=nullable_months(cube,np.array(['2021-01','2021-12','2022-01']))
        self.assertTrue(np.isnan(values[0,0,1]))
        self.assertTrue(np.isnan(values[0,1,2]))
        self.assertEqual(values[0,1,1],1)
        self.assertEqual(values[1,0,2],1)
        self.assertEqual(values[0,2,1],0)
        self.assertEqual(values[0,0,0],3)
        np.testing.assert_array_equal(cube,original)

    def test_groups_are_deliberate_and_do_not_repeat_people(self):
        totals=np.r_[np.zeros(15),np.arange(1,101)]
        groups=purposeful_groups(totals,size=8)
        chosen=sum(groups.values(),[])
        self.assertEqual(len(chosen),len(set(chosen)))
        self.assertEqual(totals[groups['Nhiều nhất']].tolist(),list(range(100,92,-1)))
        self.assertTrue((totals[groups['Không có log']]==0).all())
        self.assertTrue((totals[groups['Ít nhất có log']]>0).all())

    def test_proportional_allocation_reproducible_no_replacement(self):
        clients=pd.DataFrame({'client_id':[f'c{i:04}' for i in range(997)],'fold':np.arange(997)%5})
        totals=np.r_[np.zeros(67),np.arange(930)]
        masks=(np.arange(997)%8).astype(np.uint8)
        sample,strata,_=stratified_sample(clients,totals,masks,133,42)
        self.assertEqual(len(sample),133)
        self.assertFalse(sample.client_id.duplicated().any())
        self.assertEqual(strata.sample_n.sum(),133)
        self.assertTrue((strata.sample_n<=strata.population_n).all())
        self.assertLessEqual((strata.sample_n-strata.expected_n).abs().max(),1)
        pd.testing.assert_frame_equal(sample,stratified_sample(clients,totals,masks,133,42)[0])
        self.assertFalse(sample.equals(stratified_sample(clients,totals,masks,133,43)[0]))

    def test_cache_reuses_results_invalidates_sql_and_refuses_changed_raw(self):
        with tempfile.TemporaryDirectory() as temp:
            cache=QueryCache(Path(temp),'raw-v1','256MB',1)
            try:
                self.assertEqual(cache.get('x','SELECT 1 value').value.iloc[0],1)
                cache.get('x','SELECT 1 value')
                self.assertEqual(cache.hits,['x'])
                self.assertEqual(cache.get('x','SELECT 2 value').value.iloc[0],2)
            finally:cache.db.close()
            with self.assertRaisesRegex(ValueError,'Raw inventory changed'):
                QueryCache(Path(temp),'raw-v2','256MB',1)

    def test_full_scan_duplicate_candidates_lists_and_linked_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'data'
            def write(relative,frame):
                path=data/relative;path.mkdir(parents=True,exist_ok=True)
                pq.write_table(pa.Table.from_pandas(frame,preserve_index=False),path/'a.parquet')
            clients=pd.DataFrame({'client_id':['a','b','c','d'],'fold':[0,1,0,1]})
            write('client_split',clients)
            targets=pd.DataFrame({'client_id':['a','b','c','d']*2,'fold':[0,1,0,1]*2,
                'mon':['2022-02-28']*4+['2022-03-31']*4,'trans_count':[0,1,2,3]*2,
                'diff_trans_date':pd.array([None,1,2,None]*2,dtype='Int32')})
            for i in range(1,5):targets['target_'+str(i)]=[0,1,0,0]*2
            write('targets',targets)
            trx=pd.DataFrame({'client_id':['a','a','a','b','c'],'fold':[0,0,0,1,0],
                'event_time':pd.to_datetime(['2021-01-01','2021-01-01','2021-01-01','2022-01-01','2022-01-02']),
                'amount':[-5.,-5.,2.,0.,1e12],'event_type':[1,1,1,2,2],'event_subtype':[1]*5,
                'currency':pd.array([1,1,1,2,None],dtype='Int32')})
            for field in ['src_type11','src_type12','dst_type11','dst_type12','src_type21','src_type22','src_type31','src_type32']:
                trx[field]=pd.array([1,1,1,2,None],dtype='Int32')
            trx['amount']=trx.amount.astype('float32')
            write('detail/trx',trx)
            geo=pd.DataFrame({'client_id':['a','b'],'fold':[0,1],
                'event_time':pd.to_datetime(['2021-12-15','2022-01-02']),
                'geohash_4':[1,2],'geohash_5':[1,2],'geohash_6':[1,2]})
            dialog=pd.DataFrame({'client_id':['b'],'fold':[1],
                'event_time':pd.to_datetime(['2021-02-01']),'embedding':[[1.,2.,3.]]})
            write('detail/geo',geo);write('detail/dialog',dialog)
            for source,frame in [('trx',trx),('geo',geo),('dialog',dialog)]:
                records=[]
                for client,part in frame.groupby('client_id'):
                    row={'client_id':client,'fold':int(part.fold.iloc[0])}
                    for col in part:
                        if col not in ('client_id','fold'):
                            row[col]=part[col].astype('int64').tolist() if col=='event_time' else part[col].to_numpy().tolist()
                    records.append(row)
                write('ptls/'+source,pd.DataFrame(records))
            records,schemas,fingerprint=source_manifest(data)
            cache=QueryCache(root/'cache',fingerprint,'512MB',1)
            try:
                open_views(cache,data);tables=scan(cache,schemas)
                d=tables['duplicates'].set_index(['source','kind'])
                self.assertEqual(d.loc[('trx','full_row'),'duplicate_n'],1)
                self.assertEqual(d.loc[('trx','same_client_time_candidate'),'duplicate_n'],2)
                self.assertEqual(tables['amount_raw_bins'].n.sum(),len(trx))
                self.assertEqual(tables['amount_outliers_currency'].finite_n.sum(),len(trx))
                cube,months=make_cube(clients,tables)
                self.assertEqual(int(cube.sum()),8)
                self.assertEqual(months[0],'2021-01')
                totals=cube.sum(axis=(1,2));masks=np.array([3,7,1,0])
                sample,_,_=stratified_sample(clients,totals,masks,4,42)
                inventory=build_sample(cache,sample,root/'sample',42,fingerprint)
                self.assertEqual(sum(r['rows'] for r in inventory if r['source']=='trx'),len(trx))
                for name in ['clients','targets','trx','geo','dialog','ptls_trx','ptls_geo','ptls_dialog']:
                    ids=pd.read_parquet(root/'sample'/(name+'.parquet')).client_id
                    self.assertTrue(ids.isin(sample.client_id).all())
                from mbd_eda_review_report import write_review
                args=argparse.Namespace(output_dir=root/'report',cache_dir=root/'cache',sample_dir=root/'sample',seed=42)
                write_review(args,cache,clients,cube,months,tables,sample,inventory,records,schemas,fingerprint,np.array([1,2]))
                amount_csv=pd.read_csv(root/'report/tables/review/amount_full.csv')
                self.assertEqual(float(amount_csv['max'].iloc[0]),float(trx.amount.max()))
                grid=pd.read_parquet(root/'sample/customer_month.parquet')
                self.assertEqual(len(grid),len(clients)*len(months))
                self.assertTrue(grid.loc[(grid.client_id=='c')&(grid.month=='2021-01'),'geo_events'].isna().all())
                self.assertEqual(len(pd.read_csv(root/'report/tables/review/missing_values.csv')),len(tables['missing_values']))
                self.assertTrue((root/'report/report.md').exists())
            finally:cache.db.close()


if __name__=='__main__':unittest.main()
