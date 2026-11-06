"""Nontrivial contracts: repeated keys, month cutoffs, union-days, grouping."""
import unittest
from pathlib import Path
import tempfile
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plot_behavior_coverage import month_range, update_matrix, grouped_presence, row_order, aggregate, select_heatmap_rows


class CoverageTests(unittest.TestCase):
    def test_annotated_sample_has_real_unique_customers_and_stable_seed(self):
        counts=np.zeros((50,3),dtype=np.uint32)
        order=np.arange(49,-1,-1)
        selected=select_heatmap_rows(counts,order,sample_size=25,seed=42)
        self.assertEqual(len(np.unique(selected)),25)
        self.assertTrue(((selected>=0)&(selected<50)).all())
        np.testing.assert_array_equal(selected,select_heatmap_rows(counts,order,sample_size=25,seed=42))
        np.testing.assert_array_equal(select_heatmap_rows(counts,order,sample_size=25,method='page',page_start=47),[2,1,0])
        with self.assertRaises(ValueError):select_heatmap_rows(counts,order,sample_size=101)

    def test_repeated_pairs_sources_days_and_exclusive_cutoff(self):
        months = month_range("2022-01", "2022-03")
        counts = np.zeros((3,2), dtype=np.uint64)
        masks = np.zeros((3,2), dtype=np.uint8)
        daily = np.zeros((3,59), dtype=bool)
        positions = np.array([0,0,0,1,1,-1,2,2])
        times = np.array(["2022-01-31T12:00", "2022-01-31T12:01", "2022-02-01",
                          "2021-12-31", "2022-03-01", "2022-01-01", "NaT", "2022-02-02"], dtype="datetime64[ns]")
        stats = update_matrix(counts,masks,positions,times,months,1,daily)
        update_matrix(counts,masks,np.array([0,0]),np.array(["2022-01-31", "2022-02-02"],dtype="datetime64[ns]"),months,2,daily)
        np.testing.assert_array_equal(counts,[[3,2],[0,0],[0,1]])
        np.testing.assert_array_equal(masks,[[3,3],[0,0],[0,1]])
        self.assertEqual(int(daily[0].sum()),3)  # repeated Jan 31 counts once
        self.assertEqual(stats,{"rows":8,"unknown_or_null_client":1,"null_timestamp_known_client":1,
                                "outside_range_known_valid":2,"retained_events":4})

    def test_grouping_does_not_drop_tail_or_invent_client(self):
        counts=np.array([[0,0],[1,0],[0,2],[3,3],[0,0]],dtype=np.uint32)
        order=row_order(counts)
        raster,edges=grouped_presence(counts,order,2)
        self.assertEqual(edges[-1],5)
        np.testing.assert_allclose((raster*np.diff(edges)[:,None]).sum(axis=0),(counts>0).sum(axis=0))
        self.assertEqual(order[0],3)
        self.assertEqual(set(order),set(range(5)))

    def test_parquet_stream_is_invariant_to_batch_boundaries(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root=Path(directory)
            (root/'client_split').mkdir()
            pq.write_table(pa.table({'client_id':['b','a','no_log']}),root/'client_split/part.parquet')
            fixtures={
                'trx': [('a','2022-01-31'),('a','2022-01-31'),('a','2022-02-01'),('unknown','2022-01-01')],
                'geo': [('a','2022-01-31'),('b','2022-02-01')],
                'dialog': [('a','2022-02-01'),('b',None),('b','2022-03-01')],
            }
            for source,events in fixtures.items():
                path=root/'detail'/source;path.mkdir(parents=True)
                pq.write_table(pa.table({'client_id':[e[0] for e in events],
                    'event_time':pa.array(np.array([e[1] or 'NaT' for e in events],dtype='datetime64[ns]'))}),path/'part.parquet')
            small=aggregate(root,'2022-01','2022-03',batch_size=1,active_days=True)
            large=aggregate(root,'2022-01','2022-03',batch_size=100,active_days=True)
            for a,b in zip(small[:5],large[:5]):np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(small[0],[[3,2],[0,1],[0,0]])
            np.testing.assert_array_equal(small[1],[[3,5],[0,2],[0,0]])
            np.testing.assert_array_equal(small[2],[[1,1],[0,1],[0,0]])


if __name__=="__main__":unittest.main()
