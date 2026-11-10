"""Calendar boundaries and real CatBoost stage isolation on a tiny fixture."""
from contextlib import redirect_stdout
import ast
from dataclasses import replace
from datetime import datetime
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import nbformat

import test_propensity_data as prep
from propensity_baseline.config import BaselineConfig, VARIANTS
from propensity_baseline.data import build_calendar, previous_month, read_json
from propensity_baseline.metrics import evaluate, paired_recall_bootstrap, ranked_indices


class MetricsTests(unittest.TestCase):
    def test_topk_known_answer_and_ties_independent_of_row_order(self):
        metric = evaluate(np.array([1,0,1,0]), np.array([.9,.8,.1,0]), ['a','b','c','d'], seed=3, budgets=(.5,))
        self.assertEqual(metric['top_k']['0.5']['hits'], 1)
        self.assertEqual(metric['top_k']['0.5']['recall'], .5)
        self.assertEqual(metric['top_k']['0.5']['lift'], 1.)
        ids = np.array(['d','b','a','c'])
        first = ids[ranked_indices(ids, np.zeros(4), 7)]
        second = ids[::-1][ranked_indices(ids[::-1], np.zeros(4), 7)]
        np.testing.assert_array_equal(first, second)
        self.assertEqual(previous_month('2022-01').isoformat(), '2021-12-01')
        self.assertEqual(previous_month('2022-03').isoformat(), '2022-02-01')

    def test_paired_bootstrap_same_model_has_zero_difference(self):
        frame = pd.DataFrame([dict(client_id=str(i), month=m, model=model, y_any=i%2, score=float(i))
                              for m in ['2022-09','2022-10'] for model in ['a','b'] for i in range(10)])
        result = paired_recall_bootstrap(frame, 'a', 'b', seed=2, replicates=20)
        self.assertEqual(result['estimate'], 0.)
        self.assertEqual(result['interval_95'], [0.,0.])
        with self.assertRaises(ValueError):
            evaluate([0,1], [.2,float('inf')], ['a','b'], seed=1)


class BaselineNotebookTests(unittest.TestCase):
    def test_notebook_is_valid_unexecuted_and_default_test_guard_is_closed(self):
        path = Path(__file__).resolve().parents[1] / 'notebooks/05_calendar_baseline.ipynb'
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        guards = []
        for cell in notebook.cells:
            if cell.cell_type != 'code':
                continue
            self.assertIsNone(cell.execution_count)
            self.assertEqual(cell.outputs, [])
            tree = ast.parse(cell.source)
            for statement in tree.body:
                if isinstance(statement, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OPEN_FINAL_TEST' for t in statement.targets):
                    guards.append(ast.literal_eval(statement.value))
        self.assertEqual(guards,[False])


class CalendarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = prep.PreparationTests('test_nonbinary_label_fails')
        cls.fixture.setUp()
        cls.addClassCleanup(cls.fixture.doCleanups)
        cls.fixture.config = replace(cls.fixture.config, include_dialog_embeddings=False)
        trx = cls.fixture.trx
        additions = [(datetime(2022,1,31,23,59,59),5.), (datetime(2022,2,1),7.),
                     (datetime(2022,2,28,23,59,59),11.), (datetime(2022,3,1),13.),
                     (datetime(2022,10,31,23,59,59),17.)]
        for at, amount in additions:
            for column in trx:
                value = ('a' if column=='client_id' else at if column=='event_time'
                         else amount if column=='amount' else 11 if column=='currency' else 2)
                trx[column].append(value)
        cls.fixture.write('detail/trx', trx)
        with redirect_stdout(io.StringIO()):
            cls.fixture.prepare()
            cls.calendar = cls.fixture.config.output_dir.parent / 'calendar'
            build_calendar(cls.fixture.config.output_dir, cls.calendar)

    def frame(self, month, root=None):
        return pq.ParquetFile((root or self.calendar) / f'features/month={month}/part.parquet').read().to_pandas().set_index('client_id')

    def test_disjoint_calendar_windows_and_nonlinear_older_aggregates(self):
        feb, march, december = (self.frame(m) for m in ['2022-02','2022-03','2022-12'])
        self.assertEqual(feb.loc['a','trx_recent_event_count'],2)
        self.assertEqual(feb.loc['a','trx_recent_source_window_overlap_days'],31)
        self.assertEqual(march.loc['a','trx_recent_event_count'],2)
        self.assertEqual(march.loc['a','trx_recent_source_window_overlap_days'],28)
        self.assertEqual(march.loc['a','trx_older_event_count'],2)
        self.assertEqual(march.loc['a','trx_all_event_count'],4)
        self.assertEqual(december.loc['a','trx_recent_event_count'],2)
        self.assertEqual(december.loc['a','trx_older_event_count'],6)
        self.assertEqual(december.loc['a','trx_older_main_amount_median'],10.5)
        self.assertTrue(pd.isna(march.loc['a','dialog_older_event_count']))
        self.assertEqual(march.loc['b','cold_start'],1)
        self.assertTrue(pd.isna(march.loc['b','y_any']))
        schema = read_json(self.calendar / 'feature_schema.json')
        self.assertEqual(len(schema['predictors']),230)
        for variant, features in schema['variants'].items():
            self.assertNotIn('cold_start',features)
            self.assertFalse(any('_all_' in c or 'target_' in c for c in features))
            if variant.endswith('recent'):
                self.assertFalse(any('_older_' in c or 'difference' in c for c in features))

    def test_cached_cutoff_aggregates_equal_direct_calendar_aggregation(self):
        direct=self.calendar.parent/'calendar_direct'
        with redirect_stdout(io.StringIO()):
            build_calendar(self.fixture.config.output_dir,direct,reuse_prepared_aggregates=False)
        self.assertTrue(read_json(self.calendar/'manifest.json')['cache_reuse_enabled'])
        for month in prep.MONTHS:
            pd.testing.assert_frame_equal(self.frame(month),self.frame(month,direct))

    def test_future_cache_event_values_do_not_change_past_calendar_features(self):
        import duckdb
        before = self.frame('2022-03')
        database = self.fixture.config.output_dir / 'work.duckdb'
        with duckdb.connect(str(database)) as con:
            con.execute("UPDATE trx SET amount=987654321 WHERE event_time>=TIMESTAMP '2022-03-01'")
        second = self.calendar.parent / 'calendar_future_changed'
        with redirect_stdout(io.StringIO()):
            build_calendar(self.fixture.config.output_dir, second)
        pd.testing.assert_frame_equal(before,self.frame('2022-03',second))
        with self.assertRaises(FileExistsError):
            build_calendar(self.fixture.config.output_dir,self.calendar)

    @unittest.skipUnless(importlib.util.find_spec('catboost') and importlib.util.find_spec('sklearn'), 'Install .[baseline]')
    def test_real_model_stages_keep_holdout_locked_and_preserve_unknown_predictions(self):
        from propensity_baseline import runner
        config = BaselineConfig(self.calendar,self.calendar.parent/'models',depths=(2,),iterations=(12,),
                                variants=VARIANTS,threads=1,bootstrap_replicates=5)
        seen = []
        original = runner.read_months
        def tracked(config,months,*args):
            seen.extend(months)
            return original(config,months,*args)
        with patch.object(runner,'read_months',side_effect=tracked), redirect_stdout(io.StringIO()):
            runner.tune(config)
            self.assertNotIn('2022-12',seen)
            self.assertNotIn('2023-01',seen)
            from propensity_baseline.persistence import compare_rule
            comparison_dir = compare_rule(config)
            comparison = read_json(comparison_dir/'monthly_comparison.json')
            self.assertEqual(len(comparison),12)
            self.assertTrue(all(r['comparison_rows']==2 for r in comparison))
            self.assertFalse(read_json(comparison_dir/'manifest.json')['test_opened'])
            with self.assertRaises(ValueError):
                runner.test_final(config)
            selection = config.output_dir/'selection.json'
            saved = selection.read_bytes()
            altered = json.loads(saved)
            altered['parameters']['iterations']=13
            selection.write_text(json.dumps(altered),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Selection changed'):
                runner.fit_final(config)
            selection.write_bytes(saved)
            runner.fit_final(config)
            self.assertNotIn('2023-01',seen)
            self.assertFalse(read_json(config.output_dir/'manifest.json')['test_opened'])
            runner.test_final(config)
            test_comparison_dir=compare_rule(config,stage='test')
            self.assertTrue(read_json(test_comparison_dir/'manifest.json')['test_opened'])
        self.assertIn('2023-01',seen)
        self.assertEqual(read_json(config.output_dir/'manifest.json')['status'],'tested')
        self.assertEqual(len(read_json(config.output_dir/'ablation_comparisons.json')),4)
        prediction = pd.read_parquet(config.output_dir/'test/predictions.parquet')
        self.assertEqual(len(prediction),6)  # model + constant, all 3 scoring IDs
        self.assertEqual(prediction.y_any.isna().sum(),2)
        self.assertTrue(prediction.p_any.between(0,1).all())
        with self.assertRaises(FileExistsError):
            runner.tune(replace(config,iterations=(13,)))  # no overwrite
        with self.assertRaises(ValueError):
            runner.test_final(config)  # no second test opening


if __name__=='__main__':
    unittest.main()
