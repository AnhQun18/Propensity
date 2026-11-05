import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
try:
    import duckdb
except ImportError:
    duckdb = None

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT/'deliverables/mbd_mini_eda_fixes'
sys.path.insert(0,str(SCRIPTS))
import repair_mbd_export as fix
import complete_mbd_statistics as complete
spec=importlib.util.spec_from_file_location('original_fixture',ROOT/'tests/test_mbd_mini_eda.py')
original=importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
from mbd_mini_eda import run_analysis,export_report


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


@unittest.skipIf(duckdb is None, 'Cài extra supplement với Python >=3.11 để kiểm thử bổ sung raw')
class ExportFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=original.MBDMiniTests('test_full_pipeline_exact_moments_duplicates_and_labels')
        cls.fixture.setUp()
        for p in cls.fixture.data.rglob('*.parquet'):
            table=pq.ParquetFile(p).read().replace_schema_metadata(None)
            pq.write_table(table,p)
        cls.config=replace(cls.fixture.config,duplicate_mode='sample',sample_size=2)
        cls.source=export_report(run_analysis(cls.config),plots=True)
        cls.raw_before=hashes(cls.fixture.data)
        cls.export_before=hashes(cls.source)
        cls.output=cls.fixture.root/'completed'
        config=cls.fixture.root/'relative_config.json'
        config.write_text(json.dumps({'data_root':'data'}),encoding='utf-8')
        env=os.environ.copy(); env['PYTHONUTF8']='1'; env.pop('PYTHONPATH',None)
        result=subprocess.run([sys.executable,str(SCRIPTS/'complete_mbd_statistics.py'),
            '--export',str(cls.source),'--output-dir',str(cls.output),'--config',str(config),
            '--memory-limit','512MB','--threads','1','--temp-limit','1GB'],
            cwd=cls.fixture.root,env=env,capture_output=True,text=True,encoding='utf-8')
        if result.returncode:
            raise AssertionError(result.stdout+'\n'+result.stderr)
        print(result.stdout,flush=True)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDown()

    def test_full_quantiles_histograms_and_currency(self):
        summary=fix.read_table(self.output,'descriptive.transactions','amount_summary').iloc[0]
        raw=self.fixture.trx.amount.replace([np.inf,-np.inf],np.nan).dropna()
        for column,q in [('p25',.25),('median',.5),('p75',.75),('p95',.95),('p99',.99)]:
            self.assertAlmostEqual(summary[column],raw.quantile(q),places=8)
        self.assertAlmostEqual(summary['mean'],raw.mean())
        self.assertAlmostEqual(summary['std'],raw.std(ddof=1))
        self.assertEqual(summary.quantile_scope,complete.FULL)
        for name in ('amount_histogram','amount_log_histogram'):
            self.assertEqual(fix.read_table(self.output,'descriptive.transactions',name).n.sum(),len(raw))
        grouped=fix.read_table(self.output,'descriptive.transactions','amount_by_currency')
        self.assertEqual(grouped.rows.sum(),len(self.fixture.trx))

    def test_full_frequencies_and_duplicates(self):
        full=fix.read_table(self.output,'descriptive.transactions','transaction_categories_full')
        currency=full[full.column.eq('currency')]
        self.assertEqual(set(currency[currency.kind.eq('value')].value),{'1','2'})
        self.assertEqual(currency.n.sum(),len(self.fixture.trx))
        self.assertEqual(currency.distinct_n.unique().tolist(),[2])
        dups=fix.read_table(self.output,'descriptive.quality','duplicates')
        trx=dups[dups.source.eq('transactions') & dups.kind.eq('full_row')].iloc[0]
        self.assertEqual(trx.duplicate_n,1)
        self.assertEqual(trx.scope,'all_scanned_rows')

    def test_customer_dates_manifest_and_immutable_inputs(self):
        dates=fix.read_table(self.output,'descriptive.customers','first_last_event_month')
        expected={'transactions':3,'geo':2,'dialog':3}
        for source,n in expected.items():
            for milestone in ('first','last'):
                self.assertEqual(dates.loc[dates.source.eq(source)&dates.milestone.eq(milestone),'customers'].sum(),n)
        manifest=json.loads((self.output/'manifest.json').read_text(encoding='utf-8'))
        for item in manifest['tables']:
            frame=pd.read_csv(self.output/item['path'])
            self.assertEqual(len(frame),item['rows'])
            self.assertNotIn('client_id',frame)
        self.assertIn('figures/amount_log_histogram.png',manifest['figures'])
        self.assertIn('figures/first_event_month.png',manifest['figures'])
        metadata=json.loads((self.output/'metadata.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['raw_supplements']['status'],'completed')
        self.assertEqual(self.raw_before,hashes(self.fixture.data))
        self.assertEqual(self.export_before,hashes(self.source))

    def test_exact_nan_lists_and_nanosecond_timestamps(self):
        with duckdb.connect() as db:
            times=pa.array([1,1,2],type=pa.timestamp('ns'))
            table=pa.table({'client_id':['a','a','a'],'event_time':times,
                           'embedding':pa.array([[1.,None],[1.,np.nan],[1.,None]],type=pa.list_(pa.float64()))})
            db.register('src_dialog',table)
            descriptions={'dialog':dict(schema=table.schema,columns=table.column_names,rows=3)}
            result=complete.full_duplicates(db,self.source,descriptions)
            row=result['descriptive.quality','duplicates'].query("source=='dialog' and kind=='full_row'").iloc[0]
            self.assertEqual(row.duplicate_n,1)

    def test_repair_old_integer_float_category_export(self):
        root=self.fixture.root/'old_category_export'
        _,metadata=fix.prepare_export(self.source,root)
        frame=fix.read_table(root,'descriptive.transactions','transaction_categories')
        mask=frame.column.eq('currency') & frame.kind.eq('value') & frame.value.eq('1')
        original=int(frame.loc[mask,'n'].iloc[0])
        extra=frame.loc[mask].iloc[0].copy()
        extra['value']='1.0'; extra['n']=1
        frame.loc[mask,'n']=original-1
        frame=pd.concat([frame,pd.DataFrame([extra])],ignore_index=True)
        fix.write_table(root,'descriptive.transactions','transaction_categories',frame)
        fix.repair_categories(root,metadata)
        repaired=fix.read_table(root,'descriptive.transactions','transaction_categories')
        currency=repaired[repaired.column.eq('currency')]
        self.assertEqual(set(currency[currency.kind.eq('value')].value),{'1','2'})
        self.assertEqual(int(currency.loc[currency.value.eq('1'),'n'].sum()),original)
        self.assertEqual(int(currency.n.sum()),len(self.fixture.trx))

    def test_skip_duplicates_and_protected_paths(self):
        args=argparse.Namespace(export=self.source,output_dir=self.fixture.root/'skipped',config=None,
            data_root=self.fixture.data,work_dir=None,memory_limit='512MB',temp_limit='1GB',threads=1,
            top_categories=2,skip_duplicates=True)
        result=complete.run(args)
        metadata=json.loads((result/'metadata.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['raw_supplements']['status'],'completed_except_duplicates')
        dups=fix.read_table(result,'descriptive.quality','duplicates')
        self.assertTrue(dups.loc[dups.kind.eq('full_row'),'scope'].eq('uniform_row_sample').all())
        bad=argparse.Namespace(**{**vars(args),'output_dir':self.fixture.data/'bad_export'})
        with self.assertRaisesRegex(ValueError,'raw'):
            complete.run(bad)
        self.assertFalse(bad.output_dir.exists())
        with self.assertRaisesRegex(ValueError,'tách'):
            fix.repair(self.source,self.source/'bad')

    def test_empty_and_signed_amounts(self):
        for values in ([],[-10.,0.,10.,float('inf'),float('nan'),None]):
            with duckdb.connect() as db:
                table=pa.table({'amount':pa.array(values,type=pa.float64()),
                                'currency':pa.array([1]*len(values),type=pa.int32())})
                db.register('src_transactions',table)
                result=complete.full_amounts(db,{'transactions':{'rows':len(values)}})
                summary=result['descriptive.transactions','amount_summary'].iloc[0]
                self.assertEqual(summary.finite_n,0 if not values else 3)
                self.assertEqual(result['descriptive.transactions','amount_log_histogram'].n.sum(),summary.finite_n)
                self.assertIn('currency',result['descriptive.transactions','amount_by_currency'])


if __name__=='__main__':
    unittest.main()
