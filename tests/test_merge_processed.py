"""Verify merge and notebook 04 using only a small synthetic prepared dataset."""
import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import nbformat
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import test_propensity_data as preparation_tests
from propensity_data.merge import merge_prepared

ROOT = Path(__file__).resolve().parents[1]


class MergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp_base = ROOT / 'outputs/merge_notebook_tests'
        temp_base.mkdir(parents=True, exist_ok=True)
        cls.fixture = preparation_tests.PreparationTests('test_nonbinary_label_fails')
        with patch.object(tempfile, 'tempdir', str(temp_base)):
            cls.fixture.setUp()
        cls.addClassCleanup(cls.fixture.doCleanups)
        with redirect_stdout(io.StringIO()):
            cls.fixture.prepare()
        cls.source = cls.fixture.config.output_dir
        cls.root = cls.source.parent

    def test_notebook_merge_preserves_every_column_row_and_unknown_label(self):
        notebook = nbformat.read(ROOT / 'notebooks/04_merge_processed.ipynb', as_version=4)
        nbformat.validate(notebook)
        spec = importlib.util.spec_from_file_location('sync_merge', ROOT / 'scripts/sync_preparation_notebook.py')
        sync = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sync)
        out = self.root / 'notebook_merged'
        scope = dict(__name__=__name__, SOURCE_DIR=self.source, OUTPUT_DIR=out, BATCH_SIZE=2,
                     Path=Path, json=json, pd=pd, pq=pq, display=lambda *args: None)
        for cell in notebook.cells:
            if cell.cell_type != 'code':
                continue
            self.assertIsNone(cell.execution_count)
            self.assertEqual(cell.outputs, [])
            ast.parse(cell.source)
            if 'source_file' in cell.metadata:
                self.assertEqual(cell.source, sync.render_library_cell(ROOT, cell.metadata))
            if cell.id in ('merge-environment', 'merge-settings'):
                continue
            with redirect_stdout(io.StringIO()):
                exec(compile(cell.source, '<' + cell.id + '>', 'exec'), scope)
        actual = pq.ParquetFile(out / 'propensity.parquet').read()
        expected = pa.concat_tables([pq.ParquetFile(self.source / f'features/month={m}/part.parquet').read().cast(actual.schema)
                                     for m in preparation_tests.MONTHS])
        self.assertTrue(actual.equals(expected, check_metadata=False))
        manifest = json.loads((out / 'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'ready')
        self.assertEqual(manifest['rows'], 36)
        self.assertEqual(manifest['unique_customer_months'], 36)
        self.assertEqual(manifest['unknown_labels'], 12)
        self.assertEqual(manifest['positives'], 12)
        self.assertFalse((out / 'propensity.incomplete.parquet').exists())
        with self.assertRaises(FileExistsError):
            merge_prepared(self.source, out)

    def test_unready_source_and_overlap_do_not_create_output(self):
        source = self.root / 'unready'
        source.mkdir()
        (source / 'manifest.json').write_text(json.dumps({'status': 'running'}))
        out = self.root / 'unready_output'
        with self.assertRaisesRegex(ValueError, 'must be ready'):
            merge_prepared(source, out)
        self.assertFalse(out.exists())
        with self.assertRaisesRegex(ValueError, 'overlap'):
            merge_prepared(self.source, self.source / 'merged')

    def test_duplicate_keys_leave_failed_manifest_without_final_dataset(self):
        source = self.root / 'duplicate_source'
        shutil.copytree(self.source, source)
        file = source / 'features/month=2022-02/part.parquet'
        table = pq.ParquetFile(file).read()
        pq.write_table(pa.concat_tables([table, table.slice(0, 1)]), file)
        out = self.root / 'duplicate_output'
        with self.assertRaisesRegex(ValueError, 'Duplicate customer-month'):
            merge_prepared(source, out)
        self.assertEqual(json.loads((out / 'manifest.json').read_text())['status'], 'failed')
        self.assertFalse((out / 'propensity.parquet').exists())


if __name__ == '__main__':
    unittest.main()
