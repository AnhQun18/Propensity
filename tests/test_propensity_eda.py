"""Check that EDA distinguishes duplicates, simultaneous events and missing values."""
import sys
import shutil
import uuid
import unittest
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from propensity_eda import EDA


class EDAQualityTest(unittest.TestCase):
    def setUp(self):
        e = self.eda = EDA.__new__(EDA)
        self.test_root = (Path(__file__).resolve().parents[1] / "outputs/eda_quality_tests").resolve()
        e.root = self.test_root / uuid.uuid4().hex
        (e.root / "outputs").mkdir(parents=True)
        e.out = e.root / "report"
        (e.out / "tables").mkdir(parents=True)
        e.db = duckdb.connect()
        e.paths = dict.fromkeys(["clients","targets","trx","geo","dialog","ptls_trx","ptls_geo","ptls_dialog"])
        e.validation = {}
        e.tables = {"feature_statistics_train": pd.DataFrame()}
        statements = [
            "CREATE TABLE clients AS SELECT 'a' client_id,0 fold UNION ALL SELECT 'b',0",
            "CREATE TABLE targets AS SELECT 'a' client_id,'2022-02-28' mon,0 trans_count,NULL::INTEGER diff_trans_date,0 fold UNION ALL SELECT 'a','2022-02-28',0,NULL,0 UNION ALL SELECT 'b','2022-02-28',1,7,0",
            "CREATE TABLE trx AS SELECT 'a' client_id,TIMESTAMP '2022-01-01' event_time,10.0 amount,11 currency,0 fold UNION ALL SELECT 'a',TIMESTAMP '2022-01-01',10.0,11,0 UNION ALL SELECT 'a',TIMESTAMP '2022-01-01',20.0,NULL,0",
            "CREATE TABLE geo AS SELECT 'a' client_id,TIMESTAMP '2022-01-01' event_time,4 geohash_4,0 fold",
            "CREATE TABLE dialog AS SELECT 'a' client_id,TIMESTAMP '2022-01-01' event_time,[1.0, 'NaN'::DOUBLE, NULL] embedding,0 fold",
            "CREATE TABLE ptls_trx AS SELECT 'a' client_id,[1,1,1] event_time,[10.,10.,20.] amount,0 fold",
            "CREATE TABLE ptls_geo AS SELECT 'a' client_id,[1] event_time,[4] geohash_4,0 fold",
            "CREATE TABLE ptls_dialog AS SELECT 'a' client_id,[1] event_time,[[1.0,'NaN'::DOUBLE,NULL]] embedding,0 fold",
            "CREATE TABLE features AS SELECT 'a' client_id,TIMESTAMP '2022-02-01' cutoff UNION ALL SELECT 'b',TIMESTAMP '2022-02-01'",
        ]
        for sql in statements:e.db.execute(sql)
        e.tables["overview"] = pd.DataFrame([{"table":s,"rows":e.db.execute(f"SELECT count(*) FROM {s}").fetchone()[0]} for s in e.paths])

    def tearDown(self):
        self.eda.close()
        resolved = self.eda.root.resolve()
        assert resolved.is_relative_to(self.test_root) and resolved != self.test_root
        shutil.rmtree(resolved)

    def test_exact_duplicates_not_same_timestamp_and_nested_missing(self):
        missing,duplicates,conflicts = self.eda.quality()
        trx = duplicates[duplicates.table == "trx"].set_index("check")
        self.assertEqual(trx.loc["full_row","extra_rows"],1)
        self.assertEqual(trx.loc["same_client_timestamp_candidate","extra_rows"],2)
        self.assertEqual(self.eda.validation["label_exact_duplicates_removed_in_temp_view"],1)
        self.assertTrue(conflicts.empty)
        self.assertEqual(missing.query("table == 'trx' and column == 'currency'").iloc[0].null_n,1)
        nested = self.eda.tables["nested_element_quality"].query("table == 'dialog'").iloc[0]
        self.assertEqual(nested.null_elements,1)
        self.assertEqual(nested.nan_elements,1)
        self.assertEqual(self.eda.db.execute("SELECT count(*) FROM targets").fetchone()[0],3)

    def test_conflicting_label_key_stops_analysis(self):
        self.eda.db.execute("INSERT INTO targets VALUES ('a','2022-02-28',1,7,0)")
        self.eda.tables["overview"].loc[self.eda.tables["overview"].table == "targets","rows"] = 4
        with self.assertRaisesRegex(ValueError,"Conflicting labels"):
            self.eda.quality()


if __name__ == "__main__":
    unittest.main()
