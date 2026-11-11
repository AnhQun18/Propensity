"""Check that EDA distinguishes duplicates, simultaneous events and missing values."""
import sys
import shutil
import uuid
import unittest
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from propensity_eda import EDA, log_iqr_summary


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


class LogIQRTest(unittest.TestCase):
    def setUp(self): self.db=duckdb.connect()
    def tearDown(self): self.db.close()

    def test_both_tails_exact_transform_and_currency_isolation(self):
        values=np.r_[np.exp(np.random.default_rng(42).normal(10,1,1000)),0.,1e9,-5.,np.nan,np.inf]
        frame=pd.DataFrame({"currency":[11]*len(values)+[3]*100,"amount":np.r_[values,np.linspace(1,100,100)]})
        self.db.register("data",frame)
        result=log_iqr_summary(self.db,"data","amount",group="currency",include_ln=True).set_index("group_value")
        r=result.loc[11]
        transformed=np.log1p(values[np.isfinite(values)&(values>=0)])
        q1,q3=np.quantile(transformed,[.25,.75]);lo,hi=q1-1.5*(q3-q1),q3+1.5*(q3-q1)
        self.assertEqual(r.log_valid_n,len(transformed))
        self.assertAlmostEqual(r.log_q1,q1)
        self.assertAlmostEqual(r.iqr_upper,hi)
        self.assertEqual(r.log_below_n,int((transformed<lo).sum()))
        self.assertEqual(r.log_above_n,int((transformed>hi).sum()))
        self.assertGreater(r.log_below_n,0);self.assertGreater(r.log_above_n,0)
        self.assertEqual(r.iqr_n,r.log_below_n+r.log_above_n)
        self.assertEqual(r.negative_n,1);self.assertEqual(r.zero_n,1)
        self.assertEqual(r.ln_valid_n,len(transformed)-1)
        self.assertAlmostEqual(r.iqr_upper_input_units,np.expm1(hi))

    def test_small_unknown_and_zero_iqr_groups_have_no_primary_label(self):
        frame=pd.DataFrame({"currency":[11]*5+[3]*100+[None]*50,
                            "amount":list(range(5))+[1.]*99+[1e8]+list(range(50))})
        self.db.register("data",frame)
        result=log_iqr_summary(self.db,"data","amount",group="currency")
        self.assertFalse(result.iqr_applicable.any())
        self.assertTrue(result.iqr_n.isna().all())

    def test_quantiles_are_computed_after_log_not_before(self):
        values=np.array([1.,2.,8.,32.])
        self.db.register("data",pd.DataFrame({"x":values}))
        r=log_iqr_summary(self.db,"data","x",min_n=1).iloc[0]
        self.assertAlmostEqual(r.log_q1,np.quantile(np.log1p(values),.25))
        self.assertNotAlmostEqual(r.log_q1,np.log1p(np.quantile(values,.25)),places=5)

    def test_signed_values_and_no_valid_log_domain(self):
        self.db.register("data",pd.DataFrame({"x":np.linspace(-100,100,101)}))
        signed=log_iqr_summary(self.db,"data","x",transform="signed_log1p").iloc[0]
        self.assertTrue(signed.iqr_applicable)
        self.assertAlmostEqual(signed.log_median,0.)
        self.db.register("negative",pd.DataFrame({"x":-np.arange(1.,51.)}))
        no_domain=log_iqr_summary(self.db,"negative","x",include_ln=True).iloc[0]
        self.assertEqual(no_domain.log_valid_n,0)
        self.assertFalse(no_domain.iqr_applicable)


if __name__ == "__main__":
    unittest.main()
