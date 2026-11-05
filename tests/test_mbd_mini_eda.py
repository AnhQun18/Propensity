"""Đối chiếu với dữ liệu tự tính, cutoff, partition và thống kê xuyên batch."""

from dataclasses import replace
import hashlib
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

from mbd_mini_eda import MBDMiniConfig, run_analysis, load_config, build_plan
from mbd_mini_eda.io import iter_batches, source_paths, read_reference_tables, validate_label_windows
from mbd_mini_eda.models import AnalysisContext
from mbd_mini_eda.features import customers, customer_month
from mbd_mini_eda.reporting import export_report
from mbd_mini_eda.statistics import UniformSample


class MBDMiniTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.folds = {"a": 0, "b": 1, "c": 1, "d": 0, "z": 0}
        clients = pd.DataFrame({"client_id": ["a", "b", "c", "d"], "fold": [0, 1, 1, 0]})
        self.write("client_split", clients)
        rows = []
        for mon in ("2022-02-28", "2022-03-31", "2022-04-30"):
            for client in clients.client_id:
                rows.append(dict(client_id=client, fold=self.folds[client], mon=mon,
                    target_1=int((client, mon) in {("a", "2022-02-28"), ("b", "2022-03-31")}),
                    target_2=np.nan if (client, mon)==("c","2022-03-31") else int((client,mon)==("b","2022-03-31")),
                    target_3=int((client,mon)==("c","2022-04-30")), target_4=int((client,mon)==("a","2022-02-28")),
                    trans_count=999, diff_trans_date=np.nan))
        self.labels = pd.DataFrame(rows)
        self.write("targets", self.labels)
        self.trx = pd.DataFrame({
            "client_id": ["a","a","b","a","a","b","c","z","a",None,"c"],
            "event_time": pd.to_datetime(["2022-01-01 12:00","2022-01-10 12:00","2022-01-31 23:59",
                "2022-02-01 00:00","2022-02-20 12:00","2022-03-01 00:00","2022-04-01 00:00",
                "2022-01-05 12:00","2022-01-10 12:00","2022-01-05 12:00",None], format="mixed"),
            "amount": [10.,20.,0.,100.,30.,np.nan,np.inf,5.,20.,1.,2.],
            "event_type": pd.Series([1,1,2,1,2,2,1,1,1,1,1],dtype="Int64"),
            "currency": pd.Series([1,1,2,1,None,2,1,1,1,1,1],dtype="Int64"),
        })
        self.trx["fold"] = self.trx.client_id.map(self.folds).fillna(0).astype(int)
        self.write("detail/trx", self.trx)
        geo = pd.DataFrame({"client_id":["a","a","a","b","z"],
            "event_time":pd.to_datetime(["2022-01-09","2022-01-10","2022-02-01","2022-01-31","2022-01-15"]),
            "geohash_4":pd.Series([1,1,2,3,9],dtype="Int64"),
            "geohash_5":pd.Series([1,1,2,3,9],dtype="Int64"),
            "geohash_6":pd.Series([10,10,11,20,99],dtype="Int64")})
        geo["fold"] = geo.client_id.map(self.folds)
        self.write("detail/geo",geo)
        dialog = pd.DataFrame({"client_id":["a","b","c","c"],
            "event_time":pd.to_datetime(["2022-01-20","2022-02-12","2022-02-01","2022-03-01"]),
            "embedding":[[1.,2.,3.],[0.,0.,1.],[1.,2.],[1.,np.nan,3.]]})
        dialog["fold"] = dialog.client_id.map(self.folds)
        self.write("detail/dialog",dialog)
        self.config = MBDMiniConfig(data_root=self.data,output_dir=self.root/"report",work_dir=self.root/"work",
            batch_size=2,history_days=(30,84),sample_size=100,embedding_size=3,
            duplicate_mode="exact",label_alignment="calendar_month")

    def tearDown(self):
        self.temp.cleanup()

    def write(self,name,frame):
        for fold,group in frame.groupby("fold"):
            directory=self.data/name/f"fold={fold}"
            directory.mkdir(parents=True,exist_ok=True)
            group=group.drop(columns="fold")
            middle=max(1,len(group)//2)
            for index,part in enumerate((group.iloc[:middle],group.iloc[middle:])):
                if len(part):
                    pq.write_table(pa.Table.from_pandas(part,preserve_index=False),directory/f"part-{index}.parquet")

    def context(self,config=None):
        config=config or self.config
        context=AnalysisContext(config,source_paths(config.data_root))
        context.artifacts.update(read_reference_tables(context.sources))
        months=pd.to_datetime(sorted(self.labels.mon.unique()))
        starts=months.to_period("M").to_timestamp()
        context.artifacts["label_windows"]=pd.DataFrame({"mon":months.strftime("%Y-%m-%d"),
            "cutoff":starts,"label_start":starts,"label_end":starts+pd.offsets.MonthBegin(1)})
        return context

    def test_partition_projection_and_cross_shard_schema(self):
        batches=list(iter_batches(self.data/"detail/trx",columns=["client_id","fold"],batch_size=1))
        frame=pd.concat(batches,ignore_index=True)
        self.assertEqual(len(frame),len(self.trx))
        self.assertEqual(list(frame),["client_id","fold"])
        for r in frame.dropna().itertuples(index=False):
            self.assertEqual(r.fold,self.folds[r.client_id])
        broken=self.root/"bad"
        broken.mkdir()
        pd.DataFrame({"x":[1]}).to_parquet(broken/"a.parquet")
        pd.DataFrame({"y":[1]}).to_parquet(broken/"b.parquet")
        with self.assertRaisesRegex(ValueError,"Schema"):
            list(iter_batches(broken))

    def test_cutoff_history_cardinality_and_missing_features(self):
        with self.context() as context:
            customers.run(context)
            customer_month.run(context)
            frame=context.artifacts["customer_month_features"]
            self.assertEqual(len(frame),len(self.labels))
            self.assertFalse(frame.duplicated(["client_id","mon"]).any())
            a=frame[frame.client_id.eq("a") & frame.mon.eq("2022-02-28")].iloc[0]
            self.assertEqual(a.trx_event_count_30d,2)
            self.assertEqual(a.trx_amount_sum_30d,40)
            self.assertEqual(a.trx_event_count_84d,3)
            self.assertEqual(a.geo_unique_locations_30d,1)
            self.assertEqual(a.trx_active_days_30d,1)
            self.assertEqual(a.any_active_days_30d,3)
            self.assertLess(a.trx_last_event_30d,a.cutoff)
            d=frame[frame.client_id.eq("d")].iloc[0]
            self.assertEqual(d.trx_event_count_30d,0)
            self.assertTrue(pd.isna(d.trx_amount_mean_30d))
            self.assertTrue(pd.isna(d.trx_recency_days_30d))
            self.assertNotIn("trans_count",frame)
            report=customer_month.run(context).tables["source_window_coverage"]
            self.assertFalse(report.observation_continuity_verified.any())
            self.assertTrue((report.days_within_global_source_bounds<=report.window_days).all())
        self.assertEqual(list((self.root/"work").iterdir()),[])

    def test_full_pipeline_exact_moments_duplicates_and_labels(self):
        analysis=run_analysis(self.config)
        self.assertEqual(len(analysis.sections),16)
        self.assertTrue(all(v.metadata["status"]=="completed" for v in analysis.sections.values()))
        quality=analysis.sections["descriptive.quality"].tables
        duplicates=quality["duplicates"]
        self.assertEqual(duplicates.loc[duplicates.source.eq("transactions"),"duplicate_n"].iloc[0],1)
        amounts=analysis.sections["descriptive.transactions"].tables["amount_summary"].iloc[0]
        finite=self.trx.amount.replace([np.inf,-np.inf],np.nan)
        self.assertAlmostEqual(amounts["mean"],finite.mean())
        self.assertAlmostEqual(amounts["std"],finite.std())
        self.assertAlmostEqual(amounts["median"],finite.median())
        labels=analysis.sections["descriptive.targets"].tables["target_prevalence"].set_index("target")
        self.assertEqual(labels.loc["target_2","labelled_n"],11)
        self.assertEqual(labels.loc["target_2","positive_n"],1)
        invalid=quality["invalid_values"].set_index(["source","check"])
        self.assertEqual(invalid.loc[("transactions","unknown_client_rows"),"n"],1)
        self.assertEqual(invalid.loc[("dialog","invalid_embedding_shape"),"n"],1)
        self.assertEqual(invalid.loc[("dialog","invalid_embedding_values"),"n"],1)

    def test_batch_boundary_invariance(self):
        with self.context(replace(self.config,batch_size=1)) as left:
            customers.run(left)
            first=left.artifacts["customer_features"]
        with self.context(replace(self.config,batch_size=7)) as right:
            customers.run(right)
            second=right.artifacts["customer_features"]
        pd.testing.assert_frame_equal(first,second)
        source=pd.DataFrame({"x":range(100)})
        one,two=UniformSample(9,42),UniformSample(9,42)
        one.update(source)
        for start in range(0,100,7):
            two.update(source.iloc[start:start+7])
        pd.testing.assert_frame_equal(one.frame,two.frame)

    def test_unresolved_windows_keep_independent_analyses(self):
        analysis=run_analysis(replace(self.config,label_alignment="unresolved"))
        self.assertEqual(analysis.sections["features.customer_month"].metadata["status"],"skipped")
        self.assertEqual(analysis.sections["exploration.temporal"].metadata["status"],"completed")
        self.assertEqual(analysis.sections["exploration.missingness"].metadata["status"],"completed")
        self.assertEqual(analysis.sections["exploration.stability"].metadata["status"],"skipped")

    def test_window_validation_and_config_paths(self):
        frame=pd.DataFrame({"mon":["2022-02-28"],"cutoff":["2022-02-01 12:00"],
            "label_start":["2022-02-01 12:00"],"label_end":["2022-03-01"]})
        with self.assertRaises(ValueError):
            validate_label_windows(frame)
        frame["cutoff"]="2022-02-01"
        valid=validate_label_windows(frame)
        self.assertEqual(valid.cutoff.iloc[0],pd.Timestamp("2022-02-01"))
        with self.assertRaisesRegex(ValueError,"duy nhất"):
            validate_label_windows(pd.concat([frame,frame]))
        config=self.root/"config.json"
        config.write_text(json.dumps({"data_root":"data","output_dir":"out"}),encoding="utf-8")
        loaded=load_config(config)
        self.assertEqual(loaded.data_root,self.data)
        self.assertFalse(build_plan(loaded)["reads_raw_data"])

    def test_export_cli_figures_and_raw_unchanged(self):
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.data.rglob("*.parquet")}
        analysis=run_analysis(self.config)
        output=export_report(analysis,plots=True)
        self.assertTrue((output/"report.md").exists())
        manifest=json.loads((output/"manifest.json").read_text(encoding="utf-8"))
        self.assertGreater(len(manifest["figures"]),10)
        from PIL import Image
        for name in manifest["figures"]:
            with Image.open(output/name) as image:
                image.verify()
        for entry in manifest["tables"]:
            self.assertNotIn("client_id",entry["columns"])
        self.assertNotIn("NaN",(output/"metadata.json").read_text(encoding="utf-8"))
        after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.data.rglob("*.parquet")}
        self.assertEqual(before,after)
        config=self.root/"config.json"
        config.write_text(json.dumps(self.config.to_dict()),encoding="utf-8")
        env=os.environ.copy()
        env["PYTHONPATH"]=str(Path(__file__).resolve().parents[1]/"src")
        command=[sys.executable,"-X","utf8","-m","mbd_mini_eda","run","--config",str(config),"--quiet"]
        completed=subprocess.run(command,cwd=self.root,env=env,capture_output=True,text=True,encoding="utf-8")
        self.assertEqual(completed.returncode,0,completed.stderr)
        self.assertIn("EDA report:",completed.stdout)

    def test_prefix_scope_and_empty_event_tables(self):
        analysis=run_analysis(replace(self.config,max_batches_per_source=1))
        self.assertEqual(analysis.metadata["event_scope"],"prefix")
        self.assertLess(analysis.metadata["sources"]["transactions"]["scanned_rows"],len(self.trx))
        for path in self.data.glob("detail/*/fold=*/*.parquet"):
            table=pq.ParquetFile(path).read()
            pq.write_table(table.slice(0,0),path)
        analysis=run_analysis(self.config)
        self.assertEqual(analysis.metadata["sources"]["transactions"]["scanned_rows"],0)

    def test_sample_scope_explicit_mapping_and_source_protection(self):
        analysis=run_analysis(replace(self.config,sample_size=2,duplicate_mode="sample",timezone="UTC"))
        amounts=analysis.sections["descriptive.transactions"].tables["amount_summary"].iloc[0]
        self.assertEqual(amounts.quantile_scope,"uniform_row_sample")
        self.assertEqual(amounts.sample_rows,2)
        self.assertAlmostEqual(amounts["mean"],self.trx.amount.replace(np.inf,np.nan).mean())
        with self.assertRaisesRegex(ValueError,"dữ liệu nguồn"):
            export_report(analysis,self.data/"output")
        with self.context() as context:
            windows=context.artifacts["label_windows"].copy()
        provided=run_analysis(replace(self.config,label_alignment="unresolved"),label_windows=windows)
        self.assertEqual(provided.sections["features.customer_month"].metadata["label_alignment"],"provided_windows")
        self.assertEqual(provided.metadata["label_alignment"],"provided_windows")
        path=self.root/"windows.csv"
        windows.to_csv(path,index=False)
        explicit=replace(self.config,label_alignment="explicit",label_windows_path=path,analysis_months=("2022-02-28",))
        outcome=run_analysis(explicit)
        checks=outcome.sections["features.customer_month"].tables["customer_month_checks"]
        self.assertEqual(set(checks.mon),{"2022-02-28"})
        with self.assertRaisesRegex(ValueError,"khóa"):
            path=self.data/"targets"/"fold=0"/"extra.parquet"
            bad=self.labels[self.labels.fold.eq(0)].drop(columns="fold")
            bad.to_parquet(path,index=False)
            read_reference_tables(source_paths(self.data))


if __name__=="__main__":
    unittest.main()
