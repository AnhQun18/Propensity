"""Đối chiếu bảy bước mô tả với pandas, xuyên batch/shard và giới hạn bộ nhớ."""

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
from mbd_mini_eda.io import iter_batches, source_paths, read_reference_tables
from mbd_mini_eda.models import AnalysisContext
from mbd_mini_eda.aggregation import customer_table
from mbd_mini_eda.engine import engine
from mbd_mini_eda.reporting import export_report
from mbd_mini_eda.statistics import UniformSample, canonical, row_keys


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
            batch_size=2,sample_size=100,embedding_size=3,sqlite_cache_mb=8,
            duplicate_mode="exact")

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

    def test_customer_aggregation_cardinality_and_zero_log_clients(self):
        with self.context() as context:
            frame=customer_table(context)
            self.assertEqual(len(frame),4)
            self.assertFalse(frame.client_id.duplicated().any())
            a=frame[frame.client_id.eq("a")].iloc[0]
            self.assertEqual(a.trx_event_count,5)
            self.assertEqual(a.trx_amount_sum,180)
            self.assertEqual(a.trx_active_days,4)
            self.assertEqual(a.geo_unique_locations,2)
            self.assertEqual(a.any_active_days,6)
            self.assertEqual(a.trx_first_event,pd.Timestamp("2022-01-01 12:00"))
            self.assertEqual(a.trx_last_event,pd.Timestamp("2022-02-20 12:00"))
            self.assertEqual(a.trx_mean_gap_seconds,50*86400/4)
            d=frame[frame.client_id.eq("d")].iloc[0]
            self.assertEqual(d.trx_event_count,0)
            self.assertTrue(pd.isna(d.trx_amount_mean))
            self.assertTrue(pd.isna(d.trx_first_event))
            self.assertNotIn("trans_count",frame)
            self.assertNotIn("z",set(frame.client_id))
            # Vị trí 10 lặp ở hai ngày khác nhau chỉ tính một lần trên toàn lịch sử.
            self.assertEqual(engine(context).db.execute("SELECT COUNT(*) FROM locations WHERE client_id='a'").fetchone()[0],2)
        self.assertEqual(list((self.root/"work").iterdir()),[])

    def test_full_pipeline_exact_moments_duplicates_and_labels(self):
        analysis=run_analysis(self.config)
        self.assertEqual(len(analysis.sections),7)
        self.assertTrue(all(v.metadata["status"]=="completed" for v in analysis.sections.values()))
        quality=analysis.sections["descriptive.quality"].tables
        duplicates=quality["duplicates"]
        self.assertEqual(duplicates.loc[duplicates.source.eq("transactions"),"duplicate_n"].iloc[0],1)
        amounts=analysis.sections["descriptive.transactions"].tables["amount_summary"].iloc[0]
        finite=self.trx.amount.replace([np.inf,-np.inf],np.nan)
        self.assertAlmostEqual(amounts["mean"],finite.mean())
        self.assertAlmostEqual(amounts["std"],finite.std())
        self.assertAlmostEqual(amounts["median"],finite.median())
        self.assertEqual(amounts.whisker_low,0)
        self.assertEqual(amounts.whisker_high,30)
        labels=analysis.sections["descriptive.targets"].tables["target_prevalence"].set_index("target")
        self.assertEqual(labels.loc["target_2","labelled_n"],11)
        self.assertEqual(labels.loc["target_2","positive_n"],1)
        invalid=quality["invalid_values"].set_index(["source","check"])
        self.assertEqual(invalid.loc[("transactions","unknown_client_rows"),"n"],1)
        self.assertEqual(invalid.loc[("dialog","invalid_embedding_shape"),"n"],1)
        self.assertEqual(invalid.loc[("dialog","invalid_embedding_values"),"n"],1)

    def test_batch_boundary_invariance(self):
        with self.context(replace(self.config,batch_size=1)) as left:
            first=customer_table(left)
        with self.context(replace(self.config,batch_size=7)) as right:
            second=customer_table(right)
        pd.testing.assert_frame_equal(first,second)
        source=pd.DataFrame({"x":range(100)})
        one,two=UniformSample(9,42),UniformSample(9,42)
        one.update(source)
        for start in range(0,100,7):
            two.update(source.iloc[start:start+7])
        pd.testing.assert_frame_equal(one.frame,two.frame)

    def test_nullable_integer_codes_without_pandas_metadata(self):
        # Nguồn thật không có metadata pandas: null từng batch không được đổi mã thành float.
        for path in self.data.rglob("*.parquet"):
            raw=pq.ParquetFile(path).read().replace_schema_metadata(None)
            pq.write_table(raw,path)
        snapshots=[]
        for size in (1,7):
            with self.context(replace(self.config,batch_size=size)) as context:
                categories=engine(context).categories("transactions")
                currency=categories[categories.column.eq("currency")]
                self.assertEqual(set(currency[currency.kind.eq("value")].value),{"1","2"})
                self.assertEqual(currency.distinct_n.unique().tolist(),[2])
                self.assertEqual(currency.n.sum(),len(self.trx))
                snapshots.append(categories)
        pd.testing.assert_frame_equal(*snapshots)
        directory=self.root/"integer_precision"
        directory.mkdir()
        exact=2**53+1
        pq.write_table(pa.table({"code":pa.array([exact,None,exact],type=pa.int64())}),directory/"part.parquet")
        frame=pd.concat(iter_batches(directory,batch_size=1),ignore_index=True)
        self.assertEqual(int(frame.code.iloc[0]),exact)
        self.assertEqual(str(frame.code.dtype),"Int64")

    def test_monthly_plot_uses_calendar_axis(self):
        from types import SimpleNamespace
        from mbd_mini_eda.visualization import descriptive_figures
        monthly=pd.DataFrame({"source":["dialog","dialog","transactions","transactions"],
            "month":["2022-02","2022-03","2021-12","2022-01"],"events":[1,2,3,4]})
        result=SimpleNamespace(sections={"descriptive.temporal":SimpleNamespace(tables={"monthly_activity":monthly})})
        axis=descriptive_figures(result)["monthly_activity"].axes[0]
        for line in axis.lines:
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(line.get_xdata()))
            self.assertTrue(pd.Index(line.get_xdata()).is_monotonic_increasing)
        self.assertLess(axis.lines[1].get_xdata()[0],axis.lines[0].get_xdata()[0])

    def test_seven_steps_progress_and_embedding_batch_limit(self):
        progress=[]
        analysis=run_analysis(replace(self.config,batch_size=7,dialog_batch_size=1),progress=progress.append)
        expected=["structure","quality","temporal","transactions","modalities","customers","targets"]
        self.assertEqual(list(analysis.sections),["descriptive."+s for s in expected])
        self.assertEqual(analysis.metadata["sources"]["dialog"]["batch_size"],1)
        self.assertEqual(analysis.metadata["sources"]["dialog"]["scanned_rows"],4)
        self.assertEqual(analysis.metadata["sources"]["transactions"]["scanned_rows"],len(self.trx))
        self.assertTrue(any("dòng" in message for message in progress))
        self.assertTrue(all(s.metadata["duration_seconds"]>=0 for s in analysis.sections.values()))

    def test_embedding_duplicate_keys_match_array_and_list_content(self):
        rows=pd.DataFrame({"client_id":["a","a","a"],
                           "embedding":[np.array([1.,np.nan]),[1.,None],np.array([1.,2.])]})
        keys=list(row_keys(rows))
        self.assertEqual(keys[0],keys[1])
        self.assertNotEqual(keys[0],keys[2])
        self.assertEqual(canonical(np.array([[1.,np.nan],[2.,3.]])),[[1.,None],[2.,3.]])

    def test_legacy_config_paths_and_memory_validation(self):
        config=self.root/"config.json"
        config.write_text(json.dumps({"data_root":"data","output_dir":"out","history_days":[30,84,180],
                                      "label_alignment":"explicit","label_windows_path":"missing.csv"}),encoding="utf-8")
        with self.assertWarnsRegex(UserWarning,"7 bước"):
            loaded=load_config(config)
        self.assertEqual(loaded.data_root,self.data)
        self.assertEqual(loaded.output_dir,self.root/"out")
        self.assertFalse(build_plan(loaded)["reads_raw_data"])
        self.assertEqual(len(build_plan(loaded)["steps"]),7)
        self.assertNotIn("history_days",loaded.to_dict())
        for key in ("batch_size","dialog_batch_size","sqlite_cache_mb"):
            with self.assertRaisesRegex(ValueError,key):
                replace(self.config,**{key:0})
        with self.context() as context:
            self.assertEqual(engine(context).db.execute("PRAGMA cache_size").fetchone()[0],-8192)

    def test_export_cli_figures_and_raw_unchanged(self):
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.data.rglob("*.parquet")}
        analysis=run_analysis(self.config)
        output=export_report(analysis,plots=True)
        self.assertTrue((output/"report.md").exists())
        manifest=json.loads((output/"manifest.json").read_text(encoding="utf-8"))
        self.assertGreater(len(manifest["figures"]),10)
        self.assertIn("figures/amount_boxplot.png",manifest["figures"])
        self.assertIn("figures/source_timeline.png",manifest["figures"])
        self.assertIn("erDiagram",(output/"report.md").read_text(encoding="utf-8"))
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

    def test_sample_scope_timezone_and_source_protection(self):
        analysis=run_analysis(replace(self.config,sample_size=2,duplicate_mode="sample",timezone="UTC"))
        amounts=analysis.sections["descriptive.transactions"].tables["amount_summary"].iloc[0]
        self.assertEqual(amounts.quantile_scope,"uniform_row_sample")
        self.assertEqual(amounts.sample_rows,2)
        self.assertAlmostEqual(amounts["mean"],self.trx.amount.replace(np.inf,np.nan).mean())
        with self.assertRaisesRegex(ValueError,"dữ liệu nguồn"):
            export_report(analysis,self.data/"output")
        with self.context(replace(self.config,timezone="UTC")) as context:
            self.assertEqual(str(customer_table(context).trx_first_event.dt.tz),"UTC")
        with self.assertRaisesRegex(ValueError,"dữ liệu nguồn"):
            run_analysis(replace(self.config,work_dir=self.data/"scratch"))
        with self.assertRaisesRegex(ValueError,"khóa"):
            path=self.data/"targets"/"fold=0"/"extra.parquet"
            bad=self.labels[self.labels.fold.eq(0)].drop(columns="fold")
            bad.to_parquet(path,index=False)
            read_reference_tables(source_paths(self.data))


if __name__=="__main__":
    unittest.main()
