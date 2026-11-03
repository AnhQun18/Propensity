"""Statistical invariants, chunked I/O, multi-table audits and package CLI."""
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
from eda import DatasetSpec, FileSource, Relationship, profile, profile_dataset, export_report
from eda.analysis import bivariate, drift, events, target
from eda.analysis.relationships import audit
from eda.descriptive import categorical, dataset, quality, temporal
from eda.loader import source_manifest


class EDATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_id_roles_and_numeric_statistics_match_pandas(self):
        frame = pd.DataFrame({"user_id": [1, 2, 3, 4, 5], "x": [1., 2., 3., 4., None]})
        original = frame.copy(deep=True)
        result = profile(frame, DatasetSpec(id_columns=["user_id"]))
        table = result.tables["numeric_summary"].set_index("column")
        self.assertEqual(list(table.index), ["x"])
        for actual, expected in [(table.loc["x", "mean"], frame.x.mean()),
                                 (table.loc["x", "std"], frame.x.std()),
                                 (table.loc["x", "kurtosis"], frame.x.kurt()),
                                 (table.loc["x", "median"], frame.x.median())]:
            self.assertAlmostEqual(actual, expected)
        pd.testing.assert_frame_equal(frame, original)

    def test_missing_labels_and_multiple_targets(self):
        frame = pd.DataFrame({"g": ["a", "a", "b"], "a": [1, None, 0], "b": [0, 1, 1]})
        spec = DatasetSpec(targets={"a": "binary", "b": "binary"}, group_columns=["g"], independent_observations=True)
        result = profile(frame, spec)
        rates = result.tables["target_rates"]
        first = rates[(rates.target == "a") & (rates.group == "str:a")].iloc[0]
        self.assertEqual(first.labelled_n, 1)
        self.assertEqual(first.rate, 1)
        self.assertIn("wilson_low", rates)
        self.assertEqual(set(result.tables["target_summary"].target), {"a", "b"})

    def test_binary_validation_scans_unsampled_rows(self):
        frame = pd.DataFrame({"y": [0] * 100 + [7]})
        with self.assertRaisesRegex(ValueError, "0/1"):
            profile(frame, DatasetSpec(targets={"y": "binary"}, mode="approximate", sample_size=1))

    def test_no_target_group_summary(self):
        result = profile(pd.DataFrame({"g": ["a", "a", "b"], "x": [1., 3., 9.]}), DatasetSpec(group_columns=["g"]))
        self.assertNotIn("target_summary", result.tables)
        self.assertEqual(result.tables["group_numeric"].iloc[0]["mean"], 2)

    def test_schema_errors(self):
        for spec in [DatasetSpec(id_columns=["absent"]), DatasetSpec(id_columns=["x"], numeric_columns=["x"])]:
            with self.assertRaises(ValueError):
                profile(pd.DataFrame({"x": [1]}), spec)
        with self.assertRaises(ValueError):
            profile(pd.DataFrame([[1, 2]], columns=["x", "x"]))

    def test_empty_and_all_missing_data(self):
        for frame in [pd.DataFrame({"x": pd.Series(dtype=float)}), pd.DataFrame({"x": [np.nan, np.nan]})]:
            result = profile(frame)
            self.assertEqual(result.tables["numeric_summary"].iloc[0].finite_n, 0)
            export_report(result, self.directory / str(len(frame)))

    def test_infinite_values_counted_and_excluded_from_moments(self):
        result = profile(pd.DataFrame({"x": [1., 3., np.inf, -np.inf, None]}))
        row = result.tables["numeric_summary"].iloc[0]
        self.assertEqual(row.infinite_n, 2)
        self.assertEqual(row["mean"], 2)
        self.assertEqual(result.tables["anomalies"].iloc[0].infinite_n, 2)

    def test_nested_columns_and_literal_missing_marker(self):
        frame = pd.DataFrame({"v": [[1, 2], [1, 2], None], "g": ["<MISSING>", None, "x"]})
        result = profile(frame)
        self.assertEqual(result.tables["column_profile"].set_index("column").loc["v", "distinct_nonnull"], 1)
        freq = categorical.frequencies(frame, ["g"], top=1)
        self.assertEqual(freq.n.sum(), len(frame))
        self.assertEqual(freq.loc[freq.kind == "missing", "n"].iloc[0], 1)
        self.assertEqual(quality.duplicates(pd.DataFrame({"v": [[1], [1]]})).iloc[0].duplicate_rows, 1)

    def test_csv_chunks_concat_and_sha256(self):
        frames = [pd.DataFrame({"x": [1., 3.], "g": ["a", "b"]}), pd.DataFrame({"x": [2., 7.], "g": ["c", None]})]
        paths = []
        for i, frame in enumerate(frames):
            path = self.directory / f"{i}.csv"
            frame.to_csv(path, index=False)
            paths.append(str(path))
        source = FileSource(paths)
        from_disk = profile(source, DatasetSpec(chunk_size=1))
        from_memory = profile(pd.concat(frames, ignore_index=True))
        pd.testing.assert_frame_equal(from_disk.tables["numeric_summary"], from_memory.tables["numeric_summary"])
        manifest = source_manifest(source)
        self.assertEqual(manifest.iloc[0].sha256, hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest())
        self.assertTrue(source_manifest(source, False).sha256.isna().all())

    def test_csv_no_header(self):
        path = self.directory / "events.csv"
        path.write_text("1,10\n2,20\n", encoding="utf-8")
        result = profile(FileSource([str(path)], read_options={"header": None, "names": ["id", "x"]}), DatasetSpec(id_columns=["id"]))
        self.assertEqual(result.metadata["total_rows"], 2)

    def test_partitioned_parquet_with_vectors(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.table({"id": [1, 2, 3], "fold": [0, 1, 0], "embedding": [[1., 2.], None, [3., 4.]]})
        path = self.directory / "partitioned"
        pq.write_to_dataset(table, root_path=path, partition_cols=["fold"])
        result = profile(FileSource([str(path)]), DatasetSpec(id_columns=["id"], categorical_columns=["fold", "embedding"], chunk_size=1))
        self.assertEqual(result.metadata["total_rows"], 3)
        self.assertEqual(len(result.tables["manifest"]), 2)

    def test_sampling_is_bounded_deterministic_and_chunk_invariant(self):
        frame = pd.DataFrame({"x": np.arange(200, dtype=float), "g": [None] * 20 + ["a"] * 180})
        path = self.directory / "data.csv"
        frame.to_csv(path, index=False)
        spec = DatasetSpec(mode="approximate", sample_size=13, chunk_size=7)
        result = profile(FileSource([str(path)]), spec)
        other = profile(frame, spec)
        self.assertEqual(result.metadata["total_rows"], 200)
        self.assertEqual(result.metadata["analyzed_rows"], 13)
        self.assertEqual(set(result._frame.x), set(other._frame.x))
        self.assertEqual(result.tables["missing"].set_index("column").loc["g", "missing_n"], 20)
        self.assertEqual(result.metadata["table_scopes"]["numeric_summary"], "sample")

    def test_pairwise_spearman_and_effective_counts(self):
        frame = pd.DataFrame({"x": [1., 1., 3., None, 7.], "y": [3., None, 2., 5., 1.]})
        result = profile(frame)
        row = result.tables["correlations"].iloc[0]
        self.assertAlmostEqual(row.spearman, frame.corr(method="spearman").loc["x", "y"])
        self.assertEqual(row.pair_n, 3)

    def test_temporal_gaps_and_invalid_values(self):
        summary, counts = temporal.summarize(pd.DataFrame({"t": ["2024-01-01", "2024-01-03", "bad"]}), {"t": {}})
        self.assertEqual(summary.iloc[0].empty_periods, 1)
        self.assertEqual(summary.iloc[0].invalid_n, 1)
        self.assertEqual(list(counts.events), [1, 0, 1])
        parsed = temporal.parse_time(pd.Series([0]), {"unit": "s", "timezone": "Asia/Shanghai"})
        self.assertEqual(parsed.iloc[0].hour, 8)

    def test_time_grid_limit(self):
        with self.assertRaisesRegex(ValueError, "coarser"):
            temporal.summarize(pd.DataFrame({"t": ["1800-01-01", "2200-01-01"]}), {"t": {}}, max_periods=10)

    def test_nanosecond_timestamp_grid_does_not_overflow(self):
        frame = pd.DataFrame({"t": pd.Series(pd.date_range("2022-01-01", periods=3), dtype="datetime64[ns]")})
        summary, counts = temporal.summarize(frame, {"t": {}})
        self.assertEqual(summary.iloc[0].valid_n, 3)
        self.assertEqual(counts.events.sum(), 3)

    def test_relationships_measure_multiplication_without_join(self):
        left = pd.DataFrame({"id": [1, 1, 2, None]})
        right = pd.DataFrame({"id": [1, 1, 3]})
        rel = Relationship("users", "events", "users", ["id"], ["id"])
        table = audit(left, right, rel)
        self.assertEqual(table.iloc[0].projected_inner_rows, 4)
        self.assertEqual(table.iloc[0].left_matched_rows, 2)
        self.assertEqual(table.iloc[0].left_missing_keys, 1)
        self.assertFalse(table.iloc[0].cardinality_pass)
        self.assertEqual(len(left), 4)

    def test_sparse_matrix_and_coverage(self):
        frame = pd.DataFrame({"u": [1, 1, 2], "t": ["2024-01-01", "2024-01-01", "2024-01-03"]})
        matrix, users, days = events.customer_day_matrix(frame, "u", "t", start="2024-01-01", end="2024-01-03")
        self.assertEqual(matrix.shape, (2, 3))
        self.assertEqual(matrix.sum(), 2)
        self.assertEqual(len(users), 2)
        self.assertEqual(len(days), 3)
        with self.assertRaises(ValueError):
            events.customer_day_matrix(frame, "u", "t", start="2023-12-31", end="2024-01-03")

    def test_drift_and_categorical_bivariate(self):
        a = pd.DataFrame({"g": ["a", None], "x": [1., 2.]})
        b = pd.DataFrame({"g": ["b", None], "x": [3., 4.]})
        result = drift.compare(a, b, numeric=["x"], categorical=["g"]).set_index("metric")
        self.assertEqual(result.loc["total_variation", "value"], .5)
        self.assertEqual(result.loc["ks_statistic", "value"], 1)
        self.assertEqual(bivariate.categorical_categorical(a, "g", "x").to_numpy().sum(), 2)

    def test_wilson_and_leakage_flags(self):
        low, high = target.wilson(5, 10)
        self.assertAlmostEqual(low, .236593, places=5)
        self.assertAlmostEqual(high, .763407, places=5)
        flags = target.leakage_flags(pd.DataFrame({"y": [0, 1], "copy": [0, 1]}), ["y"], ["copy"])
        self.assertEqual(flags.iloc[0].flag, "exact_label_copy")

    def test_multitable_export_notebook_and_preview_opt_in(self):
        sources = {"events": pd.DataFrame({"id": [1, 1], "x": [2., 3.]}), "users": pd.DataFrame({"id": [1]})}
        specs = {name: DatasetSpec(dataset=name, id_columns=["id"]) for name in sources}
        result = profile_dataset(sources, specs, dataset="demo", relationships=[Relationship("users", "events", "users", ["id"], ["id"])])
        output = export_report(result, self.directory / "docs" / "demo", plots=True, notebook=True)
        self.assertTrue((output / "tables/events/numeric_summary.csv").is_file())
        self.assertTrue((output / "relationships/coverage.csv").is_file())
        self.assertTrue((output / "figures/events/histogram_0.png").is_file())
        self.assertTrue((output / "eda_report.ipynb").is_file())
        self.assertFalse(list(output.rglob("preview.csv")))
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["tables"]["events"]["total_rows"], 2)
        self.assertEqual(len(dataset.preview(sources["events"], 1)), 1)

    def test_cli_config_paths_and_default_docs(self):
        (self.directory / "data.csv").write_text("x,g\n1,a\n2,b\n", encoding="utf-8")
        config = {"source": {"paths": ["data.csv"]}, "spec": {"dataset": "demo"}}
        path = self.directory / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        env = {**os.environ, "MPLBACKEND": "Agg"}
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        result = subprocess.run([sys.executable, "-m", "eda", "--config", str(path)], cwd=self.directory,
                                capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.directory / "docs/demo/README.md").is_file())

    def test_all_visualization_modules(self):
        import matplotlib.pyplot as plt
        from eda.visualization import numerical, categorical as cat_plot, bivariate as bi_plot
        from eda.visualization import correlation, target as target_plot, temporal as time_plot, quality as q_plot
        frame = pd.DataFrame({"x": [1., 2., 3.], "y": [3., 2., 4.], "g": ["a", "a", "b"], "label": [0, 1, 1]})
        freq = categorical.frequencies(frame, ["g"])
        crosstab = bivariate.categorical_categorical(frame, "g", "label")
        rates = target.grouped_rates(frame, ["g"], {"label": "binary"}, True)
        result = profile(frame, DatasetSpec(targets={"label": "binary"}))
        figures = [numerical.histogram(frame.x), numerical.boxplot(frame.x), cat_plot.bar(freq), cat_plot.pie(freq),
                   cat_plot.stacked_bar(crosstab), bi_plot.scatter(frame, "x", "y", max_points=2),
                   bi_plot.grouped_boxplot(frame, "g", "x"), correlation.heatmap(result.tables["correlations"]),
                   target_plot.distribution(result.tables["target_summary"]), target_plot.rates(rates),
                   time_plot.activity(pd.DataFrame({"period": ["2024-01-01"], "events": [1]})),
                   q_plot.missing_bar(quality.missing(frame)), q_plot.missing_pattern(frame)]
        for i, figure in enumerate(figures):
            figure.savefig(self.directory / f"{i}.png")
            plt.close(figure)
        self.assertEqual(len(list(self.directory.glob("*.png"))), len(figures))


if __name__ == "__main__":
    unittest.main()
