"""Reproducible seven-step EDA; raw and prepared inputs are strictly read-only."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import html
import importlib.metadata
import json
import re
import time

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def qi(name):
    return '"' + name.replace('"', '""') + '"'


def qs(value):
    return "'" + str(value).replace("'", "''") + "'"


class EDA:
    def __init__(self, root):
        self.root = Path(root)
        self.raw = self.root / "data/MBD-mini"
        self.prepared = self.root / "outputs/propensity_processing_v3"
        self.out = self.root / "deliverables/eda"
        for folder in ("tables", "figures"):
            (self.out / folder).mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.tables, self.figures, self.previews = {}, {}, {}
        self.db = duckdb.connect()
        self.db.execute("SET memory_limit='4GB'")
        self.db.execute("SET threads=2")
        self.db.execute("SET preserve_insertion_order=false")
        tmp = self.root / "outputs/eda_work"
        tmp.mkdir(parents=True, exist_ok=True)
        self.db.execute(f"SET temp_directory={qs(tmp.as_posix())}")
        manifest = json.loads((self.prepared / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "ready":
            raise ValueError("Notebook 03 output must be ready")
        self.feature_schema = json.loads((self.prepared / "feature_schema.json").read_text(encoding="utf-8"))
        self.predictors = self.feature_schema["predictors"]
        self.paths = {"clients": "client_split", "targets": "targets", "trx": "detail/trx",
                      "geo": "detail/geo", "dialog": "detail/dialog", "ptls_trx": "ptls/trx",
                      "ptls_geo": "ptls/geo", "ptls_dialog": "ptls/dialog"}
        self.files = []
        for name, folder in self.paths.items():
            paths = sorted((self.raw / folder).rglob("*.parquet"))
            if not paths:
                raise FileNotFoundError(folder)
            self.files.extend(paths)
            self.db.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet({qs((self.raw / folder).as_posix() + '/**/*.parquet')}, hive_partitioning=true)")
        self.files.extend(sorted((self.prepared / "features").rglob("*.parquet")))
        self.input_before = self.inventory()
        self.db.execute(f"CREATE VIEW features AS SELECT * EXCLUDE(month) FROM read_parquet({qs((self.prepared / 'features').as_posix() + '/**/*.parquet')}, hive_partitioning=true)")
        # Only the design's training period informs associations/feature proposals.
        self.db.execute("CREATE VIEW train AS SELECT * FROM features WHERE cutoff < TIMESTAMP '2022-09-01'")
        self.scope_note = ("**Phạm vi:** raw toàn lịch sử; overview/nhãn theo tháng trên toàn cohort. "
                           "Thống kê đặc trưng và liên hệ target chỉ dùng cutoff 02–08/2022. "
                           "Không mở kết quả mô hình/test để lựa chọn biến; đây là EDA, không huấn luyện mô hình.")
        self.missing_note = ("**Định hướng xử lý:** giữ khách thiếu nguồn; phân biệt không có sự kiện, cửa sổ ngoài độ phủ và null thuộc tính. "
                             "Count=0 chỉ có nghĩa không có log trong cửa sổ có độ phủ. Với thuật toán cần imputation, fit median/mode trên train "
                             "và thêm cờ missing; CatBoost có thể giữ NaN số. Không điền nhãn thiếu thành 0, không dùng ID/fold làm predictor. "
                             "Các trường phụ trợ trans_count/diff_trans_date không dùng làm predictor do cutoff/công thức chưa xác minh.")
        self.sample_limit = 30000
        self.seed = 42
        self.sample = None
        self.validation = {}
        plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.spines.top": False,
                             "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18})

    def query(self, sql):
        progress = self.root / "outputs/eda_query_progress.json"
        progress.write_text(json.dumps({"started_utc":datetime.now(timezone.utc).isoformat(),
                                        "sql":sql},ensure_ascii=False),encoding="utf-8")
        return self.db.execute(sql).fetchdf()

    def inventory(self):
        return [{"path":p.relative_to(self.root).as_posix(), "size":p.stat().st_size,
                 "mtime_ns":p.stat().st_mtime_ns} for p in self.files]

    def table(self, name, frame):
        self.tables[name] = frame
        frame.to_csv(self.out / "tables" / f"{name}.csv", index=False, encoding="utf-8-sig")
        return frame

    def figure(self, name, fig, caption, group):
        fig.tight_layout()
        path = self.out / "figures" / f"{name}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        self.figures[name] = {"path": path, "caption": caption, "group": group}

    def show_figures(self, group):
        from IPython.display import display, Image, Markdown
        for value in self.figures.values():
            if value["group"] == group:
                display(Image(filename=str(value["path"])))
                display(Markdown(value["caption"]))

    def describe(self, view):
        return self.query(f"DESCRIBE {view}")[["column_name", "column_type"]]

    def role(self, source, col, dtype):
        if col == "client_id": return "identifier"
        if col in ("mon", "cutoff", "event_time"):
            return "datetime_sequence" if "[]" in dtype else "datetime"
        if col.startswith("target_") or col == "y_any": return "target"
        if col == "embedding": return "embedding_vector_or_sequence"
        if "[]" in dtype: return "sequence"
        if col in ("trans_count", "diff_trans_date"): return "numeric_auxiliary_not_predictor"
        if col in ("amount",): return "numeric"
        if source in ("features", "train"):
            return "binary_or_count_or_continuous_feature" if col in self.predictors else "metadata"
        return "categorical_code"

    def overview(self):
        rows, schema_rows = [], []
        grains = {"clients": "customer", "targets": "customer_month", "trx": "event",
                  "geo": "event", "dialog": "event", "features": "customer_month", "train": "customer_month"}
        for name in [*self.paths, "features", "train"]:
            schema = self.describe(name)
            count = int(self.query(f"SELECT count(*) n FROM {name}").iloc[0, 0])
            time_col = "mon" if name == "targets" else "cutoff" if name in ("features","train") else "event_time" if name in ("trx","geo","dialog") else None
            bounds = self.query(f"SELECT min({qi(time_col)}) first_time,max({qi(time_col)}) last_time FROM {name}").iloc[0].to_dict() if time_col else {}
            rows.append({"table": name, "rows": count, "columns": len(schema), "grain": grains.get(name, "customer_sequence"), **bounds})
            for col, dtype in schema.itertuples(index=False):
                schema_rows.append({"table": name, "column": col, "storage_dtype": dtype, "semantic_role": self.role(name, col, dtype)})
            sort = "client_id" + (", mon" if name == "targets" else ", cutoff" if name in ("features", "train") else ", event_time" if name in ("trx", "geo", "dialog") else "")
            selects = []
            # Complete raw preview; prepared preview shows a readable subset, full schema is above.
            cols = schema.values.tolist() if name not in ("features", "train") else schema.values.tolist()[:12]
            for col, dtype in cols:
                selects.append(f"length({qi(col)}) AS {qi(col + '_length')}" if "[]" in dtype else qi(col))
            select = ",".join(selects)
            head = self.query(f"SELECT {select} FROM {name} ORDER BY {sort} LIMIT 3")
            tail_sort = ",".join(x.strip() + " DESC" for x in sort.split(","))
            tail = self.query(f"SELECT {select} FROM {name} ORDER BY {tail_sort} LIMIT 3").iloc[::-1]
            frame = pd.concat([head.assign(preview="head"), tail.assign(preview="tail")], ignore_index=True)
            frame["client_id"] = frame["client_id"].map(lambda x: "id_" + hashlib.sha256(str(x).encode()).hexdigest()[:10])
            self.previews[name] = self.table("preview_" + name, frame)
            print(f"Overview {name}: {count:,} rows", flush=True)
        return self.table("overview", pd.DataFrame(rows)), self.table("schema", pd.DataFrame(schema_rows))

    @staticmethod
    def stats_expr(col):
        c = qi(col)
        finite = f"{c} IS NOT NULL AND isfinite({c})"
        x = f"CASE WHEN {finite} THEN {c} END"
        return f"count({x}) valid_n, avg({x}) mean, median({x}) median, stddev_samp({x}) std, min({x}) min, max({x}) max, quantile_cont({x},.25) q1, quantile_cont({x},.75) q3, skewness({x}) skewness"

    def raw_descriptive(self):
        numeric = []
        for source, cols in {"targets": ["trans_count", "diff_trans_date"], "trx": ["amount"]}.items():
            for col in cols:
                row = self.query(f"SELECT {self.stats_expr(col)} FROM {source}")
                row.insert(0, "column", col); row.insert(0, "table", source)
                row["scope"] = "full_raw"; numeric.append(row)
        cats = []
        self.cat_top = {}
        for source in ("clients", "targets", "trx", "geo"):
            schema = self.describe(source)
            cols = [c for c, t in schema.itertuples(index=False) if self.role(source, c, t) in ("categorical_code", "target")]
            for col in cols:
                summary = self.query(f"SELECT count(*) rows, count(DISTINCT {qi(col)}) unique_nonnull, count(*)-count({qi(col)}) missing_n FROM {source}").iloc[0].to_dict()
                top = self.query(f"SELECT cast({qi(col)} AS VARCHAR) value, count(*) n FROM {source} GROUP BY {qi(col)} ORDER BY n DESC, value NULLS LAST LIMIT 10")
                nonnull_top = top[top.value.notna()]
                if nonnull_top.empty:
                    mode, freq = None, 0
                else:
                    mode, freq = nonnull_top.iloc[0][["value", "n"]]
                cats.append({"table": source, "column": col, **summary, "mode_nonnull": mode,
                             "mode_n": freq, "mode_pct": 100 * freq / summary["rows"]})
                self.cat_top[(source, col)] = top
                self.table(f"category_top_{source}_{col}", top)
            print(f"Categorical stats {source}", flush=True)
        currency = self.query(f"SELECT currency, {self.stats_expr('amount')} FROM trx GROUP BY currency ORDER BY valid_n DESC")
        return (self.table("raw_numeric", pd.concat(numeric, ignore_index=True)),
                self.table("categorical_summary", pd.DataFrame(cats)), self.table("amount_by_currency", currency))

    def feature_descriptive(self):
        rows = []
        # One predictor per query bounds quantile working memory on 16GB machines.
        for i, col in enumerate(self.predictors):
            row = self.query(f"SELECT {self.stats_expr(col)}, count(*) rows, count(*) FILTER (WHERE {qi(col)} IS NULL) null_n, count(*) FILTER (WHERE isnan({qi(col)})) nan_n, count(*) FILTER (WHERE isinf({qi(col)})) infinite_n FROM train")
            row.insert(0, "feature", col)
            rows.append(row)
            if i % 40 == 0: print(f"Feature statistics {i+1}/{len(self.predictors)}", flush=True)
        stats = pd.concat(rows, ignore_index=True)
        stats["missing_pct"] = 100 * (stats.null_n + stats.nan_n) / stats.rows
        stats["constant_observed"] = (stats["min"] == stats["max"]) & stats.valid_n.gt(0)
        return self.table("feature_statistics_train", stats)

    def quality(self):
        missing, duplicates = [], []
        for source in self.paths:
            schema = self.describe(source)
            n = int(self.tables["overview"].set_index("table").loc[source, "rows"])
            expr = []
            for j, (col, dtype) in enumerate(schema.itertuples(index=False)):
                isnumeric = any(x in dtype for x in ("INT", "FLOAT", "DOUBLE", "DECIMAL")) and "[]" not in dtype
                expr += [f"count(*) FILTER (WHERE {qi(col)} IS NULL) AS c{j}_null",
                         f"count(*) FILTER (WHERE isnan({qi(col)})) AS c{j}_nan" if isnumeric else f"0 AS c{j}_nan",
                         f"count(*) FILTER (WHERE isinf({qi(col)})) AS c{j}_inf" if isnumeric else f"0 AS c{j}_inf"]
            counts = self.query(f"SELECT {','.join(expr)} FROM {source}").iloc[0]
            for j, (col, dtype) in enumerate(schema.itertuples(index=False)):
                a,b,c = (int(counts[f"c{j}_{suffix}"]) for suffix in ("null", "nan", "inf"))
                missing.append({"table": source, "column": col, "rows": n, "null_n": a,
                                "nan_n": b, "infinite_n": c, "missing_pct": 100*(a+b)/n,
                                "scope": "outer_column; nested elements checked separately" if "[]" in dtype else "full_raw"})
            keys = "client_id, mon" if source == "targets" else "client_id"
            if source in ("clients", "targets") or source.startswith("ptls_"):
                result = self.query(f"SELECT coalesce(sum(n-1),0) extra_rows, count(*) groups FROM (SELECT count(*) n FROM {source} GROUP BY {keys} HAVING count(*)>1)").iloc[0]
                key_dups = int(result.extra_rows)
                duplicates.append({"table": source, "check": "key", "extra_rows": key_dups, "groups": int(result.groups), "scope": "full_raw"})
                if key_dups == 0:
                    duplicates.append({"table": source, "check": "full_row", "extra_rows": 0, "groups": 0, "scope": "proven by unique key"})
                else:
                    result = self.query(f"SELECT coalesce(sum(n-1),0) extra_rows, count(*) groups FROM (SELECT *, count(*) n FROM {source} GROUP BY ALL HAVING count(*)>1)").iloc[0]
                    duplicates.append({"table": source, "check": "full_row", "extra_rows": int(result.extra_rows), "groups": int(result.groups), "scope": "full_raw"})
            else:
                print(f"Full row duplicate scan: {source}", flush=True)
                # Fold is part of the full row: full-row duplicates cannot cross folds.
                # Scan folds sequentially to bound the GROUP BY working set.
                fold_results=[]
                for fold in self.query(f"SELECT DISTINCT fold FROM {source} ORDER BY fold").fold:
                    fold_results.append(self.query(f"SELECT coalesce(sum(n-1),0) extra_rows, count(*) groups FROM (SELECT *, count(*) n FROM {source} WHERE fold IS NOT DISTINCT FROM {int(fold) if pd.notna(fold) else 'NULL'} GROUP BY ALL HAVING count(*)>1)").iloc[0])
                duplicates.append({"table": source, "check": "full_row", "extra_rows": sum(int(r.extra_rows) for r in fold_results), "groups": sum(int(r.groups) for r in fold_results), "scope": "full_raw; partitioned by fold (part of full row)"})
                same_time = self.query(f"SELECT coalesce(sum(n-1),0) extra_rows, count(*) groups FROM (SELECT client_id,event_time,count(*) n FROM {source} GROUP BY client_id,event_time HAVING count(*)>1)").iloc[0]
                duplicates.append({"table": source, "check": "same_client_timestamp_candidate", "extra_rows": int(same_time.extra_rows), "groups": int(same_time.groups), "scope": "full_raw; do not automatically delete"})
        # Nested vectors/sequences: count null/NaN/Inf elements in bounded SQL scans.
        nested = []
        for source in ("dialog", "ptls_trx", "ptls_geo", "ptls_dialog"):
            for col, dtype in self.describe(source).itertuples(index=False):
                if "[]" not in dtype: continue
                c = qi(col)
                arr = f"flatten({c})" if "[][]" in dtype else c
                numeric = any(t in dtype for t in ("INT", "FLOAT", "DOUBLE"))
                expr = f"sum(coalesce(length({arr}),0)) elements, sum(coalesce(length(list_filter({arr}, lambda x: x IS NULL)),0)) null_elements"
                if numeric:
                    expr += f", sum(coalesce(length(list_filter({arr}, lambda x: isnan(x))),0)) nan_elements, sum(coalesce(length(list_filter({arr}, lambda x: isinf(x))),0)) infinite_elements"
                r = self.query(f"SELECT {expr} FROM {source}")
                r.insert(0,"column",col); r.insert(0,"table",source); nested.append(r)
        self.table("nested_element_quality", pd.concat(nested, ignore_index=True).fillna(0))
        self.table("dialog_vector_lengths", self.query("SELECT length(embedding) vector_length,count(*) rows FROM dialog GROUP BY 1 ORDER BY 1"))
        reconcile = []
        for source in ("trx","geo","dialog"):
            r = self.query(f"SELECT sum(length(event_time)) sequence_events FROM ptls_{source}")
            r.insert(0,"source",source)
            r["detail_events"] = int(self.tables["overview"].set_index("table").loc[source,"rows"])
            r["detail_minus_sequence"] = r.detail_events - r.sequence_events
            reconcile.append(r)
        self.table("detail_sequence_reconciliation",pd.concat(reconcile,ignore_index=True))
        feature_missing = self.tables["feature_statistics_train"]
        for r in feature_missing.itertuples():
            missing.append({"table":"features_train", "column":r.feature,"rows":r.rows,"null_n":r.null_n,
                            "nan_n":r.nan_n,"infinite_n":r.infinite_n,"missing_pct":r.missing_pct,"scope":"train"})
        result = self.query("SELECT coalesce(sum(n-1),0) n FROM (SELECT client_id,cutoff,count(*) n FROM features GROUP BY ALL HAVING count(*)>1)")
        duplicates.append({"table":"features", "check":"customer_cutoff_key", "extra_rows":int(result.iloc[0,0]),"groups":None,"scope":"all 12 months"})
        self.db.execute("CREATE TEMP TABLE clean_targets AS SELECT DISTINCT * FROM targets")
        conflicts = self.query("SELECT client_id,mon,count(*) versions FROM clean_targets GROUP BY client_id,mon HAVING count(*)>1")
        if not conflicts.empty: raise ValueError("Conflicting labels: resolve before modelling")
        self.validation["target_conflicting_keys"] = len(conflicts)
        self.validation["label_exact_duplicates_removed_in_temp_view"] = int(self.query("SELECT (SELECT count(*) FROM targets)-(SELECT count(*) FROM clean_targets)").iloc[0,0])
        if int(result.iloc[0,0]) != 0: raise ValueError("Duplicate feature keys")
        context = self.query("SELECT trans_count=0 AS zero_transactions, count(*) rows, count(*) FILTER (WHERE diff_trans_date IS NULL) missing_diff FROM targets GROUP BY 1")
        self.table("missing_diff_context", context)
        return self.table("missing_values", pd.DataFrame(missing)), self.table("duplicates", pd.DataFrame(duplicates)), self.table("label_conflicts",conflicts)

    def load_sample(self):
        if self.sample is None:
            cols = ",".join(qi(c) for c in ["y_any",*self.predictors])
            self.sample = self.query(f"SELECT {cols} FROM train ORDER BY hash(client_id,cutoff,{self.seed}),client_id,cutoff LIMIT {self.sample_limit}")
            self.sample = self.sample.replace([np.inf,-np.inf],np.nan)
        return self.sample

    def univariate(self):
        monthly = self.query("SELECT cutoff, count(*) rows, count(y_any) labelled_n, sum(y_any) positive_n, 100.0*avg(y_any) positive_pct FROM features GROUP BY cutoff ORDER BY cutoff")
        self.table("target_monthly",monthly)
        labels = self.query("SELECT " + ",".join(f"sum(target_{i}) positive_{i}" for i in range(1,5)) + ",sum(CASE WHEN target_1+target_2+target_3+target_4>0 THEN 1 ELSE 0 END) any_positive FROM targets")
        self.table("target_counts", labels)
        fig, axes = plt.subplots(1,2,figsize=(12,4))
        axes[0].bar(pd.to_datetime(monthly.cutoff).dt.strftime("%Y-%m"),monthly.positive_pct,color="#2563eb")
        axes[0].tick_params(axis="x",rotation=60); axes[0].set_ylabel("Positive y_any (%)"); axes[0].set_title("All cohort: purchase rate by target month")
        n,p = monthly.rows.sum(),monthly.positive_n.sum()
        axes[1].bar(["y=0","y=1"],[n-p,p],color=["#94a3b8","#f97316"])
        axes[1].set_title(f"Label imbalance: {p/n:.3%} positive"); axes[1].set_ylabel("Customer-month rows")
        self.figure("target_distribution",fig,"Toàn bộ 12 tháng; nhãn có/không mua. Tháng 12/2022–01/2023 chỉ mô tả độ phủ/nhãn, không lựa chọn biến.","univariate")
        coverage = self.query("SELECT trx_ever_seen_source trx, geo_ever_seen_source geo, dialog_ever_seen_source dialog,count(*) rows,100.0*avg(y_any) target_pct FROM train GROUP BY ALL ORDER BY rows DESC")
        self.table("source_coverage",coverage)
        fig,ax = plt.subplots(figsize=(9,4))
        names = coverage.apply(lambda r:f"trx={r.trx:g}, geo={r.geo:g}, dialog={r.dialog:g}",axis=1)
        ax.barh(names,coverage.rows,color="#0d9488"); ax.invert_yaxis(); ax.set_xlabel("Train customer-month rows")
        self.figure("source_coverage",fig,"Độ phủ lịch sử trước cutoff, không phải số khách duy nhất; không loại khách thiếu nguồn.","univariate")
        # Amount distributions use a deterministic hash sample; full moments/quantiles are in CSV.
        amount = self.query("SELECT amount FROM trx WHERE currency=11 AND isfinite(amount) ORDER BY hash(client_id,event_time,amount,42) LIMIT 50000").amount
        fig,axes = plt.subplots(1,2,figsize=(12,4))
        axes[0].hist(amount,bins=60,color="#2563eb"); axes[0].set_xlabel("Raw amount (currency 11)")
        axes[1].hist(np.log1p(amount.clip(lower=0)),bins=60,color="#0d9488"); axes[1].set_xlabel("log1p(amount), currency 11")
        self.figure("amount_histogram",fig,f"Mẫu hash cố định {len(amount):,} giao dịch currency=11, seed 42. Trái giữ đơn vị gốc, phải bổ sung log1p; thống kê bảng dùng toàn bộ log.","univariate")
        sample = self.load_sample()
        chosen = [c for c in ["trx_30d_event_count","trx_30d_main_amount_sum","geo_30d_event_count","dialog_30d_event_count","trx_30d_days_since_last_event","trx_all_event_count"] if c in sample]
        fig,axes = plt.subplots(2,3,figsize=(13,7))
        for ax,c in zip(axes.flat,chosen):
            ax.hist(sample[c].dropna(),bins=45,color="#2563eb");ax.set_title(c,fontsize=8);ax.set_ylabel("Rows")
        self.figure("feature_histograms",fig,f"Train hash sample n={len(sample):,}; giữ trục gốc, null không xuất hiện trong histogram và được đếm riêng.","univariate")
        chosen_cats = [("trx","event_type"),("trx","currency"),("geo","geohash_4"),("trx","event_subtype")]
        fig,axes = plt.subplots(2,2,figsize=(12,8))
        for ax,(source,col) in zip(axes.flat,chosen_cats):
            top = self.cat_top[(source,col)].copy()
            n = int(self.tables["overview"].set_index("table").loc[source,"rows"])
            other = n-int(top.n.sum())
            if other: top = pd.concat([top,pd.DataFrame({"value":["Other"],"n":[other]})],ignore_index=True)
            ax.barh(top.value.fillna("Missing").astype(str),100*top.n/n,color="#0d9488");ax.invert_yaxis();ax.set_title(f"{source}.{col}");ax.set_xlabel("% all rows")
        self.figure("categorical_bars",fig,"Toàn bộ raw; top 10 nhóm + Other, null hiển thị Missing nếu thuộc top. Mã số là biến phân loại ẩn danh.","univariate")
        missing = self.tables["missing_values"].query("table == 'features_train'").nlargest(20,"missing_pct")
        fig,ax=plt.subplots(figsize=(10,6));ax.barh(missing.column,missing.missing_pct,color="#f97316");ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel("Missing (%)")
        self.figure("missing_features",fig,"20 đặc trưng có tỷ lệ thiếu cao nhất trên toàn bộ train; kiểm tra độ phủ và trạng thái nguồn trước khi imputation.","univariate")

    def multivariate(self):
        sample = self.load_sample()
        self.sample_n = len(sample)
        stats = self.tables["feature_statistics_train"].set_index("feature")
        rows = []
        for col in self.predictors:
            c=qi(col)
            r=self.query(f"SELECT count(*) FILTER(WHERE isfinite({c}) AND y_any IS NOT NULL) pair_n, corr(CASE WHEN isfinite({c}) THEN {c} END,y_any) pearson_target, avg({c}) FILTER(WHERE y_any=0 AND isfinite({c})) mean_y0, avg({c}) FILTER(WHERE y_any=1 AND isfinite({c})) mean_y1, avg(y_any) FILTER(WHERE {c} IS NULL OR isnan({c})) missing_target_rate FROM train").iloc[0].to_dict()
            pair=sample[[col,"y_any"]].dropna()
            r.update({"feature":col,"spearman_sample":pair.corr(method="spearman").iloc[0,1] if len(pair)>2 else np.nan,"sample_pair_n":len(pair),"missing_pct":stats.loc[col,"missing_pct"]})
            r["standardized_mean_difference"]=(r["mean_y1"]-r["mean_y0"])/stats.loc[col,"std"] if stats.loc[col,"std"]>0 else np.nan
            rows.append(r)
        assoc=pd.DataFrame(rows)
        assoc["abs_pearson_target"]=assoc.pearson_target.abs()
        assoc=assoc.sort_values("abs_pearson_target",ascending=False,na_position="last")
        self.table("feature_target_associations",assoc)
        valid=[c for c in self.predictors if not stats.loc[c,"constant_observed"] and stats.loc[c,"valid_n"]>2]
        corr=sample[valid].corr(min_periods=100)
        ns=sample[valid].notna().astype(np.int32)
        pair_n=ns.T.dot(ns)
        self.table("pearson_feature_matrix_sample",corr.rename_axis("feature").reset_index())
        self.table("correlation_pair_counts_sample",pair_n.rename_axis("feature").reset_index())
        pairs=[]
        for i,a in enumerate(valid):
            for b in valid[i+1:]:
                r=corr.loc[a,b]
                if pd.notna(r) and abs(r)>=.9:
                    pairs.append({"feature_a":a,"feature_b":b,"pearson_sample":r,"pair_n":int(pair_n.loc[a,b]),"scope":f"train hash sample <= {self.sample_limit}"})
        pairs=pd.DataFrame(pairs,columns=["feature_a","feature_b","pearson_sample","pair_n","scope"])
        if len(pairs):pairs=pairs.iloc[np.argsort(-pairs.pearson_sample.abs().values)]
        self.table("high_correlation_pairs",pairs)
        top=assoc.dropna(subset=["pearson_target"]).head(16).feature.tolist()
        fig,ax=plt.subplots(figsize=(11,9))
        matrix=sample[top+["y_any"]].corr(min_periods=100)
        im=ax.imshow(matrix,vmin=-1,vmax=1,cmap="RdBu_r")
        ax.set_xticks(range(len(matrix)),matrix.columns,rotation=90,fontsize=6);ax.set_yticks(range(len(matrix)),matrix.index,fontsize=6)
        ax.grid(False);fig.colorbar(im,ax=ax,label="Pearson r")
        self.figure("correlation_heatmap",fig,f"Heatmap 16 biến có |Pearson với target| lớn nhất trên train; ma trận hình dùng mẫu hash n={len(sample):,}, pairwise-complete (n cặp trong CSV). Không thay NaN thành 0.","multivariate")
        fig,ax=plt.subplots(figsize=(10,6))
        ax.barh(assoc.head(20).feature,assoc.head(20).pearson_target,color="#2563eb");ax.invert_yaxis();ax.set_xlabel("Pearson correlation with y_any (full train)")
        self.figure("target_associations",fig,"Tương quan với nhãn nhị phân trên toàn bộ train là mô tả liên hệ, chưa phải mức ảnh hưởng nhân quả hoặc feature importance.","multivariate")
        x,y="trx_30d_event_count","trx_30d_main_amount_sum"
        fig,ax=plt.subplots(figsize=(8,5))
        for target,color in [(0,"#94a3b8"),(1,"#ea580c")]:
            s=sample[sample.y_any==target];ax.scatter(s[x],s[y],s=8,alpha=.35,c=color,label=f"y={target} (n={len(s):,})")
        ax.set_xlabel(x);ax.set_ylabel(y);ax.legend()
        self.figure("numeric_scatter",fig,f"Số–số, mẫu hash train {len(sample):,}; trục tuyến tính. Nhãn hiện chỉ nhằm so sánh mô tả.","multivariate")
        chosen=top[:4]
        fig,axes=plt.subplots(1,4,figsize=(14,4))
        for ax,col in zip(axes,chosen):
            vals=[sample.loc[sample.y_any==k,col].dropna().values for k in (0,1)]
            ax.boxplot(vals,tick_labels=["y=0","y=1"],showfliers=True,flierprops={"markersize":2,"alpha":.2})
            ax.set_title(col,fontsize=7)
        self.figure("target_boxplots",fig,f"Số–phân loại trên cùng mẫu hash train n={len(sample):,}; giữ tỷ lệ lớp tự nhiên, không cân bằng lại mẫu. Râu 1,5×IQR.","multivariate")
        bands=[]
        for col in chosen:
            # NTILE only on observed finite values; ties may split, so boundaries are reported.
            c=qi(col)
            r=self.query(f"WITH b AS (SELECT {c} value,y_any,ntile(5) OVER(ORDER BY {c},client_id,cutoff) band FROM train WHERE isfinite({c})) SELECT band,min(value) min,max(value) max,count(*) rows,sum(y_any) positives,100.0*avg(y_any) target_pct FROM b GROUP BY band ORDER BY band")
            r.insert(0,"feature",col);bands.append(r)
        band=pd.concat(bands,ignore_index=True)
        self.table("feature_band_target_rates",band)
        fig,axes=plt.subplots(1,4,figsize=(14,4))
        for ax,col in zip(axes,chosen):
            s=band[band.feature==col];ax.bar(s.band,s.target_pct,color="#0d9488");ax.set_title(col,fontsize=7);ax.set_xlabel("Observed-value quintile");ax.set_ylabel("Positive (%)")
        self.figure("band_target_rates",fig,"Toàn bộ train, 5 nhóm xếp theo giá trị hữu hạn. Giá trị bằng nhau có thể chia nhiều nhóm: đọc min/max trong bảng; liên hệ chưa kiểm soát tháng/khách.","multivariate")
        return assoc,pairs

    def outliers(self):
        rows=[]
        stats=self.tables["feature_statistics_train"]
        for r in stats.itertuples():
            if not r.valid_n: continue
            lo,hi=r.q1-1.5*(r.q3-r.q1),r.q3+1.5*(r.q3-r.q1)
            c=qi(r.feature)
            z=f"abs({c}-{float(r.mean)})>{3*float(r.std)}" if pd.notna(r.std) and r.std>0 else "FALSE"
            counts=self.query(f"SELECT count(*) FILTER (WHERE isfinite({c}) AND ({c}<{lo} OR {c}>{hi})) iqr_n,count(*) FILTER (WHERE isfinite({c}) AND ({z})) zscore_n FROM train").iloc[0]
            rows.append({"feature":r.feature,"valid_n":r.valid_n,"iqr_lower":lo,"iqr_upper":hi,"iqr_zero":r.q1==r.q3,
                         "iqr_n":int(counts.iqr_n),"iqr_pct":100*counts.iqr_n/r.valid_n,"zscore_n":int(counts.zscore_n),"zscore_pct":100*counts.zscore_n/r.valid_n,
                         "zscore_defined":pd.notna(r.std) and r.std>0,"action":"retain; investigate semantics"})
        feature=self.table("feature_outliers_train",pd.DataFrame(rows))
        currency=self.tables["amount_by_currency"].copy()
        self.db.register("currency_summary",currency)
        amounts=self.query("""SELECT t.currency, count(*) valid_n, any_value(s.q1) q1,any_value(s.q3) q3,
            any_value(s.q1-1.5*(s.q3-s.q1)) iqr_lower,any_value(s.q3+1.5*(s.q3-s.q1)) iqr_upper,
            count(*) FILTER(WHERE t.amount<s.q1-1.5*(s.q3-s.q1) OR t.amount>s.q3+1.5*(s.q3-s.q1)) iqr_n,
            count(*) FILTER(WHERE s.std>0 AND abs(t.amount-s.mean)>3*s.std) zscore_n,
            any_value(s.std>0) zscore_defined FROM trx t JOIN currency_summary s ON t.currency IS NOT DISTINCT FROM s.currency
            WHERE isfinite(t.amount) GROUP BY t.currency ORDER BY valid_n DESC""")
        amounts["iqr_pct"]=100*amounts.iqr_n/amounts.valid_n
        amounts["zscore_pct"]=100*amounts.zscore_n/amounts.valid_n
        amounts["action"]="retain original; currency-specific investigation"
        self.table("amount_outliers_by_currency",amounts)
        sample=self.load_sample()
        cols=["trx_30d_event_count","geo_30d_event_count","dialog_30d_event_count"]
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,col in zip(axes,cols):
            ax.boxplot(sample[col].dropna(),showfliers=True,flierprops={"markersize":2,"alpha":.2});ax.set_title(col,fontsize=8)
        self.figure("outlier_boxplots",fig,f"Boxplot trên mẫu hash train n={len(sample):,}; số lượng gắn cờ IQR/Z-score trong CSV được tính trên toàn bộ train. Không tự động xóa/cap.","outliers")
        return feature,amounts

    def conclude(self):
        overview=self.tables["overview"].set_index("table")
        monthly=self.tables["target_monthly"]
        stats=self.tables["feature_statistics_train"]
        assoc=self.tables["feature_target_associations"].dropna(subset=["pearson_target"])
        dupe=self.tables["duplicates"]
        missing=self.tables["missing_values"].query("table in ['clients','targets','trx','geo','dialog']")
        self.conclusions=(f"### Kết quả chính\n\n"
            f"- Có **{int(overview.loc['clients','rows']):,} khách**, **{int(overview.loc['targets','rows']):,} dòng nhãn**, "
            f"{len(self.predictors)} predictor được kiểm tra; EDA đặc trưng dùng **{int(overview.loc['train','rows']):,} dòng train**.\n"
            f"- `y_any` có **{int(monthly.positive_n.sum()):,} nhãn dương / {int(monthly.rows.sum()):,} dòng** "
            f"(**{100*monthly.positive_n.sum()/monthly.rows.sum():.4f}%**). Tỷ lệ theo tháng từ {monthly.positive_pct.min():.3f}% đến {monthly.positive_pct.max():.3f}%. "
            "Dùng PR-AUC/AP, Precision/Recall/Lift theo ngân sách; accuracy đơn thuần ít hữu ích.\n"
            f"- Raw có {int((missing.missing_pct>0).sum())} cột thiếu ở các bảng chính; "
            f"train có {int((stats.missing_pct>0).sum())} predictor thiếu, {int(stats.constant_observed.sum())} predictor hằng trong phần quan sát. "
            "Không loại cohort vì thiếu nguồn. Xem missing_diff_context để phân biệt null phụ trợ với mất nhãn.\n"
            f"- Số dòng raw trùng toàn bộ: {int(dupe.loc[dupe['check']=='full_row','extra_rows'].sum()):,}; "
            f"xung đột nhãn sau DISTINCT: {self.validation['target_conflicting_keys']}. Ứng viên cùng khách–timestamp phải so thuộc tính trước khi kết luận.\n"
            f"- Các liên hệ mạnh nhất theo |Pearson| trong train: "
            + "; ".join(f"`{r.feature}` (r={r.pearson_target:.4f})" for r in assoc.head(5).itertuples()) + ". "
            "Đây là ứng viên feature, không phải bằng chứng nhân quả hay kết quả chọn mô hình.\n"
            f"- Có {len(self.tables['high_correlation_pairs']):,} cặp |r|≥0,90 trong mẫu train. "
            "Các cửa sổ/count/log-count có thể dư thừa; chỉ bỏ dựa trên validation và ý nghĩa biến.\n"
            "- Amount và số đếm có đuôi dài; xem thống kê skewness và tỷ lệ outlier theo biến/currency. "
            "Giữ giá trị gốc, cân nhắc log1p hoặc signed-log trước mô hình tuyến tính. IQR/Z-score chỉ gắn cờ, không xác nhận lỗi.\n\n"
            "### Giới hạn diễn giải\n\n"
            "Khách xuất hiện lặp qua tháng nên các quan sát không độc lập; chưa tính p-value hoặc suy ra hiệu quả chiến dịch. "
            "Correlation pairwise có thể dùng tập dòng khác nhau; số lượng cặp được xuất riêng. "
            "Mẫu hash phục vụ đồ thị/Spearman/ma trận đa biến, không thay thống kê toàn tập. "
            "Chưa có từ điển giải mã nghiệp vụ các mã ẩn danh; không gán tên tiền thật cho currency. "
            "Độ phủ lịch sử không đồng nghĩa thời gian quan sát liên tục của từng khách. "
            "EDA này không huấn luyện hoặc đánh giá tập test; biến phụ trợ nhãn không dùng làm predictor.\n")
        actions=pd.DataFrame([
            ["client_id, fold, cutoff, mon", "Exclude predictors", "ID/partition chỉ để nối, chia tập và audit thời gian"],
            ["target_1..4, y_any, supervised_eligible", "Target/metadata only", "Không đưa vào X; giữ unknown nếu nhãn thiếu"],
            ["targets.trans_count, diff_trans_date", "Exclude pending definition", "Chưa xác minh cửa sổ/cutoff; có nguy cơ leakage"],
            ["Raw event_type/subtype, currency, src/dst, geohash", "Categorical encoding", "CatBoost category hoặc one-hot/hashing fit trên train; unknown/missing riêng; không scale mã danh nghĩa"],
            ["Numeric null", "Missing flags + optional imputation", "Fit median train nếu estimator cần; giữ NaN cho estimator hỗ trợ; không điền nhãn"],
            ["Counts / monetary aggregates", "log1p / signed-log; optional scaling", "Theo currency, không gộp đơn vị; Standard/RobustScaler fit train cho linear/distance, tree không bắt buộc"],
            ["Constant/all-missing predictors", "Review for removal", "Bảng feature_statistics_train; không quyết định từ test; giữ schema nếu pipeline phụ thuộc"],
            ["Highly correlated pairs", "Review redundancy", "Bảng high_correlation_pairs + cỡ mẫu; chọn theo validation và độ ổn định thời gian"],
            ["New behavioral features", "Candidate engineering", "Recency/frequency/active-day rate, share từng loại, xu hướng 30d/84d, flags coverage/missing; tất cả event_time < cutoff"],
            ["Dialog embedding / geo content", "Optional grouped ablation", "Hiện dùng metadata; nếu thêm pooling/entropy/content phải fit trên train, đối chứng và kiểm tra RAM"],
            ["Outliers", "Retain + investigate", "Không xóa/cap chỉ dựa IQR/Z; nếu cap dùng threshold fit train và so validation"],
            ["Evaluation", "Time split + group-aware uncertainty", "Train 02–08, validation 09–11, calibration 12, test khóa 01/2023; bootstrap theo khách khi ước lượng bất định"],
        ],columns=["columns_or_group","recommendation","reason_or_implementation"])
        return self.conclusions,self.table("feature_engineering_plan",actions)

    def export(self):
        # Check assertions against raw and prepared data, rather than assuming historical counts.
        counts=self.query("SELECT count(*) rows, count(DISTINCT client_id) clients, sum(y_any) positives, count(*) FILTER(WHERE y_any IS NULL) unknown FROM features").iloc[0]
        mismatch=int(self.query("SELECT count(*) FROM features f JOIN targets t ON f.client_id=t.client_id AND f.cutoff=date_trunc('month',cast(t.mon AS TIMESTAMP)) WHERE f.y_any IS DISTINCT FROM CASE WHEN t.target_1=1 OR t.target_2=1 OR t.target_3=1 OR t.target_4=1 THEN 1 WHEN t.target_1=0 AND t.target_2=0 AND t.target_3=0 AND t.target_4=0 THEN 0 ELSE NULL END").iloc[0,0])
        joined=int(self.query("SELECT count(*) FROM features f JOIN targets t ON f.client_id=t.client_id AND f.cutoff=date_trunc('month',cast(t.mon AS TIMESTAMP))").iloc[0,0])
        assert mismatch == 0 and joined == int(counts.rows), "Raw/prepared target mismatch or incomplete join"
        assert len(self.tables["feature_statistics_train"])==len(self.predictors)
        assert len(self.tables["feature_target_associations"])==len(self.predictors)
        self.validation.update({"raw_prepared_target_mismatches":mismatch,"raw_prepared_join_rows":joined,
                                "features_rows":int(counts.rows),"unique_customers":int(counts.clients),"positives":int(counts.positives),
                                "unknown_labels":int(counts.unknown),"predictors_described":len(self.predictors),
                                "sample_rows":len(self.sample),"train_max_cutoff":str(self.query("SELECT max(cutoff) FROM train").iloc[0,0])})
        snapshots=self.inventory()
        assert snapshots == self.input_before, "Input metadata changed during EDA"
        self.validation["input_metadata_unchanged"] = True
        fingerprint=hashlib.sha256(json.dumps(snapshots,sort_keys=True).encode()).hexdigest()
        sections=[("1. Tổng quan",["overview","schema"]),("2. Thống kê mô tả",["raw_numeric","categorical_summary","amount_by_currency","feature_statistics_train"]),
                  ("3. Thiếu và trùng lặp",["missing_values","nested_element_quality","dialog_vector_lengths","detail_sequence_reconciliation","duplicates","missing_diff_context","label_conflicts"]),
                  ("4. Đơn biến",["target_monthly","target_counts","source_coverage"]),
                  ("5. Hai biến và đa biến",["feature_target_associations","high_correlation_pairs","feature_band_target_rates"]),
                  ("6. Ngoại lai",["amount_outliers_by_currency","feature_outliers_train"]),
                  ("7. Feature Engineering",["feature_engineering_plan"]) ]
        body=["<h1>EDA MBD-mini — 7 bước</h1>","<p>Ngày 10/11/2026 · Báo cáo tính lại từ raw và output notebook 03.</p>",f"<p>{html.escape(self.scope_note)}</p>",
              "<p>Biểu đồ được nhúng trực tiếp trong file HTML. Bảng dài có thanh cuộn; CSV trong thư mục tables giữ dữ liệu đầy đủ. Không thay đổi raw hoặc output tiền xử lý.</p>",
              "<nav>"+" · ".join(f'<a href="#step{i}">Bước {i}</a>' for i in range(1,8))+"</nav>"]
        md=["# EDA MBD-mini — 7 bước\n",self.scope_note,"\n"+self.conclusions]
        groups={4:"univariate",5:"multivariate",6:"outliers"}
        for i,(title,names) in enumerate(sections,1):
            body.append(f'<section id="step{i}"><h2>{title}</h2>')
            md.append("\n## "+title)
            if i==1:
                body.append("<p>detail là sự kiện, ptls là chuỗi tương ứng; không cộng hai dạng. Numeric storage có thể là categorical code. Head/tail được sắp xếp, ID được hash, mảng chỉ hiển thị độ dài.</p>")
            if i==3:body.append(f"<p>{html.escape(self.missing_note)}</p>")
            if i==5:body.append("<p>Liên hệ Pearson với target dùng toàn bộ train. Spearman/ma trận nhiều biến dùng mẫu hash cố định tối đa 30.000 dòng, pairwise-complete, min 100 cặp cho ma trận. Không suy ra nhân quả. Bảng band chia theo thứ tự giá trị; ties có thể rơi nhiều band.</p>")
            if i==6:body.append("<p>IQR ngoài Q1−1,5×IQR / Q3+1,5×IQR; Z-score |x−mean| &gt; 3×std. Denominator là số giá trị hữu hạn. IQR=0 và std=0 được đánh dấu. Giữ nguyên các giá trị gắn cờ.</p>")
            if i==7:
                # Keep this renderer dependency-free, escape all data before simple Markdown conversion.
                for line in self.conclusions.splitlines():
                    if line.startswith("### "):body.append("<h3>"+html.escape(line[4:])+"</h3>")
                    elif line:
                        rendered=html.escape(line)
                        rendered=re.sub(r"\*\*(.+?)\*\*",r"<strong>\1</strong>",rendered)
                        rendered=re.sub(r"`(.+?)`",r"<code>\1</code>",rendered)
                        body.append("<p>"+rendered+"</p>")
            for name in names:
                frame=self.tables[name]
                body.append(f'<h3>{html.escape(name)}</h3><p><a href="tables/{name}.csv">CSV đầy đủ</a> · {len(frame):,} dòng</p><div class="table">'+frame.to_html(index=False,border=0,float_format=lambda x:f"{x:,.6g}")+"</div>")
                md.append(f"- [{name}](tables/{name}.csv): {len(frame):,} dòng")
            if i==1:
                for name,frame in self.previews.items():
                    body.append(f'<details><summary>Head/tail {name}</summary><div class="table">'+frame.to_html(index=False,border=0)+"</div></details>")
            for name,f in self.figures.items():
                if f["group"]==groups.get(i):
                    encoded=base64.b64encode(f["path"].read_bytes()).decode()
                    body.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{name}"><figcaption>{html.escape(f["caption"])}</figcaption></figure>')
                    md.append(f'\n![{name}](figures/{name}.png)\n\n{f["caption"]}')
            body.append("</section>")
        body.append("<h2>Đối soát</h2><pre>"+html.escape(json.dumps(self.validation,ensure_ascii=False,indent=2))+"</pre>")
        css="body{font:16px/1.6 Segoe UI,Arial,sans-serif;color:#172337;max-width:1280px;margin:40px auto;padding:0 24px;background:#f8fafc}h1,h2,h3{color:#124b78}section{background:white;padding:24px;margin:24px 0;border-radius:12px}a{color:#075bb5}.table{overflow:auto;max-height:650px;border:1px solid #dbe4ed}table{border-collapse:collapse;font-size:12px;white-space:nowrap}th,td{padding:7px 12px;border-bottom:1px solid #e2e8f0;text-align:right}th{position:sticky;top:0;background:#e4eff8}td:first-child,th:first-child{text-align:left}figure{margin:24px 0}img{max-width:100%;height:auto}figcaption{font-size:14px;color:#475569}pre{overflow:auto;background:#eaf1f8;padding:20px}nav{position:sticky;top:0;background:#f8fafc;padding:12px;z-index:2}@media print{.table{max-height:none;overflow:visible}nav{position:static}section{break-before:page}table{white-space:normal;font-size:9px}body{margin:0}}"
        (self.out/"report.html").write_text('<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>EDA MBD-mini</title><style>'+css+'</style><body>'+"\n".join(body)+"</body></html>",encoding="utf-8")
        (self.out/"report.md").write_text("\n".join(md)+"\n",encoding="utf-8")
        (self.out/"input_inventory.json").write_text(json.dumps(snapshots,ensure_ascii=False,indent=2),encoding="utf-8")
        manifest={"status":"ready","created_utc":datetime.now(timezone.utc).isoformat(),"duration_seconds":round(time.time()-self.started,2),
                  "versions":{x:importlib.metadata.version(x) for x in ("duckdb","pandas","numpy","matplotlib","scipy")},
                  "scope":"full raw; features training 2022-02 through 2022-08; all-month target overview only",
                  "input_metadata_sha256":fingerprint,"fingerprint_note":"Paths, sizes and mtimes, not content checksums; no old cache used",
                  "sample":{"method":"ORDER BY hash(client_id, cutoff, 42), client_id, cutoff LIMIT 30000","rows":len(self.sample),"seed":42},
                  "raw_modified":False,"prepared_modified":False,"models_fitted":False,"validation":self.validation,
                  "tables":list(self.tables),"figures":{k:{"caption":v["caption"],"group":v["group"]} for k,v in self.figures.items()}}
        (self.out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
        return self.out/"report.html"

    def close(self):
        self.db.close()
