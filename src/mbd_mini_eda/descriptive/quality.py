"""Chất lượng được kiểm tra trên toàn bộ các dòng đã quét."""
import numpy as np
import pandas as pd
from ..engine import engine
from ..schema import TARGET_COLUMNS
from ..statistics import result

from ..models import AnalysisContext, SectionResult


def run(context: AnalysisContext) -> SectionResult:
    store = engine(context)
    missing, invalid, duplicates, coverage = [], [], [], []
    clients = context.artifacts["clients"]
    for name in ("clients", "targets"):
        frame = context.artifacts[name]
        for column in frame:
            n = int(frame[column].isna().sum())
            missing.append(dict(source=name,column=column,rows=len(frame),missing_n=n,missing_rate=n/len(frame) if len(frame) else np.nan))
        keys = ["client_id", "mon"] if name == "targets" else ["client_id"]
        duplicates.append(dict(source=name,kind="unique_key",duplicate_n=int(frame.duplicated(keys).sum()),analyzed_rows=len(frame),scope="full"))
    for name, stats in store.stats.items():
        for column, n in stats.missing.items():
            missing.append(dict(source=name,column=column,rows=stats.rows,missing_n=n,missing_rate=n/stats.rows if stats.rows else np.nan))
        invalid.extend(dict(source=name,check=k,n=v) for k,v in stats.invalid.items())
        duplicates.append(dict(source=name,kind="full_row",duplicate_n=stats.duplicate_n,
            analyzed_rows=stats.duplicate_rows,scope=stats.duplicate_scope))
        ids = store.query("SELECT DISTINCT client_id FROM daily WHERE source=?", (name,)).client_id
        matched = int(clients.client_id.isin(ids).sum())
        coverage.append(dict(source=name,reference_customers=len(clients),customers_with_valid_time_log=matched,
            distinct_ids_with_valid_time=len(ids),
            customers_without_valid_time_log=len(clients)-matched,coverage=matched/len(clients) if len(clients) else np.nan,
            unknown_client_rows=stats.invalid["unknown_client_rows"],missing_id_rows=stats.missing["client_id"]))
    return result(context, "descriptive.quality", {"missing_values":pd.DataFrame(missing),
        "invalid_values":pd.DataFrame(invalid,columns=["source","check","n"]),
        "duplicates":pd.DataFrame(duplicates),"key_coverage":pd.DataFrame(coverage)},
        findings=["Trùng nguyên dòng được đếm sau lần xuất hiện đầu; scope của từng phép kiểm tra nằm trong bảng duplicates."],
        grain="column / key", duplicate_mode=context.config.duplicate_mode)
