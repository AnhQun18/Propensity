"""Moment toàn bộ dữ liệu; phân vị, outlier và histogram trên mẫu đã khai báo."""
import numpy as np
import pandas as pd
from ..engine import engine
from ..statistics import result

from ..models import AnalysisContext, SectionResult


def run(context: AnalysisContext) -> SectionResult:
    stats = engine(context).stats["transactions"]
    sample = stats.sample.frame
    values = pd.to_numeric(sample.get("amount",pd.Series(dtype=float)),errors="coerce")
    values = values[np.isfinite(values)].astype(float)
    q = values.quantile([.25,.5,.75,.95,.99])
    scope = "full_scanned_rows" if len(sample)==stats.rows else "uniform_row_sample"
    summary = {**stats.amount.summary(),"scanned_rows":stats.rows,"sample_rows":len(sample),
        "sample_finite_n":len(values),"missing_n":stats.missing["amount"],
        "infinite_n":stats.invalid["infinite_amount"],"p25":q.loc[.25],"median":q.loc[.5],
        "p75":q.loc[.75],"p95":q.loc[.95],"p99":q.loc[.99],
        "moments_scope":"all_scanned_finite_amounts","quantile_scope":scope}
    lower,upper=q.loc[.25]-1.5*(q.loc[.75]-q.loc[.25]),q.loc[.75]+1.5*(q.loc[.75]-q.loc[.25])
    outliers=pd.DataFrame([dict(lower=lower,upper=upper,outlier_n=int(((values<lower)|(values>upper)).sum()),
        analyzed_finite_n=len(values),scope=scope)])
    if len(values):
        counts,edges=np.histogram(values,bins=min(30,max(1,int(np.sqrt(len(values))))))
        hist=pd.DataFrame({"left":edges[:-1],"right":edges[1:],"n":counts,"scope":scope})
    else:
        hist=pd.DataFrame(columns=["left","right","n","scope"])
    return result(context,"descriptive.transactions",{"amount_summary":pd.DataFrame([summary]),
        "transaction_categories":engine(context).categories("transactions"),"transaction_outliers":outliers,
        "amount_histogram":hist},grain="transaction event",quantile_scope=scope,
        findings=["Số tiền được mô tả theo giá trị lưu trong file; chưa quy đổi tiền tệ hoặc diễn giải thành doanh thu."])
