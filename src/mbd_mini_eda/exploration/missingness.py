"""Phân biệt null thuộc tính với không có log trong lịch quan sát."""
import numpy as np
import pandas as pd
from ..engine import engine,PREFIXES
from ..statistics import result

from ..models import AnalysisContext, SectionResult


def run(context: AnalysisContext) -> SectionResult:
    store=engine(context)
    rows,patterns=[],[]
    for source,state in store.stats.items():
        for (dimension,group,column),n in sorted(state.missing_groups.items()):
            total=state.group_rows[(dimension,group)]
            rows.append(dict(source=source,dimension=dimension,group=group,column=column,rows=total,
                missing_n=n,missing_rate=n/total if total else np.nan))
        customers=context.artifacts["customer_features"]
        if state.start is None:
            patterns.append(dict(source=source,metric="no_log",group="empty_source",customers=len(customers),denominator=len(customers)))
            continue
        days=(state.end.date()-state.start.date()).days+1
        active=customers[PREFIXES[source]+"_active_days"]
        ratio=1-active/days
        bins=pd.cut(ratio,[-.001,0,.5,1],labels=["0","(0,0.5]","(0.5,1]"]).astype("string")
        for group,n in bins.value_counts().items():
            patterns.append(dict(source=source,metric="fraction_days_without_log",group=group,
                customers=n,denominator=len(customers),calendar_days=days))
    return result(context,"exploration.missingness",{"missingness_patterns":pd.DataFrame(patterns),
        "missingness_by_group":pd.DataFrame(rows)},grain="source / period / group",
        findings=["Ngày không có log mô tả độ thưa quan sát, không phải tỷ lệ thất thoát dữ liệu thu thập."])
