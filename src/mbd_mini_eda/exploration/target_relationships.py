"""So sánh hành vi lịch sử giữa nhãn 0/1, có mẫu số theo tháng."""
import pandas as pd
from .shared import feature_group,window_features
from ..schema import TARGET_COLUMNS
from ..statistics import numeric_summary,result,target_rates

from ..models import AnalysisContext, SectionResult


def run(context: AnalysisContext) -> SectionResult:
    frame=context.artifacts["customer_month_features"]
    summaries,rates=[],[]
    for days in context.config.history_days:
        for column in window_features(frame,days):
            feature=column.removesuffix("_"+str(days)+"d")
            for target in TARGET_COLUMNS:
                for (mon,label),part in frame.groupby(["mon",target],observed=True,dropna=True):
                    summaries.append(numeric_summary(part[column],feature=feature,target=target,label=label,
                        mon=mon,history_days=days,customers=part.client_id.nunique(),
                        positive_n=int(part[target].eq(1).sum())))
            part=frame[["client_id","mon","fold",*TARGET_COLUMNS]].copy()
            part["feature_group"]=feature_group(frame[column],feature)
            grouped=target_rates(part,["mon","fold","feature_group"],TARGET_COLUMNS,window=days)
            grouped["feature"]=feature
            rates.append(grouped)
    return result(context,"exploration.target_relationships",{"feature_by_target":pd.DataFrame(summaries),
        "target_rates_by_feature_group":pd.concat(rates,ignore_index=True) if rates else pd.DataFrame()},
        grain="customer-month",findings=["Các bảng mô tả liên hệ, không phải tác động nhân quả. Một khách có thể đóng góp nhiều tháng."])
