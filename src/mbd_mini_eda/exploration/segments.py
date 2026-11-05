"""Nhóm khách theo hoạt động và tỷ lệ nhãn ở cùng cutoff."""
import pandas as pd
from .shared import feature_group
from ..schema import TARGET_COLUMNS
from ..statistics import numeric_summary,result,target_rates

from ..models import AnalysisContext, SectionResult


def run(context: AnalysisContext) -> SectionResult:
    customers=context.artifacts["customer_features"]
    profiles=[]
    for feature in ("trx_event_count","any_active_days"):
        groups=feature_group(customers[feature],feature)
        for group,part in customers.groupby(groups,observed=True):
            profiles.append(numeric_summary(part[feature],segment_feature=feature,segment=group,
                customers=len(part),feature=feature))
    monthly=context.artifacts["customer_month_features"]
    rates=[]
    for days in context.config.history_days:
        for feature in ("trx_event_count","trx_recency_days","any_active_days"):
            column=feature+"_"+str(days)+"d"
            part=monthly[["client_id","mon","fold",*TARGET_COLUMNS]].copy()
            part["segment"]=feature_group(monthly[column],feature)
            rate=target_rates(part,["mon","fold","segment"],TARGET_COLUMNS,window=days)
            rate["segment_feature"]=feature
            rates.append(rate)
    return result(context,"exploration.segments",{"segment_profiles":pd.DataFrame(profiles),
        "segment_target_rates":pd.concat(rates,ignore_index=True) if rates else pd.DataFrame()},
        grain="customer / customer-month",group_boundaries="fixed; intervals in group labels")
