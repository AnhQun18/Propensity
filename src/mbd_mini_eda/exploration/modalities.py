"""Độ phủ tổ hợp nguồn và tỷ lệ nhãn của các nhóm."""
import pandas as pd
from .shared import source_combination
from ..schema import TARGET_COLUMNS
from ..statistics import result,target_rates

from ..models import AnalysisContext, SectionResult


def run(context: AnalysisContext) -> SectionResult:
    customers=context.artifacts["customer_features"]
    overlap=source_combination(customers).value_counts().rename_axis("combination").reset_index(name="customers")
    overlap["population"]=len(customers)
    overlap["rate"]=overlap.customers/len(customers) if len(customers) else 0
    monthly=context.artifacts["customer_month_features"]
    rates=[]
    for days in context.config.history_days:
        part=monthly[["client_id","mon","fold",*TARGET_COLUMNS]].copy()
        part["combination"]=source_combination(monthly,"_"+str(days)+"d")
        rates.append(target_rates(part,["mon","fold","combination"],TARGET_COLUMNS,window=days))
    return result(context,"exploration.modalities",{"modality_overlap":overlap,
        "modality_target_rates":pd.concat(rates,ignore_index=True) if rates else pd.DataFrame()},
        grain="customer / customer-month",combination_order="trx,geo,dialog",
        findings=["Tổ hợp nguồn liên hệ với nhãn được xác định từ chính cửa sổ trước cutoff."])
