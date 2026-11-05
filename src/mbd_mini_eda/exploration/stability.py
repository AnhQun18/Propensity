"""Độ dao động mô tả; không giả định các tháng của cùng khách độc lập."""
import json
import pandas as pd
from ..statistics import result

from ..models import AnalysisContext, SectionResult


def run(context: AnalysisContext) -> SectionResult:
    summaries,small=[],[]
    for section,table in (("exploration.segments","segment_target_rates"),
                          ("exploration.modalities","modality_target_rates"),
                          ("exploration.target_relationships","target_rates_by_feature_group")):
        rates=context.section_results[section].tables[table]
        if rates.empty:
            continue
        group_cols=[c for c in ("history_days","target","segment_feature","segment","combination","feature","feature_group") if c in rates]
        for keys,part in rates.groupby(group_cols,dropna=False,observed=True,sort=True):
            keys=keys if isinstance(keys,tuple) else (keys,)
            group=dict(zip(group_cols,keys))
            record={**group,"analysis":section,"cells":len(part),"months":part.mon.nunique(),
                "folds":part.fold.nunique(),"min_rate":part.rate.min(),"max_rate":part.rate.max(),
                "std_rate":part.rate.std(),"labelled_n":part.labelled_n.sum(),"positive_n":part.positive_n.sum()}
            summaries.append(record)
        flags=rates[(rates.customers<context.config.min_group_size)|(rates.positive_n<context.config.min_positive_count)].copy()
        flags["analysis"]=section
        small.append(flags)
    return result(context,"exploration.stability",{"pattern_stability":pd.DataFrame(summaries),
        "small_group_flags":pd.concat(small,ignore_index=True) if small else pd.DataFrame()},
        grain="month / fold / history window",min_customers=context.config.min_group_size,
        min_positive_count=context.config.min_positive_count,
        findings=["Độ dao động là mô tả; không tính khoảng tin cậy giả định các khách hàng–tháng độc lập."])
