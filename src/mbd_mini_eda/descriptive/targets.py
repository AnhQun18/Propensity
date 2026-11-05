"""Bốn nhãn độc lập: tách tỷ lệ theo dòng và tỷ lệ theo khách."""
from itertools import combinations
import numpy as np
import pandas as pd
from ..schema import TARGET_COLUMNS
from ..statistics import result,target_rates

from ..models import AnalysisContext, SectionResult


def run(context: AnalysisContext) -> SectionResult:
    frame=context.artifacts["targets"]
    rows=[]
    for target in TARGET_COLUMNS:
        n=int(frame[target].notna().sum()); positive=int(frame[target].eq(1).sum())
        rows.append(dict(target=target,rows=len(frame),labelled_n=n,positive_n=positive,negative_n=n-positive,
            missing_n=len(frame)-n,rate=positive/n if n else np.nan))
    grouped=target_rates(frame,["mon","fold"],TARGET_COLUMNS)
    pairs=[]
    for left,right in combinations(TARGET_COLUMNS,2):
        valid=frame[[left,right]].notna().all(axis=1)
        pairs.append(dict(left=left,right=right,observed_pairs=int(valid.sum()),
            both_positive=int((frame[left].eq(1)&frame[right].eq(1)).sum())))
    customers=frame.groupby("client_id")[list(TARGET_COLUMNS)].agg(["sum","count"])
    history=[]
    for target in TARGET_COLUMNS:
        observed=customers[(target,"count")]>0
        positive=customers[(target,"sum")]>0
        history.append(dict(target=target,customers=len(customers),observed_customers=int(observed.sum()),
            ever_positive_customers=int(positive.sum()),
            rate=int(positive.sum())/int(observed.sum()) if observed.any() else np.nan,
            mean_positive_months=customers.loc[observed,(target,"sum")].mean()))
    positive_counts=frame[list(TARGET_COLUMNS)].eq(1).sum(axis=1).value_counts().sort_index()
    return result(context,"descriptive.targets",{"target_prevalence":pd.DataFrame(rows),"target_by_month_fold":grouped,
        "target_cooccurrence":pd.DataFrame(pairs),"customer_positive_history":pd.DataFrame(history),
        "positive_labels_per_row":positive_counts.rename_axis("positive_labels").reset_index(name="rows")},
        grain="customer-month / customer",scope="full_reference_rows")
