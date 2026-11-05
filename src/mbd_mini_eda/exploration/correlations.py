"""Spearman của feature cùng cửa sổ/tháng với số cặp hợp lệ."""
import numpy as np
import pandas as pd
from .shared import window_features
from ..statistics import result

from ..models import AnalysisContext, SectionResult


def run(context: AnalysisContext) -> SectionResult:
    frame=context.artifacts["customer_month_features"]
    rows=[]
    for days in context.config.history_days:
        columns=window_features(frame,days)
        for mon,part in frame.groupby("mon",sort=True):
            n=len(part)
            if n>context.config.correlation_sample_size:
                part=part.sample(context.config.correlation_sample_size,random_state=context.config.random_seed)
            clean=part[columns].replace([np.inf,-np.inf],np.nan).astype(float)
            matrix=clean.corr(method="spearman")
            observed=clean.notna().astype("int64")
            counts=observed.T.dot(observed)
            for i,left in enumerate(columns):
                for right in columns[i+1:]:
                    rows.append(dict(mon=mon,history_days=days,feature_a=left,feature_b=right,
                        spearman=matrix.loc[left,right],pair_n=int(counts.loc[left,right]),
                        analyzed_rows=len(part),population_rows=n,
                        scope="full" if len(part)==n else "uniform_row_sample"))
    pairs=pd.DataFrame(rows,columns=["mon","history_days","feature_a","feature_b","spearman","pair_n","analyzed_rows","population_rows","scope"])
    strong=pairs[pairs.spearman.abs().ge(.8)].copy()
    return result(context,"exploration.correlations",{"feature_correlations":pairs,"correlated_feature_pairs":strong},
        grain="customer-month",threshold=.8)
