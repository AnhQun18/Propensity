"""Bảng nội bộ duy nhất theo client_id, giữ mọi khách trong client_split."""
import pandas as pd
from .shared import customer_table
from ..statistics import result

from ..models import AnalysisContext, SectionResult


def run(context: AnalysisContext) -> SectionResult:
    frame=customer_table(context)
    context.artifacts["customer_features"]=frame
    checks=pd.DataFrame([dict(input_customers=len(context.artifacts["clients"]),output_customers=len(frame),
        duplicate_keys=int(frame.client_id.duplicated().sum()),customers_without_log=int(frame.any_active_days.eq(0).sum()))])
    return result(context,"features.customers",{"customer_feature_checks":checks},
        grain="customer",population=len(frame))
