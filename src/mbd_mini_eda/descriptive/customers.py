"""Phân bố hoạt động và tổ hợp nguồn trên toàn bộ bảng khách hàng."""
import pandas as pd
from ..statistics import numeric_columns,numeric_summary,result
from ..aggregation import customer_table

from ..models import AnalysisContext, SectionResult


def run(context: AnalysisContext) -> SectionResult:
    frame=customer_table(context)
    summary=pd.DataFrame([numeric_summary(frame[c],feature=c) for c in numeric_columns(frame)])
    combo=(frame.has_trx.astype(int).astype(str)+frame.has_geo.astype(int).astype(str)+frame.has_dialog.astype(int).astype(str))
    coverage=combo.value_counts().rename_axis("trx_geo_dialog").reset_index(name="customers")
    coverage["population"]=len(frame)
    coverage["rate"]=coverage.customers/len(frame) if len(frame) else 0
    checks=pd.DataFrame([dict(input_customers=len(context.artifacts["clients"]),output_customers=len(frame),
        duplicate_keys=int(frame.client_id.duplicated().sum()),customers_without_log=int(frame.any_active_days.eq(0).sum()))])
    return result(context,"descriptive.customers",{"customer_activity":summary,"modality_coverage":coverage,
        "customer_aggregation_checks":checks},
        grain="customer",population=len(frame),findings=["Tổ hợp 000 giữ các khách không có log; cờ nguồn mô tả toàn lịch sử đã quét."])
