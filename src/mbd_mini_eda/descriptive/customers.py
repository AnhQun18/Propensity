"""Phân bố hoạt động và tổ hợp nguồn trên toàn bộ bảng khách hàng."""
import pandas as pd
from ..statistics import numeric_columns,numeric_summary,result

from ..models import AnalysisContext, SectionResult


def run(context: AnalysisContext) -> SectionResult:
    frame=context.artifacts["customer_features"]
    summary=pd.DataFrame([numeric_summary(frame[c],feature=c) for c in numeric_columns(frame)])
    combo=(frame.has_trx.astype(int).astype(str)+frame.has_geo.astype(int).astype(str)+frame.has_dialog.astype(int).astype(str))
    coverage=combo.value_counts().rename_axis("trx_geo_dialog").reset_index(name="customers")
    coverage["population"]=len(frame)
    coverage["rate"]=coverage.customers/len(frame) if len(frame) else 0
    return result(context,"descriptive.customers",{"customer_activity":summary,"modality_coverage":coverage},
        grain="customer",population=len(frame),findings=["Tổ hợp 000 giữ các khách không có log; cờ nguồn mô tả toàn lịch sử đã quét."])
