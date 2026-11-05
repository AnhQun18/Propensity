"""Feature lịch sử có đúng một dòng mỗi khách hàng–tháng."""
import pandas as pd
from .shared import customer_table
from ..io import validate_label_windows
from ..engine import engine
from ..schema import TARGET_COLUMNS
from ..statistics import result

from ..models import AnalysisContext, SectionResult


def run(context: AnalysisContext) -> SectionResult:
    windows=validate_label_windows(context.artifacts["label_windows"],context.config.timezone)
    labels=context.artifacts["targets"]
    if context.config.analysis_months:
        missing=set(context.config.analysis_months)-set(labels.mon)
        if missing:
            raise ValueError("analysis_months không có trong targets: "+str(sorted(missing)))
        labels=labels[labels.mon.isin(context.config.analysis_months)]
    if set(labels.mon)-set(windows.mon):
        raise ValueError("label_windows thiếu mapping cho tháng được phân tích")
    outputs,checks,coverage=[],[],[]
    for row in windows[windows.mon.isin(labels.mon)].itertuples(index=False):
        base=labels.loc[labels.mon.eq(row.mon),["client_id","mon","fold",*TARGET_COLUMNS]].copy()
        base["cutoff"],base["label_start"],base["label_end"]=row.cutoff,row.label_start,row.label_end
        for days in context.config.history_days:
            start=row.cutoff-pd.DateOffset(days=days)
            for source,state in engine(context).stats.items():
                first=state.start.normalize() if state.start is not None else None
                end=state.end.normalize()+pd.DateOffset(days=1) if state.end is not None else None
                overlap=max(0,(min(row.cutoff,end).date()-max(start,first).date()).days) if first is not None else 0
                coverage.append(dict(mon=row.mon,history_days=days,source=source,
                    history_start=start,cutoff=row.cutoff,source_first_event=state.start,
                    source_last_event=state.end,days_within_global_source_bounds=overlap,
                    window_days=days,observation_continuity_verified=False))
            features=customer_table(context,start=start,end=row.cutoff).drop(columns="fold")
            last_cols=[c for c in features if c.endswith("_last_event")]
            violations=sum(int(features[c].ge(row.cutoff).sum()) for c in last_cols)
            if violations:
                raise ValueError("Feature chứa sự kiện tại hoặc sau cutoff")
            columns={c:c+"_"+str(days)+"d" for c in features if c!="client_id"}
            base=base.merge(features.rename(columns=columns),on="client_id",how="left",validate="one_to_one")
            checks.append(dict(mon=row.mon,history_days=days,history_start=start,cutoff=row.cutoff,
                label_start=row.label_start,label_end=row.label_end,input_rows=int(labels.mon.eq(row.mon).sum()),
                output_rows=len(base),future_event_violations=violations,
                customers_with_log=int(base["any_active_days_"+str(days)+"d"].gt(0).sum())))
        outputs.append(base)
    frame=pd.concat(outputs,ignore_index=True) if outputs else pd.DataFrame()
    if not frame.empty and frame.duplicated(["client_id","mon"]).any():
        raise ValueError("Khóa khách hàng–tháng không duy nhất")
    context.artifacts["customer_month_features"]=frame
    return result(context,"features.customer_month",{"customer_month_checks":pd.DataFrame(checks),
        "source_window_coverage":pd.DataFrame(coverage)},
        grain="customer-month",population=len(frame),label_alignment=context.runtime.get("label_alignment",context.config.label_alignment),
        findings=["Chỉ dùng lịch sử trước cutoff; khách không có log vẫn được giữ. Mapping mon và timezone là lựa chọn cấu hình."])
