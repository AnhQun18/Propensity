"""Xu hướng tháng, thay đổi quan sát được, ngày trong tuần và giờ."""
import pandas as pd
from ..engine import engine
from ..statistics import result

from ..models import AnalysisContext, SectionResult


def run(context: AnalysisContext) -> SectionResult:
    store=engine(context)
    monthly=context.section_results["descriptive.temporal"].tables["monthly_activity"].copy()
    monthly=monthly.sort_values(["source","month"])
    monthly["event_change_rate"]=monthly.groupby("source").events.pct_change(fill_method=None)
    monthly["customer_change_rate"]=monthly.groupby("source").customers.pct_change(fill_method=None)
    calendar=[]
    for source,state in store.stats.items():
        n=sum(state.calendar.values())
        for (weekday,hour),count in sorted(state.calendar.items()):
            calendar.append(dict(source=source,weekday=weekday,hour=hour,events=count,denominator=n,
                share=count/n if n else None))
    return result(context,"exploration.temporal",{"temporal_patterns":pd.DataFrame(calendar),
        "activity_changes":monthly},grain="source / period",
        findings=["Tháng đầu/cuối có thể chưa trọn kỳ; thay đổi lượng log chưa chứng minh thay đổi nhu cầu."])
