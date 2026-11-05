"""Phạm vi thời gian, số khách hoạt động và phân phối độ dài lịch sử."""
import pandas as pd
from ..engine import engine
from ..statistics import numeric_summary, result

from ..models import AnalysisContext, SectionResult


def run(context: AnalysisContext) -> SectionResult:
    store = engine(context)
    ranges = []
    for name, stats in store.stats.items():
        ranges.append(dict(source=name,first_event=stats.start,last_event=stats.end,scanned_rows=stats.rows,
            valid_time_rows=stats.rows-stats.missing["event_time"]-stats.invalid["invalid_event_time"]))
    monthly = store.query("""SELECT source,substr(day,1,7) month,SUM(event_n) events,
        COUNT(DISTINCT client_id) customers,COUNT(DISTINCT day) observed_days
        FROM daily GROUP BY source,substr(day,1,7) ORDER BY source,month""")
    monthly["events_per_active_customer"] = monthly.events / monthly.customers
    aggregates = store.aggregate()
    rows = []
    for source, group in aggregates.groupby("source"):
        span = (group.last_ns - group.first_ns) / 86400e9
        rows.extend([numeric_summary(span,source=source,metric="span_days"),
                     numeric_summary(group.active_days,source=source,metric="active_days")])
    return result(context,"descriptive.temporal",{"source_time_ranges":pd.DataFrame(ranges),
        "monthly_activity":monthly,"history_length_summary":pd.DataFrame(rows)},
        findings=["Span giữa sự kiện đầu/cuối khác với số ngày có log. Số khách theo tháng chỉ tính ID có timestamp hợp lệ."],
        grain="source / month / customer")
