"""Vị trí, hội thoại và kiểm tra vector embedding."""
import pandas as pd
from ..engine import engine
from ..statistics import numeric_summary,result

from ..models import AnalysisContext, SectionResult


def run(context: AnalysisContext) -> SectionResult:
    store=engine(context)
    aggregates=store.aggregate()
    clients=context.artifacts["clients"]
    summaries={}
    for source in ("geo","dialog"):
        group=clients[["client_id"]].merge(aggregates[aggregates.source.eq(source)],on="client_id",how="left",validate="one_to_one")
        rows=[numeric_summary(group[column].fillna(0),metric=column) for column in ("event_count","active_days")]
        if source=="geo":
            locations=clients[["client_id"]].merge(store.unique_locations(),on="client_id",how="left",validate="one_to_one")
            rows.append(numeric_summary(locations.geo_unique_locations.fillna(0),metric="unique_locations"))
        summaries[source]=pd.DataFrame(rows)
    state=store.stats["dialog"]
    checks=[dict(check=k,n=v) for k,v in state.invalid.items() if "embedding" in k]
    checks.extend(dict(check="dimension_"+str(k),n=v) for k,v in state.embedding_lengths.items())
    norm=state.norms.summary()
    for k,v in norm.items():
        checks.append(dict(check="valid_embedding_norm_"+k,value=v))
    return result(context,"descriptive.modalities",{"geo_summary":summaries["geo"],
        "location_categories":store.categories("geo"),"dialog_summary":summaries["dialog"],
        "embedding_quality":pd.DataFrame(checks)},grain="event / customer",
        location_column=store.stats["geo"].location_column,embedding_size=context.config.embedding_size)
