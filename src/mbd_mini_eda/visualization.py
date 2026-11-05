"""Biểu đồ từ bảng kết quả; dùng Figure trực tiếp, không đổi backend khi import."""
import numpy as np
import pandas as pd

COLORS=("#2563eb","#059669","#d97706","#9333ea")

def table(result,section,name):
    item=result.sections.get(section)
    return item.tables.get(name,pd.DataFrame()) if item else pd.DataFrame()

def figure(title,xlabel="",ylabel=""):
    from matplotlib.figure import Figure
    fig=Figure(figsize=(9,5),layout="constrained")
    ax=fig.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig,ax

def bars(frame,x,y,title):
    fig,ax=figure(title,x,y)
    ax.bar(frame[x].astype(str),frame[y],color=COLORS[0])
    ax.tick_params(axis="x",rotation=45)
    return fig

def descriptive_figures(result):
    figures={}
    missing=table(result,"descriptive.quality","missing_values")
    if not missing.empty:
        candidates=missing[missing.missing_n.gt(0)]
        part=(candidates if not candidates.empty else missing).nlargest(20,"missing_rate").copy()
        part["field"]=part.source+"."+part.column
        fig,ax=figure("Missing attributes (top 20)","Missing fraction","Field")
        ax.barh(part.field.iloc[::-1],part.missing_rate.iloc[::-1],color=COLORS[0])
        figures["missing_attributes"]=fig
    hist=table(result,"descriptive.transactions","amount_histogram")
    if not hist.empty:
        fig,ax=figure("Amount distribution ("+str(hist.scope.iloc[0])+")","Stored amount","Sample events")
        ax.bar(hist.left,hist.n,width=hist.right-hist.left,align="edge",color=COLORS[0])
        figures["amount_histogram"]=fig
    amount=table(result,"descriptive.transactions","amount_summary")
    if not amount.empty and amount[["p25","median","p75"]].notna().all(axis=None):
        r=amount.iloc[0]
        fig,ax=figure("Amount quantiles ("+str(r.quantile_scope)+")","Stored amount","")
        # Whiskers dùng P95/P99 thay vì giả lập phân phối từ các quantile.
        ax.plot([r.p25,r.p75],[1,1],linewidth=12,color=COLORS[0],alpha=.35)
        ax.scatter([r["median"],r.p95,r.p99],[1,1,1],color=[COLORS[0],COLORS[2],COLORS[3]])
        ax.set_yticks([])
        ax.text(r["median"],1.04,"median"); ax.text(r.p95,.96,"P95"); ax.text(r.p99,1.04,"P99")
        figures["amount_quantiles"]=fig
        fig,ax=figure("Amount boxplot ("+str(r.quantile_scope)+"; outlier points omitted)","Stored amount","")
        ax.bxp([dict(med=r["median"],q1=r.p25,q3=r.p75,whislo=r.whisker_low,
                     whishi=r.whisker_high,fliers=[])],showfliers=False,orientation="horizontal")
        ax.set_yticks([])
        figures["amount_boxplot"]=fig
    categories=table(result,"descriptive.transactions","transaction_categories")
    if not categories.empty:
        part=categories[categories.kind.eq("value") & categories.column.eq("event_type")].head(20)
        if not part.empty:
            figures["transaction_event_types"]=bars(part,"value","n","Transaction event types")
    ranges=table(result,"descriptive.temporal","source_time_ranges")
    if not ranges.empty:
        part=ranges.dropna(subset=["first_event","last_event"])
        if not part.empty:
            from matplotlib import dates
            first=dates.date2num(pd.to_datetime(part.first_event).dt.to_pydatetime())
            last=dates.date2num(pd.to_datetime(part.last_event).dt.to_pydatetime())
            fig,ax=figure("Observed time range by source","Date","Source")
            ax.barh(part.source,last-first,left=first,color=COLORS[0])
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m"))
            ax.tick_params(axis="x",rotation=45)
            figures["source_timeline"]=fig
    monthly=table(result,"descriptive.temporal","monthly_activity")
    if not monthly.empty:
        from matplotlib import dates
        fig,ax=figure("Events by month","Month","Events with valid ID/time")
        for color,(source,part) in zip(COLORS,monthly.groupby("source")):
            part=part.assign(month=pd.to_datetime(part.month)).sort_values("month")
            ax.plot(part.month,part.events,label=source,color=color)
        ax.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m"))
        ax.legend(); ax.tick_params(axis="x",rotation=45)
        figures["monthly_activity"]=fig
    targets=table(result,"descriptive.targets","target_prevalence")
    if not targets.empty:
        figures["target_prevalence"]=bars(targets,"target","rate","Positive label fraction (observed labels)")
    rates=table(result,"descriptive.targets","target_by_month_fold")
    if not rates.empty:
        data=rates.groupby(["mon","target"],as_index=False)[["positive_n","labelled_n"]].sum()
        fig,ax=figure("Positive labels by month","Label marker mon","Positive / observed")
        for color,(target,part) in zip(COLORS,data.groupby("target")):
            ax.plot(part.mon,part.positive_n/part.labelled_n.replace(0,np.nan),label=target,color=color)
        ax.legend(); ax.tick_params(axis="x",rotation=45)
        figures["target_by_month"]=fig
    coverage=table(result,"descriptive.customers","modality_coverage")
    if not coverage.empty:
        figures["source_coverage"]=bars(coverage,"trx_geo_dialog","customers","Source combination (trx, geo, dialog)")
    activity=table(result,"descriptive.customers","customer_activity")
    if not activity.empty:
        part=activity[activity.feature.isin(["trx_event_count","geo_event_count","dialog_event_count"])].copy()
        if not part.empty:
            figures["customer_event_medians"]=bars(part,"feature","median","Events per customer: medians, including zero-log customers")
    categories=table(result,"descriptive.modalities","location_categories")
    if not categories.empty:
        part=categories[categories.kind.eq("value") & categories.column.eq("geohash_6")].head(10)
        if not part.empty:
            figures["location_codes"]=bars(part,"value","n","Most frequent encoded location values")
    embeddings=table(result,"descriptive.modalities","embedding_quality")
    if not embeddings.empty:
        part=embeddings[embeddings.check.str.startswith("dimension_")].copy()
        if not part.empty:
            figures["embedding_dimensions"]=bars(part,"check","n","Embedding dimensions")
    return figures
