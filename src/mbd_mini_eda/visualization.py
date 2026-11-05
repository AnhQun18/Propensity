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
    monthly=table(result,"descriptive.temporal","monthly_activity")
    if not monthly.empty:
        fig,ax=figure("Events by month","Month","Events with valid ID/time")
        for color,(source,part) in zip(COLORS,monthly.groupby("source")):
            ax.plot(part.month,part.events,label=source,color=color)
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

def exploration_figures(result):
    figures={}
    calendar=table(result,"exploration.temporal","temporal_patterns")
    if not calendar.empty:
        source=calendar.source.iloc[0]
        part=calendar[calendar.source.eq(source)]
        matrix=part.pivot(index="weekday",columns="hour",values="share").reindex(index=range(7),columns=range(24),fill_value=0).fillna(0)
        fig,ax=figure("Weekday/hour activity: "+source,"Hour","Weekday (Monday=0)")
        im=ax.imshow(matrix,aspect="auto",cmap="Blues")
        fig.colorbar(im,ax=ax,label="Fraction of valid-time events")
        figures["weekday_hour"]=fig
    for section,name,group,title in (
        ("exploration.segments","segment_target_rates","segment","Target fraction by activity segment"),
        ("exploration.modalities","modality_target_rates","combination","Target fraction by source combination")):
        data=table(result,section,name)
        if data.empty:
            continue
        data=data[data.history_days.eq(data.history_days.min())]
        if "segment_feature" in data:
            data=data[data.segment_feature.eq("trx_event_count")]
        data=data.groupby([group,"target"],as_index=False)[["positive_n","labelled_n"]].sum()
        fig,ax=figure(title+" (shortest window)",group,"Positive / observed")
        for color,(target,part) in zip(COLORS,data.groupby("target")):
            ax.plot(part[group].astype(str),part.positive_n/part.labelled_n.replace(0,np.nan),marker="o",label=target,color=color)
        ax.legend(); ax.tick_params(axis="x",rotation=35)
        figures[name]=fig
    missing=table(result,"exploration.missingness","missingness_by_group")
    if not missing.empty:
        part=missing[missing.dimension.eq("month") & missing.column.eq("client_id")]
        if not part.empty:
            fig,ax=figure("Missing client ID over time","Month","Null fraction")
            for color,(source,group) in zip(COLORS,part.groupby("source")):
                ax.plot(group["group"],group.missing_rate,label=source,color=color)
            ax.legend(); ax.tick_params(axis="x",rotation=45)
            figures["missingness_by_month"]=fig
    relations=table(result,"exploration.target_relationships","feature_by_target")
    if not relations.empty:
        part=relations[relations.feature.eq("trx_event_count") & relations.target.eq("target_1")]
        part=part[part.history_days.eq(part.history_days.min())]
        if not part.empty:
            fig,ax=figure("Past transaction count by target_1","Label marker mon","Mean event count")
            for label,group in part.groupby("label"):
                ax.plot(group.mon,group["mean"],marker="o",label="label="+str(label))
            ax.legend(); ax.tick_params(axis="x",rotation=45)
            figures["past_behavior_by_target"]=fig
    corr=table(result,"exploration.correlations","feature_correlations")
    if not corr.empty:
        part=corr[corr.mon.eq(corr.mon.iloc[0]) & corr.history_days.eq(corr.history_days.iloc[0])]
        names=sorted(set(part.feature_a)|set(part.feature_b))
        matrix=pd.DataFrame(np.nan,index=names,columns=names)
        for row in part.itertuples(index=False):
            matrix.loc[row.feature_a,row.feature_b]=row.spearman
            matrix.loc[row.feature_b,row.feature_a]=row.spearman
        fig,ax=figure("Spearman: first month/window (gray = undefined)","","")
        from matplotlib import colormaps
        cmap=colormaps["coolwarm"].with_extremes(bad="#e5e7eb")
        image=ax.imshow(matrix,vmin=-1,vmax=1,cmap=cmap)
        ax.set_xticks(range(len(names)),names,rotation=60,ha="right",fontsize=7)
        ax.set_yticks(range(len(names)),names,fontsize=7)
        fig.colorbar(image,ax=ax)
        figures["feature_correlations"]=fig
    stability=table(result,"exploration.stability","pattern_stability")
    if not stability.empty:
        fig,ax=figure("Variation across month/fold cells","Observed labels per group","Max rate - min rate")
        part=stability.head(1000)
        ax.scatter(part.labelled_n,part.max_rate-part.min_rate,alpha=.5,s=12)
        ax.set_xscale("symlog")
        figures["pattern_stability"]=fig
    return figures
