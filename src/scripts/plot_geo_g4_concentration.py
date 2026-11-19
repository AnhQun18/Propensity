"""Compact g4 diversity and top-cell concentration chart from full-history profiles."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/eda_figures_moscow'

def main():
    profiles=pd.read_parquet(BASE/'geo_all_history_customer_profiles.parquet')
    q=profiles[profiles.level.eq(4)]
    n_all=pd.read_parquet(ROOT/'outputs/full_population_temporal_eda_20261118/labels_any.parquet',columns=['client_id']).client_id.nunique()
    assert len(q)==q.client_id.nunique() and n_all>=len(q)
    values=q.unique_cells.to_numpy()
    edges=np.arange(.5,100.6,5)
    counts,_=np.histogram(values,bins=edges)
    overflow=int((values>100).sum())
    share=counts/len(q)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4.7),gridspec_kw={'width_ratios':[1.25,1]})
    bars=ax1.bar((edges[:-1]+edges[1:])/2,share,width=np.diff(edges)*.92,color='#75a8ef',edgecolor='white',linewidth=.5)
    ax1.bar(115,overflow/len(q),width=5,color='#4773aa')
    ax1.set_xticks([1,20,40,60,80,100,115],['1','20','40','60','80','100','>100'])
    ax1.set_xlim(.5,122);ax1.set_xlabel('Số ô geohash_4 khác nhau / khách')
    ax1.set_ylabel('% khách có GEO theo số ô');ax1.yaxis.set_major_formatter(PercentFormatter(1))
    event_axis=ax1.twiny()
    cdf_axis=event_axis.twinx()
    event_axis.set_xlim(0,6000)
    event_axis.set_xlabel('Số sự kiện GEO / khách',color='#d43d3d')
    event_axis.tick_params(axis='x',colors='#d43d3d')
    event_axis.spines['top'].set_color('#d43d3d')
    cdf_axis.set_xlim(0,6000);cdf_axis.set_ylim(0,1)
    event_grid=np.arange(0,6001,100)
    event_sorted=np.sort(q.events.to_numpy())
    cumulative=np.searchsorted(event_sorted,event_grid,side='right')/len(q)
    line,=cdf_axis.plot(event_grid,cumulative,color='#d43d3d',linewidth=2)
    cdf_axis.set_ylabel('% khách có ≤ số sự kiện',color='#d43d3d')
    cdf_axis.yaxis.set_major_formatter(PercentFormatter(1))
    cdf_axis.tick_params(axis='y',colors='#d43d3d')
    ax1.set_title('Đa dạng ô g4',weight='bold');ax1.grid(axis='y',alpha=.2)
    fig.legend([bars,line],['Phân phối số ô g4','Tích lũy theo số sự kiện GEO'],
               loc='lower left',bbox_to_anchor=(.1,.19),ncol=2,frameon=False,fontsize=8)
    for x,threshold in [(.1,20),(.38,40)]:
        pct=(q.events<threshold).mean()
        fig.text(x,.115,f'{pct:.1%} khách có < {threshold} sự kiện GEO',fontsize=11,weight='bold',
                 color='#a92e2e',bbox=dict(facecolor='#fff1f0',edgecolor='#e9b4b1',boxstyle='round,pad=.5'))
    ax1.text(.98,.50,f'Trung vị: {q.unique_cells.median():.0f} ô\nTrung bình: {q.unique_cells.mean():.1f} ô',
             transform=ax1.transAxes,ha='right',va='center',fontsize=9)
    top=q.top_share.mean()
    ax2.pie([top,1-top],startangle=90,counterclock=False,colors=['#3b8ce8','#dbe5f5'],
            wedgeprops={'width':.28,'edgecolor':'white'})
    ax2.text(0,0,f'{top:.1%}',ha='center',va='center',fontsize=18,weight='bold')
    ax2.set_title('Tỷ lệ ở ô chính g4',weight='bold')
    ax2.legend(['Ô nhiều sự kiện nhất','Các ô còn lại'],loc='center left',bbox_to_anchor=(.95,.5),frameon=False,fontsize=9)
    fig.suptitle('GEO: tập trung và đa dạng theo ô geohash_4',weight='bold',fontsize=15)
    fig.text(.02,.02,f'Toàn lịch sử GEO · {len(q):,}/{n_all:,} khách có log; {n_all-len(q):,} khách không GEO ở ngoài hai phép tính. '
             f'Đường đỏ: ≤ mốc sự kiện, đến 6.000; {int((q.events>6000).sum()):,} khách vượt mốc vẫn thuộc mẫu số. '
             f'Vòng tròn: tỷ trọng ô chính trung bình theo khách.',fontsize=8)
    fig.subplots_adjust(left=.08,right=.85,bottom=.32,top=.76,wspace=.45)
    fig.savefig(BASE/'geo_g4_diversity_top1_compact.png',dpi=180,bbox_inches='tight')
    fig.savefig(BASE/'geo_g4_diversity_top1_compact.svg',bbox_inches='tight')
    print(f'customers={len(q)} no_geo={n_all-len(q)} mean_top1={top:.4f}')

if __name__=='__main__':main()
