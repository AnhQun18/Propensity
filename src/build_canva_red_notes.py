"""Revised pages for the user's annotated 20-page Canva pitch deck.

Reads exported full-history aggregates only. Never modifies raw data or Canva.
"""
from pathlib import Path
import base64
import html
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/canva_red_notes_20261107'
T = ROOT / 'docs/mbd_mini_eda/tables'
OUT.mkdir(parents=True, exist_ok=True)
GREEN, MINT, CORAL, INK, MUTED = '#267969', '#80b89c', '#d38368', '#173d38', '#637970'
COLORS = [GREEN, MINT, CORAL, '#7693b0']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'text.color':INK,
                     'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,
                     'axes.spines.top':False,'axes.spines.right':False})
def read(s): return pd.read_csv(T / (s+'.csv'))
def vi(v, decimals=0):
    return f'{v:,.{decimals}f}'.replace(',', '~').replace('.', ',').replace('~', '.')
def clean(ax):
    ax.spines[['left','bottom']].set_color('#d5dfd9')
    ax.grid(axis='y',color='#e7ece8',lw=.8)
    ax.set_axisbelow(True)
def save(fig,name):
    fig.savefig(OUT / f'{name}.png',dpi=160,facecolor='white')
    plt.close(fig)
def img(name,cls='chart'):
    b64=base64.b64encode((OUT/f'{name}.png').read_bytes()).decode()
    return f'<img class="{cls}" src="data:image/png;base64,{b64}"/>'

# Full-history monthly customers, stacked vertically to match the daily page.
monthly=read('review/monthly_coverage')
fig,axes=plt.subplots(3,1,figsize=(16.96,6.0),sharex=True)
for ax,s,label,c in zip(axes,['trx','geo','dialog'],['Giao dịch','Vị trí','Hội thoại'],COLORS):
    d=monthly[monthly.source.eq(s)].sort_values('month').copy()
    dates=pd.to_datetime(d.month)
    values=d.customers_with_log.to_numpy(dtype=float)
    # Before 2022, no observed geo/dialog may mean unobserved source.
    # Keep those cells as gaps, rather than inventing a zero activity count.
    values[(d.month.lt('2022-01') & d.events.eq(0)).to_numpy()]=np.nan
    ax.plot(dates,values,color=c,lw=2.0,marker='o',ms=3.5)
    ax.set_ylim(0,float(np.nanmax(values))*1.12)
    ax.set_ylabel(label)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:vi(x)))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    clean(ax)
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3,6,9,12]))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
axes[-1].set_xlim(pd.Timestamp('2020-11-15'),pd.Timestamp('2023-01-15'))
fig.subplots_adjust(left=.10,right=.97,bottom=.12,top=.96,hspace=.24)
save(fig,'monthly_zoom')

targets=read('review/target_monthly')
assert targets.labelled_n.eq(100224).all()
fig,axes=plt.subplots(2,2,figsize=(16.96,6.5))
target_ranges=[]
for ax,s,c in zip(axes.flat,['target_1','target_2','target_3','target_4'],COLORS):
    d=targets[targets.target.eq(s)].sort_values('mon')
    y=d.positive_percent.to_numpy(); x=pd.to_datetime(d.mon)
    ax.plot(x,y,color=c,lw=2.6,marker='o',ms=4)
    pad=max(np.ptp(y)*.2,.003)
    ax.set_ylim(max(0,min(y)-pad),max(y)+pad)
    ax.set_title(s,loc='left',fontweight='bold',pad=12)
    ax.set_ylabel('Nhãn dương (%)')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:vi(v,3)))
    ax.yaxis.set_major_locator(MaxNLocator(4));clean(ax)
    target_ranges.append({'target':s,'min':float(y.min()),'max':float(y.max()),'ylim':list(ax.get_ylim())})
fig.subplots_adjust(left=.07,right=.98,bottom=.10,top=.93,hspace=.62,wspace=.24)
save(fig,'targets_zoom')

# True datetime geometry with flat ends and endpoint ticks.
ranges=read('descriptive.temporal/source_time_ranges')
fig,ax=plt.subplots(figsize=(16.96,5.5))
timeline=[('transactions','Giao dịch',GREEN),('dialog','Hội thoại',CORAL),('geo','Vị trí',MINT)]
for row,(s,label,c) in enumerate(timeline):
    d=ranges[ranges.source.eq(s)].iloc[0]
    start,end=pd.Timestamp(d.first_event),pd.Timestamp(d.last_event)
    y=3-row
    ax.plot([start,end],[y,y],lw=8,color=c,solid_capstyle='butt')
    for time in [start,end]: ax.plot([time,time],[y-.13,y+.13],lw=2,color=INK)
    ax.text(start,y+.22,start.strftime('%d/%m/%Y'),fontsize=12,ha='left',color=MUTED)
    ax.text(end,y+.22,end.strftime('%d/%m/%Y'),fontsize=12,ha='right',color=MUTED)
start,end=pd.Timestamp('2022-02-28'),pd.Timestamp('2023-01-31')
ax.plot([start,end],[0,0],color=COLORS[3],lw=8,solid_capstyle='butt')
for time in [start,end]: ax.plot([time,time],[-.13,.13],lw=2,color=INK)
ax.text(start,.22,'28/02/2022',fontsize=12,color=MUTED)
ax.text(end,.22,'31/01/2023',fontsize=12,ha='right',color=MUTED)
ax.set_yticks([3,2,1,0],['Giao dịch','Hội thoại','Vị trí','Nhãn / mon'])
ax.set_ylim(-.55,3.65);ax.set_xlim(pd.Timestamp('2020-12-01'),pd.Timestamp('2023-03-01'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3));ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
ax.grid(axis='x',color='#e7ece8');ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0,pad=20)
fig.subplots_adjust(left=.11,right=.99,top=.98,bottom=.15);save(fig,'timeline')

# Linear quantile intervals: shaded IQR, median tick, separate labelled P95.
eventstats=read('review/events_per_customer')
eventstats=eventstats[eventstats.population.eq('all_customers') & eventstats.source.isin(['trx','geo','dialog'])]
fig,axes=plt.subplots(3,1,figsize=(16.96,5.9))
for ax,s,label,c in zip(axes,['trx','geo','dialog'],['Giao dịch','Vị trí','Hội thoại'],COLORS):
    d=eventstats[eventstats.source.eq(s)].iloc[0]
    ax.hlines(.5,0,d.p95,color='#ccd9d0',lw=3)
    ax.plot([d.p25,d.p75],[.5,.5],lw=24,color=c,alpha=.5,solid_capstyle='butt')
    ax.vlines(d['median'],.24,.76,color=INK,lw=3)
    ax.plot(d.p95,.5,'o',color=c,ms=8)
    ax.text(0,1.12,label,transform=ax.transAxes,fontweight='bold',fontsize=14)
    ax.text(.99,1.12,f"P25 {vi(d.p25)}   ·   P50 {vi(d['median'])}   ·   P75 {vi(d.p75)}   ·   P95 {vi(d.p95)}",ha='right',transform=ax.transAxes,fontsize=12)
    ax.set_xlim(0,d.p95*1.12);ax.set_ylim(0,1);ax.set_yticks([])
    ax.spines[['top','left','right']].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(6));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:vi(x)))
fig.subplots_adjust(left=.035,right=.99,bottom=.08,top=.88,hspace=1.2);save(fig,'quantiles')

# Full-history daily counts; no log transformation, daily observations are retained.
daily=read('review/daily_behavior')
fig,axes=plt.subplots(3,1,figsize=(16.96,6.0),sharex=True)
for ax,s,label,c in zip(axes,['trx','geo','dialog'],['Giao dịch','Vị trí','Hội thoại'],COLORS):
    d=daily[daily.source.eq(s)].copy();d.day=pd.to_datetime(d.day);d=d.sort_values('day')
    ax.plot(d.day,d.events,color=c,lw=1.1)
    ax.set_ylabel(label);ax.set_ylim(bottom=0);clean(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:vi(x)));ax.yaxis.set_major_locator(MaxNLocator(3))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3));axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
fig.subplots_adjust(left=.10,right=.99,bottom=.12,top=.96,hspace=.24);save(fig,'daily')
assert int(daily.events.sum())==105277859

# Dense square matrix, 80 actual customers selected by stated activity groups.
with np.load(ROOT/'outputs/behavior_coverage/matrix.npz',allow_pickle=False) as cache:
    counts=cache['counts']; months=cache['months']; ids=cache['client_ids']
totals=counts.sum(axis=1,dtype=np.int64)
assert counts.shape==(100224,25) and totals.sum()==105277859
used=set(); selected=[]
mean,median=float(totals.mean()),float(np.median(totals))
groups=[('Nhiều log nhất',-totals),('Gần trung bình',np.abs(totals-mean)),('Gần trung vị',np.abs(totals-median)),('Ít log, >0',np.where(totals>0,totals,np.inf)),('Không có log',np.where(totals==0,0,np.inf))]
for label,score in groups:
    order=np.lexsort((ids,score)); group=[int(i) for i in order if int(i) not in used][:16]
    selected.extend(group);used.update(group)
matrix=counts[selected].T
fig,ax=plt.subplots(figsize=(17.92,6.6))
from matplotlib.colors import ListedColormap, BoundaryNorm
cmap=ListedColormap(['#f0f3f0','#e1eee4','#bedac4','#91bda0','#55997e','#267969','#104f43'])
bounds=[0,1,11,51,201,501,2001,np.inf]
im=ax.imshow(matrix,cmap=cmap,norm=BoundaryNorm(bounds,cmap.N),aspect='equal',interpolation='nearest')
ax.set_yticks(np.arange(25),[m[5:]+'/'+m[:4][2:]+(' *' if m<'2022-01' else '') for m in months],fontsize=9)
ax.set_xticks([7.5,23.5,39.5,55.5,71.5],[x[0] for x in groups],fontsize=12)
ax.tick_params(axis='both',length=0,pad=10)
for x in [15.5,31.5,47.5,63.5]:ax.axvline(x,color='white',lw=3)
for spine in ax.spines.values():spine.set_visible(False)
fig.subplots_adjust(left=.075,right=.99,bottom=.13,top=.99)
save(fig,'matrix')
pd.DataFrame(matrix,index=months,columns=[f'K{i+1:02}' for i in range(80)]).to_csv(OUT/'matrix_80.csv')
pd.DataFrame([{'alias':f'K{k+1:02}','group':groups[k//16][0],'total_events':int(totals[i])} for k,i in enumerate(selected)]).to_csv(OUT/'matrix_selection_summary.csv',index=False)

CSS='''*{box-sizing:border-box}html,body{margin:0;background:#fff;color:#173d38;font-family:Arial,sans-serif}section{width:1920px;height:1080px;position:relative;padding:74px 100px;background:#fff;overflow:hidden}.brand{font-size:18px;letter-spacing:3px;color:#637970;margin-bottom:40px}h1{font-size:58px;line-height:1.13;font-weight:600;letter-spacing:-1.4px;margin:0 0 20px}.sub{font-size:26px;color:#637970;margin:0;line-height:1.45}.body{margin-top:35px}.chart{width:1696px;height:600px;object-fit:contain;display:block}.callout{font-size:25px;line-height:1.5;border-left:5px solid #80b89c;padding:8px 0 8px 24px;margin-top:24px}.caption{font-size:21px;line-height:1.45;color:#637970;margin:14px 0 0}footer{position:absolute;left:100px;right:100px;bottom:40px;border-top:1px solid #d5dfd9;padding-top:17px;font-size:17px;color:#637970}.num{float:right}table{border-collapse:collapse;width:100%;font-size:27px}td,th{text-align:left;padding:20px 15px;border-bottom:1px solid #d5dfd9;vertical-align:top;line-height:1.35}th{font-size:22px;color:#637970;font-weight:500}td b{font-weight:600}.metricrow{display:flex;gap:100px;margin:26px 0 36px}.metric{flex:1}.metric strong{display:block;font-size:75px;font-weight:500;color:#267969}.metric span{font-size:26px}.two{display:flex;gap:70px}.two>div{flex:1}.two h2{font-size:34px;font-weight:500}.two p{font-size:27px;line-height:1.6}.barrow{display:flex;align-items:center;gap:20px;margin:35px 0;font-size:26px}.barrow .label{width:165px}.track{background:#edf1ec;width:820px;height:32px}.bar{height:32px;background:#267969}.barrow .value{width:480px;text-align:right}.smalltable td{font-size:24px;padding:16px}.matrix{width:1792px;height:660px;object-fit:contain;margin-left:-36px}.legend{display:flex;gap:24px;font-size:20px;align-items:center;margin-top:2px}.legend i{display:inline-block;width:20px;height:20px;margin-right:6px;vertical-align:middle}.compact .brand{margin-bottom:28px}.compact h1{font-size:54px}.compact .body{margin-top:16px}.compact .chart{height:650px}'''
HEAD='<!doctype html><html lang="vi"><head><meta charset="utf-8"><style>'+CSS+'</style></head><body>'
pages=[]
NOTES = {
'schema': 'Nguồn: descriptive.structure/table_inventory.csv và column_roles.csv. Có 31 cột vật lý lưu trong file Parquet. Thêm fold từ partition cho mỗi bảng tạo 36 cột logic; 26 tên cột phân biệt. Đây là schema, không phải 36 đặc trưng mô hình. Số dòng nguồn chưa qua thao tác loại trùng.',
'duplicates': 'Nguồn: descriptive.quality/duplicates.csv, quét toàn bộ xuyên shard và batch. client_split có 100.224 client_id duy nhất. targets có 1.202.688 khóa client_id × mon duy nhất. Không phát hiện dòng trùng hoàn toàn trong clients, targets và ba nguồn detail. Có một cặp giao dịch cùng client_id và event_time nhưng khác thuộc tính; giữ cả hai vì chưa đủ cơ sở coi là giao dịch trùng. Không deduplicate dữ liệu nguồn.',
'null': 'Nguồn: review/missing_values.csv và null_target_context.csv. Bốn cột target_1…4 không có ô null. 120.786 / 1.202.688 dòng có diff_trans_date null (10,043%), trùng hoàn toàn với nhóm trans_count = 0. Chưa xác minh công thức hoặc đơn vị của diff_trans_date; giữ nguyên null. Trong trx, 342.181 dòng có ít nhất một thuộc tính null (0,883%); amount không có null, NaN hoặc vô cực. Không có dòng sự kiện khác với một dòng có ô null.',
'raw_stats': 'Nguồn: descriptive.transactions/transaction_categories.csv, descriptive.modalities/location_categories.csv, review/embedding_element_quality.csv, descriptive.targets/target_prevalence.csv. Đếm giá trị phân biệt của mã phân loại không bao gồm null. Thống kê trực tiếp trên các cột gốc và đơn vị dòng tương ứng; chưa tạo feature cho mô hình. Một vector 768 chiều là một cột embedding. Không trình bày phân phối Amount hoặc diễn giải tiền gộp khi chưa xác minh đơn vị.',
'matrix': 'Nguồn: outputs/behavior_coverage/matrix.npz, tổng 105.277.859 log detail. Trục Y là 25 tháng từ 12/2020 đến 12/2022, X là 80 khách. Mỗi nhóm chọn 16 khách thực, không lặp: nhiều log nhất; gần trung bình 1.050,43 nhất; gần trung vị 646 nhất; ít log nhất nhưng có log; không có log. Tie theo ID sắp xếp. Các nhóm có chủ đích, không đại diện quần thể. Ô vuông, cùng thang màu phân đoạn theo số log gốc; không log-transform. Tháng trước 01/2022 có nguồn geo/dialog chưa đủ quan sát: tổng chỉ phản ánh log hiện có, không xác nhận không hoạt động. Không công bố ID thật.',
'full_scope': 'Nguồn: table_inventory.csv, events_per_customer.csv và source_time_ranges.csv. Toàn lịch sử có 38.764.930 giao dịch, 66.295.724 log vị trí, 217.205 log hội thoại, tổng 105.277.859. Tỷ trọng mỗi nguồn dùng cùng mẫu số tổng log, không phải tỷ lệ giữ lại năm 2022. Có 99.756 khách có ít nhất một log, 468 khách không có bất kỳ log nào trong toàn lịch sử. Không cộng thêm ptls vì là dạng biểu diễn liên quan của dữ liệu sự kiện.',
'timeline': 'Nguồn: descriptive.temporal/source_time_ranges.csv và descriptive.targets/label_coverage.csv. Vị trí đoạn theo datetime thực, hai đầu phẳng có vạch đứng. Đầu và cuối được ghi theo ngày, không làm tròn vị trí theo tháng. Nhãn mon từ 28/02/2022 đến 31/01/2023. Khoảng min–max không chứng minh log liên tục ở từng khách; không cắt raw theo phần giao nhau giữa các nguồn.',
'quantiles': 'Nguồn: review/events_per_customer.csv, population = all_customers. Thống kê đếm số log theo khách là yêu cầu mô tả riêng, không phải các cột gốc đã có hoặc feature đã tạo cho mô hình. Dân số 100.224 gồm khách 0 log. Dải P25–P75, vạch trung vị P50 và chấm P95; không gọi là boxplot Tukey. Mỗi nguồn dùng trục tuyến tính riêng. Hiển thị đến P95 để đọc rõ; max được ghi riêng, không xóa dữ liệu đuôi.',
'monthly': 'Nguồn: review/monthly_coverage.csv. Toàn bộ lịch sử gồm 25 tháng lịch từ 12/2020 đến 12/2022. Tháng 12/2020 chỉ có sự kiện ngày 31/12, không phải một tháng quan sát đầy đủ. Mỗi khách được đếm một lần trong từng nguồn và từng tháng nếu có ít nhất một log. Ba biểu đồ xếp dọc, dùng cùng trục X và thang Y tuyến tính riêng như trang số log theo ngày. Ô geo/dialog không có log trước 01/2022 để khoảng trống, không diễn giải là không hoạt động. Không cộng số khách qua các tháng để tính khách toàn kỳ. Không cộng hoặc loại bỏ dữ liệu raw.',
'daily': 'Nguồn: review/daily_behavior.csv. Đếm dòng sự kiện theo ngày của timestamp gốc, không tự đổi múi giờ. Giữ toàn bộ lịch sử từ 31/12/2020 đến 31/12/2022. Tổng ba nguồn 105.277.859. Ba trục Y tuyến tính riêng, không log và không làm mượt. NaN do chưa quan sát nguồn được để khoảng trống; không diễn giải thành 0 hoạt động.',
'targets': 'Nguồn: review/target_monthly.csv. Mỗi target có 12 tháng nhãn từ 02/2022 đến 01/2023, mỗi tháng 100.224 dòng. Tỷ lệ dương = tổng positive_n / tổng labelled_n qua 5 fold × 100, không trung bình tỷ lệ fold không trọng số. Giữ bốn biểu đồ trên cùng trang, trục Y tuyến tính riêng và có thể không bắt đầu từ 0, theo xác nhận người dùng. Không so chiều cao giữa các biểu đồ. Tháng nhãn khác khoảng lịch sử sự kiện.'
}
def add(key,title,subtitle,body,source,notes,number,cls=''):
    notes=NOTES[key]
    section=f'<section class="{cls}" data-document-role="page" data-label="{html.escape(title)}" data-speaker-notes="{html.escape(notes,quote=True)}"><div class="brand">MBD / PROPENSITY LAB</div><h1>{title}</h1><p class="sub">{subtitle}</p><div class="body">{body}</div><footer>{source}<span class="num">{number:02} / 22</span></footer></section>'
    pages.append({'key':key,'title':title,'final_page':number,'html':section})
def table(headers,rows,cls=''):
    return '<table class="'+cls+'"><thead><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{v}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'

inventory=read('descriptive.structure/table_inventory')
names={'clients':'Khách hàng','targets':'Nhãn','transactions':'Giao dịch','geo':'Vị trí','dialog':'Hội thoại'}
body=table(['Bảng','Số dòng gốc','Cột trong file','+ fold → cột logic'],[[names[r.source],vi(r.rows),int(r.logical_columns)-1,f'+ 1 → {r.logical_columns}'] for r in inventory.itertuples()])
body+='<div class="two"><div><h2>31 cột vật lý</h2><p>Cột được lưu bên trong file Parquet.</p></div><div><h2>36 cột logic · 26 tên khác nhau</h2><p>Thêm <b>fold</b> từ thư mục partition vào mỗi bảng; có tên cột lặp giữa các bảng.</p></div></div><p class="caption">Đếm cột schema, chưa phải số feature mô hình. “Unique” của dòng và khóa được trình bày ở trang sau.</p>'
add('schema','Cấu trúc cột và đơn vị đếm','5 bảng tham chiếu và detail · toàn bộ lịch sử',body,'Nguồn: table_inventory.csv · column_roles.csv','Cột vật lý là cột trong shard Parquet. fold được lấy từ partition, cộng một cột cho mỗi bảng. Tổng31 vật lý,36 logic,26 tên phân biệt; không phải36feature. Số dòng là số dòng nguồn, kiểm tra trùng không tự xóa.',3)

body=table(['Phép kiểm tra','Kết quả toàn dữ liệu','Cách hiểu'],[
['client_id trong client_split','100.224 khóa duy nhất','0 khóa lặp'],
['client_id × mon trong targets','1.202.688 khóa duy nhất','0 khóa lặp'],
['Trùng toàn bộ cột trong detail','<b>0 dòng trùng hoàn toàn</b>','trx · geo · dialog'],
['Cùng client_id + event_time','<b>1 cặp ở trx</b>','Khác thuộc tính → chưa kết luận trùng giao dịch']])
body+='<div class="callout">Đã kiểm tra unique và duplicate trên toàn dữ liệu; giữ nguyên các dòng gốc, không tự deduplicate.</div><p class="caption">Một khách có nhiều sự kiện là cấu trúc bình thường của bảng detail.</p>'
add('duplicates','Unique của khách, khóa nhãn và sự kiện','Ba phép đếm khác nhau · không dùng unique(amount) để đếm giao dịch',body,'Nguồn: descriptive.quality/duplicates.csv · full scan','Kiểm tra xuyên shard/batch. Không có duplicate full-row trong clients,targets,trx,geo,dialog. Một nhóm trx hai dòng cùng client_id+event_time khác thuộc tính; giữ cả hai. duplicate_n=sum(group_size-1).',4)

frame_n=100224*25
null_rows=[]
for source,label in [('trx','Giao dịch'),('geo','Vị trí'),('dialog','Hội thoại')]:
    d=monthly[monthly.source.eq(source)]
    observed=int(d.customers_with_log.sum())
    missing=frame_n-observed
    null_rows.append([label,vi(observed),vi(missing),vi(missing/frame_n*100,2)+'%'])
body=table(['Nguồn','Ô có ≥1 log','Ô trống (null khung)','Tỷ lệ null / toàn khung'],null_rows)
body+='<div class="callout">Null khung = không có bản ghi tại nguồn tương ứng trong ô khách–tháng.</div><p class="caption">Mẫu số cố định: 100.224 khách × 25 tháng = 2.505.600 ô/nguồn, gồm các khoảng trước 2022.<br>Đây là độ thiếu log trên khung độ phủ; 12/2020 chỉ có ngày 31.</p>'
NOTES['null']='Theo xác nhận của người dùng, null được đo trên toàn khung 100.224 khách × 25 tháng lịch từ 12/2020 đến 12/2022, tổng 2.505.600 ô mỗi nguồn. Một ô nguồn–khách–tháng không có bản ghi được biểu diễn null cho bảng độ phủ này, gồm mọi khoảng trước 2022; không áp dụng chính sách chỉ null sớm geo/dialog cho phép đo này. Trx có 1.796.643 ô có log và 708.957 ô trống (28,29%); geo có 745.856 và 1.759.744 (70,23%); dialog có 133.780 và 2.371.820 (94,66%). Không có log không chứng minh không phát sinh hành vi ngoài dữ liệu. Đây không phải tỷ lệ ô thuộc tính null của bảng sự kiện. Chỉ thay cách biểu diễn/báo cáo độ phủ; không sửa dữ liệu raw hoặc cache. Nguồn: review/monthly_coverage.csv.'
add('null','Null trên toàn khung khách hàng–tháng','100.224 khách × 25 tháng · 12/2020–12/2022 · gồm cả khoảng không có log',body,'Nguồn: review/monthly_coverage.csv · mẫu số 2.505.600 ô/nguồn','',5)

cat=read('descriptive.transactions/transaction_categories').groupby('column').distinct_n.first()
geo=read('descriptive.modalities/location_categories').groupby('column').distinct_n.first()
body=table(['Cột gốc','Quan sát trên','Thống kê đã kiểm tra'],[
['amount','38.764.930 giao dịch','0 null · 0 NaN / ±Inf'],
['event_type / event_subtype','38.764.930 giao dịch',f'{cat.event_type} / {cat.event_subtype} mã khác nhau'],
['geohash_4 / _5 / _6','66.295.724 log vị trí',f'{vi(geo.geohash_4)} / {vi(geo.geohash_5)} / {vi(geo.geohash_6)} mã'],
['embedding','217.205 log hội thoại','768 chiều / vector · 0 phần tử thiếu hoặc không hữu hạn'],
['target_1 / _2 / _3 / _4','1.202.688 dòng khách–tháng','0 ô nhãn thiếu · tỷ lệ dương 0,446% / 0,048% / 0,377% / 0,254%']],'smalltable')
body+='<div class="callout">Thống kê trực tiếp trên cột gốc; mã phân loại không có ý nghĩa lớn–nhỏ theo giá trị số.</div>'
add('raw_stats','Thống kê mô tả trên các biến gốc','Giữ đúng đơn vị quan sát của từng bảng',body,'Nguồn: transaction_categories.csv · location_categories.csv · embedding_element_quality.csv · target_prevalence.csv','Tập giá trị phân loại distinct bỏ null; tần suất dùng tất cả dòng nguồn. Không tạo feature khách hàng mới. Amount không trình bày phân phối hoặc thống kê tiền gộp khi chưa xác minh đơn vị. Embedding kiểm tra từngphầntử từfullscan,768 chiều tính là1cộtvector.',7)

legend=''.join(f'<span><i style="background:{c}"></i>{label}</span>' for c,label in zip(cmap.colors,['0','1–10','11–50','51–200','201–500','501–2.000','>2.000']))
body=img('matrix','matrix')+f'<div class="legend">Log / ô: {legend}</div><p class="caption">* Trước 01/2022: phần không có log geo/dialog chưa được coi là 0 hoạt động; tổng log quan sát có thể chưa đủ nguồn.</p>'
add('matrix','Độ phủ hành vi theo tháng và khách hàng','25 tháng × 80 khách · 5 nhóm có chủ đích, mỗi nhóm 16 người · ô vuông',body,'Nguồn: matrix.npz · tổng detail/trx + geo + dialog · nhóm minh họa, không đại diện quần thể','TrụcY25tháng12/2020–12/2022,trụcX80khách. Tổngsốlogquan sátcả3nguồn.5nhóm:16tổnglogcaonhất,16gầnmean1050.4256,16gầnmedian646,16ítnhất>0,16khônglog; disjoint, tie theoID; khôngngẫunhiên,khôngđạidiện. Chungthangmàuphânđoạntheogiátrịgốc,khônglog. O vuông aspectequal;dải0,1–10,11–50,51–200,201–500,501–2000,>2000. Giữmọi rawlogtrước2022. KhônglộIDthật. Thángtrước01/2022 thiếugeo/dialog không đồngnghĩa0hànhvi; tổnglàlogquan sát.',8,'compact')

total=105277859
body='<div class="metricrow"><div class="metric"><strong>105.277.859</strong><span>log hành vi toàn lịch sử</span></div><div class="metric"><strong>99.756</strong><span>khách có ≥1 log / 100.224 khách</span></div></div>'
for name,n,c in [('Giao dịch',38764930,GREEN),('Vị trí',66295724,MINT),('Hội thoại',217205,CORAL)]:
    body+=f'<div class="barrow"><span class="label">{name}</span><div class="track"><div class="bar" style="width:{n/total*100}%;background:{c}"></div></div><span class="value">{vi(n)} · {vi(n/total*100,2)}%</span></div>'
body+='<div class="callout">468 khách không có dòng sự kiện ở cả ba nguồn trong toàn lịch sử.</div><p class="caption">Tỷ trọng = số log nguồn / 105.277.859; không cộng thêm ptls.</p>'
add('full_scope','Tỷ trọng log trong toàn bộ lịch sử','31/12/2020–31/12/2022 · mốc đầu và cuối khác nhau giữa các nguồn',body,'Nguồn: table_inventory.csv · events_per_customer.csv · source_time_ranges.csv','Fullhistorytrx38764930geo66295724dialog217205,total105277859. Có99756khách>=1log/100224,468khôngcóbấtkỳlog. Tỷtrọngchungmẫusốtổng105277859,khôngphảitỷlệgiữlạinăm2022. ptls khôngcộngchồng. Các nguồnkhôngquan sátđồngthờihoặc liêntục.',10)

add('timeline','Các nguồn bắt đầu ở các thời điểm khác nhau','Độ dài đoạn theo thời gian thực · vạch thẳng đánh dấu hai đầu',img('timeline')+'<p class="caption">Đoạn thể hiện khoảng min–max timestamp quan sát; không khẳng định có log liên tục cho mọi khách.</p>','Nguồn: source_time_ranges.csv · label_coverage.csv','Đầuđuôi theo datetimechínhxác trongsource_time_ranges;nhãn mon28/02/2022–31/01/2023. Ngàyđầucuối hiểnthịdd/mm/yyyy, không làmtrònvịtrítheotháng. solidcapbutt,endpointticks. Không cắtbỏlịchsửđểlấygiao3nguồn.',13)

statsrow=' · '.join(f"{name}: TB {vi(eventstats[eventstats.source.eq(s)].iloc[0]['mean'],2)}, max {vi(eventstats[eventstats.source.eq(s)].iloc[0]['max'])}" for s,name in [('trx','trx'),('geo','geo'),('dialog','dialog')])
add('quantiles','Phân vị số log trên mỗi khách','Toàn lịch sử · 100.224 khách, gồm cả khách có 0 log',img('quantiles')+'<p class="caption">Dải màu P25–P75 · vạch P50 (trung vị) · chấm P95 · trục tuyến tính riêng từng nguồn.</p>'+f'<p class="caption">{statsrow}</p>','Nguồn: review/events_per_customer.csv','Đây là thốngkêđếmđãđượcyêucầu,khôngphảicộtgốchoặcfeatuređãtạochomôhình. Phânvịtrêntoànbộ100224kháchkểcả0; hiểnthịđếnP95, cựcđạighiriêng;khôngxóaphầnđuôi. CácdảilàIQR,khônggọilàTukeyboxplot. Trụclinearriêng,khônglog.',14)

add('monthly','Khách có log theo tháng','Toàn lịch sử 12/2020–12/2022 · khách duy nhất / nguồn / tháng · thang Y riêng',img('monthly_zoom')+'<p class="caption">Mỗi khách tính một lần trong từng nguồn–tháng · 12/2020 chỉ có ngày 31.<br>Khoảng trống trước 01/2022: chưa đủ cơ sở diễn giải là không hoạt động.</p>','Nguồn: review/monthly_coverage.csv · 25 tháng lịch · quần thể 100.224 khách','',15,'compact')

add('daily','Số log được ghi nhận theo ngày','Toàn lịch sử · 3 nguồn, trục Y riêng · số dòng sự kiện / ngày',img('daily')+'<p class="caption">Mỗi điểm là số log quan sát trong một ngày; phần chưa có nguồn không được diễn giải là không hoạt động.</p>','Nguồn: review/daily_behavior.csv · tổng 105.277.859 log','Gộptheongàytimestampgốcnaive,khôngtựđổimúigiờ. Hàngngàychưađượcminhchứngquan sátchỉlàkhôngcólog. Giữmọingàytừfirstđếnlastnguồn,rawcountslinear,khôngrollingkhônglog. Tổng105277859;3Yriêngđểđọclogdialogthưa.',16,'compact')

add('targets','Biến động tỷ lệ nhãn dương theo tháng','4 biểu đồ trên cùng trang · mỗi target có thang Y riêng',img('targets_zoom')+'<p class="caption">Y: % khách–tháng có nhãn dương · mỗi tháng 100.224 dòng · trục Y đã zoom và có thể không bắt đầu từ 0.</p>','Nguồn: review/target_monthly.csv · gộp 5 fold · tháng nhãn 02/2022–01/2023','Mỗitarget12tháng,100224mẫumỗitháng. Tỷlệ=sumpositive/sumlabelledqua5fold×100. 4trụclinearriêng,khôngsochiềucaoquapanel. Userxácnhậngiữcùngtrang,thangYriêng. Phạmvinhãnkháckhoảngeventhistory.',18,'compact')

for i,p in enumerate(pages,1):
    (OUT/f'page_{i:02}_{p["key"]}.html').write_text(HEAD+p['html']+'</body></html>',encoding='utf-8')
(OUT/'revised_pages.html').write_text(HEAD+''.join(p['html'] for p in pages)+'</body></html>',encoding='utf-8')
manifest={'original_design_id':'DAHaGBuj6pg','backup_design_id':'DAHaM69w9Lk','original_pages':20,'final_pages':22,'removed_original_pages':[14],
          'replacements':{3:'schema',4:'null',6:'matrix',8:'full_scope',11:'timeline',12:'quantiles',13:'monthly',16:'targets'},
          'insertions':[{'after_original':3,'key':'duplicates'},{'after_original':5,'key':'raw_stats'},{'after_original':13,'key':'daily'}],
          'pages':[{k:v for k,v in p.items() if k!='html'} for p in pages], 'target_ranges':target_ranges,
          'amount_decision':'Remove distribution chart following the latest user decision in the referenced conversation; retain amount null validation on raw stats page.'}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
