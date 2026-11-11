"""Build the Vietnamese EDA narrative. Aggregate outputs only; no holdout predictions."""
from pathlib import Path
import csv
import html
import json
import math
import shutil
import duckdb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'deliverables/propensity-data-story'
OUT.mkdir(exist_ok=True, parents=True)
ASSETS = OUT / 'assets'
ASSETS.mkdir(exist_ok=True)
SKILL = Path.home() / '.codex/skills/html-ppt'
for name in ['base.css', 'runtime.js']:
    dest = ASSETS / name
    if not dest.exists():
        shutil.copyfile(SKILL / 'assets' / name, dest)
    dest.write_text('\n'.join(line.rstrip() for line in dest.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')

def rows(p):
    with (ROOT / p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def fmt(v, n=0):
    return f'{float(v):,.{n}f}'.replace(',', '_').replace('.', ',').replace('_', '.')

def esc(v):
    return html.escape(str(v))

def tx(x, y, value, size=20, fill='--text-1', anchor='start', weight=400):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="var({fill})" text-anchor="{anchor}" font-weight="{weight}">{esc(value)}</text>'

def rect(x, y, w, h, color='--accent', radius=6):
    return f'<rect x="{x}" y="{y}" width="{max(0,w)}" height="{max(0,h)}" rx="{radius}" fill="var({color})"/>'

def line(x1,y1,x2,y2,color='--border-strong',width=1):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="var({color})" stroke-width="{width}"/>'

def svg(body, label, height=350):
    return f'<svg viewBox="0 0 1160 {height}" role="img" aria-label="{esc(label)}"><title>{esc(label)}</title>{body}</svg>'

def table(headers, values, widths=None):
    cols=''
    if widths:
        cols='<colgroup>'+''.join(f'<col style="width:{w}%">' for w in widths)+'</colgroup>'
    return '<div class="table-wrap"><table>'+cols+'<thead><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{v}</td>' for v in r)+'</tr>' for r in values)+'</tbody></table></div>'

def bars(labels, values, unit='%', maximum=None, colors=None, note=None):
    maximum = maximum or max(values)*1.22
    b=''
    step=min(58,270/len(values)); start=28
    for i,(label,value) in enumerate(zip(labels,values)):
        y=start+i*step
        precision=2 if unit=='%' else 4 if unit=='' else 0
        b+=tx(0,y+23,label,21)+rect(385,y,620,28,'--surface-2')+rect(385,y,620*value/maximum,28,colors[i] if colors else '--accent')+tx(1030,y+23,fmt(value,precision)+('%' if unit=='%' else ''),22,weight=650)
    b+=line(385,310,1005,310)+tx(385,338,'0',16,'--text-3')+tx(1005,338,fmt(maximum,2 if unit=='' else 0 if unit!='%' else 1)+unit,16,'--text-3','end')
    if note:b+=tx(0,335,note,17,'--text-2')
    return svg(b,'So sánh '+', '.join(labels))

def trend(labels, series, maximum, ticks, unit='', annotation=''):
    b=''; left=65; right=1120; top=38; bottom=285
    for tick in ticks:
        y=bottom-(bottom-top)*tick/maximum
        b+=line(left,y,right,y,'--border')+tx(left-12,y+6,fmt(tick,1 if maximum<5 else 0)+unit,16,'--text-3','end')
    for name,vals,color in series:
        coords=[(left+i*(right-left)/max(1,len(labels)-1),bottom-(bottom-top)*v/maximum) for i,v in enumerate(vals)]
        b+=f'<polyline points="'+ ' '.join(f'{x},{y}' for x,y in coords)+f'" fill="none" stroke="var({color})" stroke-width="3.5"/>'
        b+=''.join(f'<circle cx="{x}" cy="{y}" r="4" fill="var({color})"/>' for x,y in coords)
    stride=max(1,math.ceil(len(labels)/12))
    for i,label in enumerate(labels):
        if i%stride==0 or i==len(labels)-1:b+=tx(left+i*(right-left)/max(1,len(labels)-1),310,label,15,'--text-2','middle')
    for i,(name,vals,color) in enumerate(series):
        b+=rect(65+i*360,332,20,5,color,1)+tx(96+i*360,341,name,17,'--text-2')
    if annotation:b+=tx(750,20,annotation,18,'--accent',weight=650)
    return svg(b,'Diễn biến theo tháng: '+', '.join(s[0] for s in series))

# Read only the seven training months for new feature/label exploration.
c=duckdb.connect()
c.execute("SET threads=2")
paths=[str(ROOT/f'outputs/propensity_data_v1/features/month=2022-{m:02d}/part.parquet').replace('\\','/') for m in range(2,9)]
c.read_parquet(paths,hive_partitioning=True).create_view('train')
assert c.execute('select count(*), sum(y_any) from train').fetchone()==(701568,8672)
base=c.execute('select avg(y_any) from train').fetchone()[0]

def grouped(expression):
    return c.execute(f'SELECT {expression} grp,count(*) n,sum(y_any) positives,avg(y_any)*100 rate FROM train GROUP BY 1 ORDER BY 1').fetchall()

aggregate={
    'scope':'train 02–08/2022, 701568 customer-months, 8672 positives; 30-day rolling features',
    'baseline_rate_pct':base*100,
    'rfm':{},
}
for key,col,cuts in [
    ('recency','trx_30d_days_since_last_event',[3,10]),
    ('frequency','trx_30d_event_count',[0,5,20]),
    ('monetary','trx_30d_main_amount_abs_sum',[0,10000,100000]),
    ('dialog','dialog_30d_event_count',[0,1,2]),
    ('geo','geo_30d_geohash_6_nunique',[0,5,20]),
]:
    expression='CASE WHEN '+col+" IS NULL THEN 'Z. Không xác định' "
    def bound(v):
        return f'{v//1000}k' if v>=1000 else str(v)
    for i,cut in enumerate(cuts):
        label=('≤'+bound(cut)) if i==0 else f'({bound(cuts[i-1])}; {bound(cut)}]'
        expression+=f"WHEN {col}<={cut} THEN '{i}. {label}' "
    expression+=f"ELSE '{len(cuts)}. >{bound(cuts[-1])}' END"
    aggregate['rfm'][key]=[{'group':r[0],'n':r[1],'positives':r[2],'rate_pct':r[3],'lift':r[3]/(base*100)} for r in grouped(expression)]

# Recompute exact consecutive-month persistence, rather than ever-bought rates.
target_path=str(ROOT/'data/MBD-mini/targets/**/*.parquet').replace('\\','/')
c.execute(f"CREATE VIEW labels AS SELECT client_id,date_trunc('month',cast(mon AS TIMESTAMP)) m, CASE WHEN target_1=1 OR target_2=1 OR target_3=1 OR target_4=1 THEN 1 WHEN target_1=0 AND target_2=0 AND target_3=0 AND target_4=0 THEN 0 ELSE NULL END y FROM read_parquet('{target_path}')")
assert c.execute('SELECT count(*) FROM labels WHERE y IS NULL').fetchone()[0]==0
persist=c.execute("SELECT p.y,count(*) n,sum(t.y) positives,avg(t.y)*100 rate FROM labels t JOIN labels p ON t.client_id=p.client_id AND t.m=p.m+INTERVAL '1 month' WHERE t.m>='2022-09-01' AND t.m<'2022-12-01' GROUP BY p.y ORDER BY p.y").fetchall()
aggregate['persistence']=[dict(previous=int(r[0]),n=r[1],positives=r[2],rate_pct=r[3]) for r in persist]
hist=c.execute("SELECT k,count(*) n FROM (SELECT client_id,sum(y) k FROM labels GROUP BY client_id) GROUP BY k ORDER BY k").fetchall()
aggregate['positive_month_histogram']=[{'months':int(k),'customers':n} for k,n in hist]
assert sum(n for k,n in hist)==100224
assert sum(k*n for k,n in hist)==12854
c.close()

monthly=rows('docs/mbd_mini_eda/tables/descriptive.temporal/monthly_activity.csv')
target=rows('docs/propensity_feasibility/any_product/monthly.csv')
assoc={r['feature']:r for r in rows('deliverables/eda/tables/feature_target_associations.csv')}
metrics={r['model']:r for r in rows('deliverables/baseline-final/summary.csv')}
overlap=rows('docs/baseline_diagnostics_20261110/overlap_top1.csv')
cat=sorted([r for r in rows('docs/mbd_mini_eda/tables/descriptive.transactions/transaction_categories_full.csv') if r['column']=='event_type'],key=lambda r:int(r['n']),reverse=True)

slides=[]
def add(section,title,chart,meaning,method,source,notes='',appendix=False):
    slides.append(dict(section=section,title=title,chart=chart,meaning=meaning,method=method,source=source,notes=notes,appendix=appendix))

def flow(nodes, labels=None):
    b=''; w=330; gap=58
    for i,(head,body) in enumerate(nodes):
        x=i*(w+gap)
        b+=rect(x,55,w,185,'--surface-2')+rect(x,55,7,185,'--accent')+tx(x+24,100,head,25,weight=700)
        for j,v in enumerate(body):b+=tx(x+24,143+j*31,v,20,'--text-2')
        if i<len(nodes)-1:b+=tx(x+w+29,157,'→',33,'--accent','middle')
    return svg(b,' → '.join(n[0] for n in nodes))

add('A · BÀI TOÁN','Dùng hành vi trước T để xếp hạng người mua tháng T',
    flow([('Hành vi đến T−1',['Trx · Geo · Dialog','recent + lịch sử cũ']),('Cutoff: đầu tháng T',['1 khách × 1 tháng','chỉ event_time < cutoff']),('Nhãn trong tháng T',['y_any = 1: ≥1 target = 1','y_any = 0: cả 4 target = 0'])]),
    'Ý nghĩa cho mô hình: tạo xác suất mua để chọn 1.003 khách/tháng ở ngân sách 1%.',
    'Sơ đồ căn chỉnh feature và nhãn; ví dụ cutoff 01/12/2022 → hành vi trước 01/12 → nhãn tháng 12.',
    'deliverables/baseline-final/REPORT.md',
    'Nếu chưa đủ căn cứ xác định 0/1 thì giữ unknown. Snapshot hiện tại không có unknown. Lịch sử target chỉ hợp lệ khi đã khả dụng tại cutoff. Đây là dự đoán mua bất kỳ, chưa chọn sản phẩm và chưa đo uplift chiến dịch.')

add('A · THÁCH THỨC & CÂU HỎI','Chỉ 1,07% mẫu dương: cần đo tín hiệu ở đúng thời điểm',
    table(['Thách thức đã xác minh','Câu hỏi dữ liệu dẫn dắt deck'],[
    ['100.224 khách · 1,20 triệu khách–tháng','01 · Dữ liệu có tín hiệu dự đoán ngoài rule không?'],
    ['105,28 triệu sự kiện · 98,93% nhãn âm','02 · Tín hiệu nằm ở trx, dialog hay geo?'],
    ['Nguồn bắt đầu khác nhau · lịch sử nhãn 12 tháng','03 · Chọn cutoff và cửa sổ tích lũy thế nào?'],
    ['SLA nhãn chưa rõ · nhiều khách thiếu nguồn','04 · Rủi ro leakage, drift và loại khách là gì?']],[45,55]),
    'Ý nghĩa cho mô hình: giữ phân bố thật; đánh giá AP và Recall@K qua thời gian.',
    'Tỷ lệ dương = 12.854 / 1.202.688; số sự kiện chỉ cộng ba bảng detail.',
    'docs/mbd_inventory/assessment.md; docs/propensity_feasibility/any_product/summary.json',
    'Không có bằng chứng cho target trễ công bố 6 tháng. Gantt ở slide 4 phân biệt khoảng lịch sử không có target với SLA công bố nhãn.')

add('B · TOÀN CẢNH','Trx và geo chiếm gần toàn bộ 105,28 triệu sự kiện',
    table(['Nguồn','Số dòng','ID có dữ liệu','Cột¹','Vai trò'],[
    ['Trx','38.764.930','98.721','14 + 1','Đầu vào · giao dịch'],
    ['Geo','66.295.724','72.573','5 + 1','Đầu vào · mã vị trí'],
    ['Dialog','217.205','46.320','3 + 1','Đầu vào · metadata, vector 768D'],
    ['Targets','1.202.688','100.224','8 + 1','Nhãn · khách × tháng']],[15,20,19,12,34]),
    'Ý nghĩa cho mô hình: aggregate từng nguồn theo khách–cutoff rồi mới ghép, tránh nhân dòng.',
    '¹ Cột vật lý + fold trong partition; embedding tính là một cột. ID tính trên toàn lịch sử mỗi nguồn.',
    'docs/mbd_inventory/assessment.md; docs/mbd_temporal_coverage/assessment.md',
    'Không cộng ptls với detail: hai dạng biểu diễn cùng lịch sử. PTLS/trx thiếu 2.020.020 sự kiện ở 13 khách so với detail; dùng detail làm nguồn chuẩn.')

b=''; x0=170; scale=960/26
for m,label in [(0,'01/2021'),(6,'07/2021'),(12,'01/2022'),(18,'07/2022'),(24,'01/2023')]:
    b+=line(x0+m*scale,40,x0+m*scale,292,'--border')+tx(x0+m*scale,27,label,17,'--text-2','middle')
b+=rect(x0,49,13*scale,245,'--warn-soft')+tx(x0+15,285,'Có trx, chưa có target tháng: ~13 tháng',19,'--warn',weight=650)
for i,(name,start,end,color,label) in enumerate([('Trx',0,24,'--accent','31/12/2020 → 31/12/2022'),('Dialog',1.33,24,'--accent-2','10/02/2021 → 31/12/2022'),('Geo',11.35,24,'--accent-3','11/12/2021 → 31/12/2022'),('Targets',13,25,'--text-1','02/2022 → 01/2023')]):
    y=60+i*51;b+=tx(0,y+21,name,22,weight=650)+rect(x0+start*scale,y,(end-start)*scale,29,color)+tx(x0+start*scale+10,y+20,label,16,'--accent-ink')
add('B · THỜI GIAN','Trx có khoảng 13 tháng lịch sử trước tháng nhãn đầu tiên',svg(b,'Gantt phạm vi bốn nguồn'),
    'Ý nghĩa cho mô hình: dùng giai đoạn chưa có nhãn làm lịch sử; không biến thành mẫu âm.',
    'Thanh thể hiện min–max timestamp (targets: tháng chứa mon); tô vàng phần trước 02/2022. Tháng biên có thể không trọn.',
    'docs/mbd_mini_eda/tables/descriptive.temporal/source_time_ranges.csv; docs/propensity_feasibility/remaining_questions/assessment.md',
    'Nguồn timestamp chưa có timezone xác minh; protocol nghiên cứu giả định UTC. Không có available_at nên chưa biết nhãn đến chậm thực tế bao lâu.')

comb=rows('docs/mbd_mini_eda/tables/descriptive.customers/modality_coverage.csv')
b=tx(0,25,'35.424 khách có đủ ba nguồn trong toàn lịch sử',24,'--accent',weight=700)
for j,r in enumerate(comb):
    x=200+j*117; n=int(r['customers']);h=170*n/36223
    b+=rect(x,212-h,63,h,'--accent' if r['trx_geo_dialog']=='111' else '--accent-2')+tx(x+31,202-h,fmt(n),17,'--text-1','middle')
    active=[]
    for i,v in enumerate(r['trx_geo_dialog']):
        y=253+i*36
        if v=='1':active.append(y)
        b+=f'<circle cx="{x+31}" cy="{y}" r="7" fill="var({"--accent" if v=="1" else "--border-strong"})"/>'
    if len(active)>1:b+=line(x+31,min(active),x+31,max(active),'--accent',3)
for i,name in enumerate(['Trx','Geo','Dialog']):b+=tx(95,260+i*36,name,20,'--text-2','end')
add('B · ĐỘ PHỦ','Bắt buộc đủ ba nguồn sẽ loại gần 65% khách',svg(b,'UpSet: các tổ hợp nguồn không chồng lấn'),
    'Ý nghĩa cho mô hình: giữ cả 100.224 khách; thêm cờ có log và giữ NaN đúng nghĩa.',
    'UpSet đếm ID có ≥1 sự kiện mỗi nguồn trên toàn kỳ; từng cột là một tổ hợp riêng. Không dùng làm cohort tại cutoff quá khứ.',
    'docs/mbd_mini_eda/tables/descriptive.customers/modality_coverage.csv',
    '35,34% có đủ ba nguồn trong toàn lịch sử; không suy ra đủ nguồn mỗi tháng. Tại cutoff 01/12/2022, số có đủ ba nguồn trong quá khứ là 34.135.')

vals=[float(r['positive_rate'])*100 for r in target]
add('C · TARGET / PHÂN PHỐI & THỜI GIAN','Tỷ lệ mua giảm từ đỉnh 1,69% xuống 0,67%',
    trend([r['month'][5:]+'/'+r['month'][2:4] for r in target],[('y_any dương (%)',vals,'--accent')],2,[0,.5,1,1.5,2],'%', 'Mỗi tháng: 100.224 khách'),
    'Ý nghĩa cho mô hình: dùng validation cuốn chiếu; theo dõi prevalence và calibration từng tháng.',
    'Tỷ lệ = số khách có ≥1 target dương / 100.224 mỗi tháng. Các tháng 12/2022–01/2023 chỉ mô tả nhãn.',
    'docs/propensity_feasibility/any_product/monthly.csv',
    'Đỉnh 04/2022; tháng đầu 02/2022 có 1.041 khách mua. Một năm nhãn chưa đủ chứng minh mùa vụ. Không đọc dự đoán test để chọn phương án; thống kê nhãn cuối kỳ đã có trong EDA công bố.')

p0,p1=aggregate['persistence'];ratio=p1['rate_pct']/p0['rate_pct']
add('C · TARGET / PERSISTENCE',f'Mua tháng trước gắn với xác suất mua cao gấp {fmt(ratio,1)} lần',
    bars(['Không mua ở T−1','Có mua ở T−1'],[p0['rate_pct'],p1['rate_pct']],maximum=7,colors=['--accent-2','--accent']),
    'Ý nghĩa cho mô hình: giữ lịch sử mua; hơn 93% người mua validation không mua ở tháng liền trước.',
    f'P(yT=1 | yT−1): gộp 09–11/2022. Nhóm có mua: {fmt(p1["positives"])} / {fmt(p1["n"])}; nhóm còn lại: {fmt(p0["positives"])} / {fmt(p0["n"])}.',
    'Tính lại từ targets; aggregate_data.json → persistence',
    'So đúng tháng liền trước, không dùng đã từng mua toàn lịch sử. Tỷ số là liên hệ quan sát; rate không phải recall. Rule ở top 1% có thể chọn thêm khách không mua T−1 khi số người mua trước ít hơn K.')

trx=[r for r in monthly if r['source']=='transactions' and '2021-01'<=r['month']<='2022-12']
if not trx:trx=[r for r in monthly if r['source']=='trx' and '2021-01'<=r['month']<='2022-12']
add('C · TRX / ĐỘNG HỌC','Trx có lịch sử dài, nhưng mức hoạt động thay đổi theo tháng',
    trend([r['month'][5:]+'/'+r['month'][2:4] for r in trx],[('Số dòng / 25.000',[int(r['events'])/25000 for r in trx],'--accent'),('ID active / 1.000',[int(r['customers'])/1000 for r in trx],'--accent-2')],100,[0,25,50,75,100]),
    'Ý nghĩa cho mô hình: bổ sung rate và recency để tách cường độ giao dịch khỏi độ dài cửa sổ.',
    'Hai chuỗi cùng trục đã chia đơn vị để so nhịp; không đọc là cùng số lượng. Active = ID có ≥1 trx trong tháng.',
    'docs/mbd_mini_eda/tables/descriptive.temporal/monthly_activity.csv',
    'Chỉ dùng các tháng 01/2021–12/2022; tháng 12/2020 có khoảng ba giờ dữ liệu. Tổng trx gồm cả tháng biên là 38.764.930. Độ phủ log không chứng minh theo dõi liên tục.')

def small_group_bars(keys, labels):
    b=''
    for j,(key,head) in enumerate(zip(keys,labels)):
        rr=aggregate['rfm'][key];x=j*390
        b+=tx(x,25,head,23,'--text-1',weight=700)
        vmax=max(r['rate_pct'] for r in rr)*1.15
        for i,r in enumerate(rr):
            y=55+i*53
            label=r['group'].split('. ',1)[1]
            b+=tx(x,y+18,label,17,'--text-2')+rect(x+160,y,135*r['rate_pct']/vmax,24,'--accent')+tx(x+305,y+18,fmt(r['rate_pct'],2)+'%',18,weight=650)
        b+=tx(x,337,'Nhãn dương / số mẫu từng nhóm',16,'--text-3')
    return svg(b,'Tỷ lệ mua theo các nhóm '+', '.join(labels))

add('C · TRX / LIÊN HỆ NHÃN','Trên 20 giao dịch trong 30 ngày gắn với tỷ lệ mua 2,50%',
    small_group_bars(['recency','frequency','monetary'],['R · ngày từ trx cuối','F · số trx / 30 ngày','M · |amount|, mã 11']),
    'Ý nghĩa cho mô hình: dùng R/F/M cùng nhau; giữ riêng nhóm không có log hoặc không xác định.',
    'Train 02–08/2022; feature 30 ngày trước cutoff. Cột M là tổng trị tuyệt đối amount currency 11, đơn vị nguồn ẩn danh.',
    'aggregate_data.json → rfm (DuckDB trên feature train đã chuẩn hóa)',
    'Ngưỡng nhóm là mô tả cố định, không tối ưu target. Pooled train có 701.568 dòng và 8.672 dương. Không cộng currency khác nhau. Không suy hiệu ứng nhân quả hay hiệu quả ngoài mẫu từ chart này.')

# Composite category graphic: one treemap with a quantitative diversity annotation.
top=cat[:5];other=38764930-sum(int(r['n']) for r in top)
catvals=[(r['value'],int(r['n'])) for r in top]+[('Khác',other)]
b='';x=0
for i,(name,n) in enumerate(catvals):
    w=1160*n/38764930
    b+=rect(x,35,w-4,220,['--accent','--accent-2','--accent-3','--text-2','--good','--surface-2'][i])
    color='--text-1' if i==5 else '--accent-ink'
    b+=tx(x+12,86,'Mã '+name if name!='Khác' else name,20,color,weight=650)+tx(x+12,120,fmt(n/38764930*100,1)+'%',20,color)
    x+=w
bands=[r for r in rows('deliverables/eda/tables/feature_band_target_rates.csv') if r['feature']=='trx_30d_event_type_nunique']
b+=tx(0,302,'Đa dạng event_type: nhóm 20% cao nhất có tỷ lệ mua '+fmt(float(bands[-1]['target_pct']),2)+'%',24,'--accent',weight=700)
b+=tx(0,337,'Tương đương '+fmt(float(bands[-1]['target_pct'])/(base*100),2)+'× prevalence train; chưa phải lift riêng từng mã giao dịch.',20,'--text-2')
add('C · TRX / DANH MỤC','Nhóm giao dịch đa dạng nhất có tỷ lệ mua 3,03%',svg(b,'Treemap event_type toàn kỳ và lift theo độ đa dạng trên train'),
    'Ý nghĩa cho mô hình: dùng count, tỷ trọng và nunique; kiểm tra từng mã bằng ablation theo thời gian.',
    'Diện tích = tỷ trọng dòng trx toàn kỳ; dòng lift dùng 5 nhóm bằng số dòng, xếp theo nunique trên train (có thể tách ties).',
    'transaction_categories_full.csv; deliverables/eda/tables/feature_band_target_rates.csv',
    '55 event_type là mã ẩn danh, chưa có tên danh mục nghiệp vụ. Phân phối toàn kỳ chỉ mô tả, vocabulary mô hình cần fit trên train. Chưa có lift đã kiểm chứng theo từng mã.')

dialog=[r for r in monthly if r['source']=='dialog']
add('C · DIALOG / CẤU TRÚC & THỜI GIAN','Dialog chỉ phủ 46,2% khách và rất thưa ở từng tháng',
    trend([r['month'][5:]+'/'+r['month'][2:4] for r in dialog],[('ID active / tháng',[int(r['customers'])/1000 for r in dialog],'--accent')],10,[0,2,4,6,8,10],'k'),
    'Ý nghĩa cho mô hình: tách “không có hội thoại” khỏi metadata chưa xác định.',
    '217.205 sự kiện từ 10/02/2021; 46.320 ID toàn kỳ. Đường thể hiện ID có ≥1 dialog trong mỗi tháng (nghìn ID).',
    'monthly_activity.csv; descriptive.modalities/dialog_summary.csv',
    'Mỗi sự kiện có client_id, event_time, embedding (768 phần tử), thêm fold partition. Không có văn bản, độ dài hội thoại hay duration trong schema; không giả lập các trường này. Median toàn bộ khách = 0 dialog, với khách có log = 3 dialog toàn kỳ.')

dd=aggregate['rfm']['dialog']
add('C · DIALOG / LIÊN HỆ NHÃN','Tần suất dialog có liên hệ với nhãn, embedding chưa được thử',
    bars([r['group'].split('. ',1)[1]+' dialog / 30 ngày' for r in dd],[r['rate_pct'] for r in dd],maximum=max(r['rate_pct'] for r in dd)*1.22),
    'Ý nghĩa cho mô hình: giữ metadata làm mốc; thử embedding để đo phần tín hiệu bổ sung.',
    'Tỷ lệ mua = số dương / khách–tháng từng nhóm; train 02–08/2022, cửa sổ 30 ngày trước cutoff.',
    'aggregate_data.json → rfm.dialog; deliverables/baseline-final/REPORT.md',
    'Baseline hiện có count, active days và recency; chưa có kết quả nội dung/PCA/embedding. Cần fit PCA trên train của từng fold và so metadata với metadata + embedding trên cùng cohort.')

geo=[r for r in monthly if r['source']=='geo']
add('C · GEO / ĐỘ PHỦ & CHẤT LƯỢNG','Geo đến cuối 2021; lượng log lớn chưa bảo đảm tín hiệu mạnh',
    trend([r['month'][5:]+'/'+r['month'][2:4] for r in geo],[('ID active / tháng',[int(r['customers'])/1000 for r in geo],'--accent-3')],80,[0,20,40,60,80],'k'),
    'Ý nghĩa cho mô hình: coi geo là nguồn bổ trợ cần ablation; không suy quãng đường từ mã số.',
    '66.295.724 dòng, 72.573 ID; bắt đầu 11/12/2021. Chart đếm ID có log từng tháng; tháng đầu chỉ có một phần lịch.',
    'monthly_activity.csv; descriptive.modalities/geo_summary.csv; feature_target_associations.csv',
    'Không có null trong cột geo theo audit; điều đó không chứng minh không có nhiễu đo vị trí. Mean số geohash_6 trong 30 ngày: y0=10,65 và y1=13,37; r=0,0168. Mã geohash đã ẩn danh, chưa giải mã thành địa lý; unique code phản ánh đa dạng vùng, không phải khoảng cách.')

add('D · TỔNG HỢP TÍN HIỆU','Hành vi bổ sung tín hiệu cho lịch sử mua: AP đạt 0,0455',
    bars(['Rule mua T−1','Lịch sử mua (15 biến)','Hành vi (141 biến)','Hành vi + mua (156)'],[float(metrics[k]['mean_ap']) for k in ['previous_month_purchase','purchase','behavior','combined']],unit='',maximum=.06,colors=['--text-3','--accent-2','--accent-3','--accent']),
    'Ý nghĩa cho mô hình: ưu tiên B+H; giá trị tăng thêm riêng trx/geo/dialog vẫn chưa được đo.',
    'AP (Average Precision), trung bình đều 3 tháng validation 09–11/2022; cùng khách, split, cấu hình CatBoost.',
    'deliverables/baseline-final/summary.csv; REPORT.md',
    'AP là phiên bản báo cáo diện tích PR theo bước precision–recall, không gọi nhầm là tích phân hình thang. Recall@1% của B+H = 9,74%, H = 6,44%, B = 8,89%. Chênh B+H−H = +3,30 đpt, bootstrap 95% [+2,12;+4,28]. Không có ablation bỏ từng nguồn, không suy từ feature importance thành đóng góp độc lập.')

common=sum(int(r['common_hits']) for r in overlap)
mlonly=sum(int(r['ml_only_hits']) for r in overlap)
ruleonly=sum(int(r['rule_only_hits']) for r in overlap)
add('D · PHÂN TÍCH LỖI','Baseline tìm thêm người mua mới, nhưng vẫn bỏ sót nhóm rule bắt được',
    bars(['Cả baseline và rule bắt đúng','Chỉ baseline bắt đúng','Chỉ rule bắt đúng'],[common,mlonly,ruleonly],unit=' khách–tháng',maximum=240,colors=['--accent-2','--accent','--warn']),
    'Ý nghĩa cho mô hình: thêm lịch sử mua và kiểm tra riêng khách quay lại / chưa từng ghi nhận mua.',
    'Đếm true positive trong giao/hiệu hai danh sách top 1%; cộng 09–11/2022, mỗi danh sách 1.003 khách/tháng.',
    'docs/baseline_diagnostics_20261110/overlap_top1.csv',
    'Chẩn đoán này thuộc baseline hành vi gốc 153 biến ngày 10/11, không phải B+H 156 biến. Khách có thể lặp qua tháng. Tổng true positives baseline=238, rule=169; không phải số khách unique. B+H mới chưa có phân tích lỗi tương đương, cần tái chạy diagnostics trước kết luận về nhóm sai còn lại.')

add('D · SNAPSHOT','Snapshot tháng lịch đã có bằng chứng; cutoff bất kỳ vẫn là thí nghiệm',
    table(['Thiết kế','Đã đo','Kết quả / giới hạn'],[
    ['Tháng lịch T−1 → nhãn tháng T','Có · baseline 09–11/2022','75 biến recent: Recall@1% 7,55%'],
    ['Tháng gần nhất + lịch sử cũ','Có · cùng protocol','153 biến: 8,70% (+1,15 đpt)'],
    ['30/84 ngày trước đầu tháng','Có feature và EDA','Chưa so metric cửa sổ cùng model'],
    ['Cutoff T bất kỳ, nhãn 30 ngày tới','Chưa có nhãn sản phẩm theo ngày','Không gán y tháng cho rolling 30 ngày']],[34,28,38]),
    'Ý nghĩa cho mô hình: giữ tháng lịch làm mốc; thử cửa sổ mới trên cùng cutoff trước khi đổi bài toán.',
    'So recent vs history thuộc run hành vi gốc, cấu hình cố định; các cửa sổ chưa chạy được ghi đúng trạng thái.',
    'BASELINE_REPORT.md; docs/mbd_temporal_coverage/assessment.md',
    '30 ngày khác tháng lịch 28–31 ngày; 84 ngày khác 3 tháng. Older và recent không chồng nhau. Không lấy median all trừ median recent. Các cutoff ngày chồng lấn không phải tập test độc lập.')

fitrows=[('Quy mô','Tốt','100.224 khách; 1,20M mẫu tháng',0),('Lịch sử','Có điều kiện','2 năm trx; chỉ 12 tháng nhãn',1),('Lệch nhãn','Khó','1,07% dương; cần AP / Recall@K',2),('Độ phủ nguồn','Có điều kiện','35,34% có đủ ba nguồn toàn kỳ',1),('Leakage / khả dụng','Cần khóa','SLA nhãn và timezone chưa xác minh',2),('Tín hiệu quan sát','Tốt','B+H vượt H trên validation',0)]
b=''
for i,(name,status,evidence,level) in enumerate(fitrows):
    y=i*53
    b+=tx(0,y+29,name,21,weight=650)+rect(245,y+3,230,40,['--good-soft','--warn-soft','--bad-soft'][level])+tx(360,y+30,status,20,['--good','--warn','--bad'][level],'middle',650)+tx(510,y+29,evidence,21,'--text-2')
add('E · MỨC ĐỘ PHÙ HỢP','Dữ liệu phù hợp để nghiên cứu xếp hạng, còn điều kiện vận hành',svg(b,'Heatmap đánh giá mức phù hợp theo sáu tiêu chí'),
    'Ý nghĩa cho mô hình: tiếp tục baseline có kiểm soát; chưa dùng validation làm cam kết chất lượng triển khai.',
    'Heatmap là đánh giá định tính theo bằng chứng bên phải; xanh = thuận lợi, vàng = có điều kiện, đỏ = cần xử lý.',
    'Tổng hợp inventory, coverage, target audit và baseline-final/REPORT.md',
    'Đây không phải điểm số thống kê hoặc ngưỡng chuẩn bên ngoài. Một năm nhãn chưa đo được ổn định nhiều năm; test độc lập chưa mở.')

add('E · RỦI RO & CÁCH XỬ LÝ','Khóa khả dụng nhãn và cutoff trước khi tối ưu thêm mô hình',
    table(['Rủi ro quan sát','Cách xử lý cụ thể'],[
    ['Trước 02/2022 không có nhãn; chưa biết SLA','Giữ làm lịch sử; unknown không thành 0; xác nhận available_at'],
    ['Geo/dialog đến muộn, nhiều khách không có log','Giữ cohort; cờ có nguồn; count 0 khác mean/recency NaN'],
    ['Nhãn lệch và prevalence thay đổi','AP, Recall@K theo tháng; calibration theo nhãn đã khả dụng'],
    ['Cột phụ nhãn chưa tái tạo được công thức','Loại trans_count, diff_trans_date khỏi predictor'],
    ['Biên thời gian / encoder có thể nhìn tương lai','event_time < cutoff; fit vocabulary/PCA chỉ trên train']],[44,56]),
    'Ý nghĩa cho mô hình: protocol đúng thời điểm là điều kiện để metric có giá trị sử dụng.',
    'Bảng ánh xạ rủi ro audit sang quy tắc xử lý. Không có bằng chứng “target null sáu tháng đầu” trong bảng nhãn.',
    'docs/propensity_feasibility/remaining_questions/assessment.md; docs/baseline_feature_contract_20261111.md',
    'Baseline nghiên cứu đang giả định label delay=0 và UTC. Độ trễ dương cần dựng lại lịch train/calibration và feature target quá khứ. Ngoại lai amount là cờ mô tả, không tự xóa/cap; thống kê toàn kỳ không dùng làm ngưỡng cho quá khứ.')

add('E · KẾT LUẬN & HƯỚNG ĐI','Đủ tín hiệu để tiếp tục: khóa protocol, đo từng nguồn, rồi mở test',
    flow([('01 · Khóa thời điểm',['Xác nhận SLA nhãn / timezone','Chốt model và calibration']),('02 · Đo phần tăng thêm',['H + trx / dialog / geo','Cửa sổ mới; embedding sau']),('03 · Kiểm chứng độc lập',['Test 01/2023 sau khi khóa','Theo dõi AP, Recall@K, drift'])]),
    'Ý nghĩa cho mô hình: B+H là mốc hiện tại — Recall@1% 9,74%; dữ liệu phù hợp có điều kiện.',
    'Kết luận dựa trên validation 09–11/2022. Test 01/2023 chưa được đánh giá; thứ tự việc tiếp theo là đề xuất.',
    'deliverables/baseline-final/REPORT.md',
    'Bằng chứng cải thiện offline chưa đo tác động tiếp cận, số mua tăng thêm hoặc ROI. Nếu thực tế SLA khác giả định, backtest lại lịch trước khi dùng kết quả cho vận hành.')

# Appendix: details requested in the longer outline, outside the 19-slide narrative.
dist=aggregate['positive_month_histogram']
add('APPENDIX A1 · TARGET / ĐUÔI PHÂN PHỐI','81,6% người từng mua chỉ dương ở một tháng quan sát',
    bars(['Không dương tháng nào','Dương 1 tháng','Dương 2 tháng','Dương ≥3 tháng'],[sum(r['customers'] for r in dist if r['months']==0),sum(r['customers'] for r in dist if r['months']==1),sum(r['customers'] for r in dist if r['months']==2),sum(r['customers'] for r in dist if r['months']>=3)],unit=' ID',maximum=100000),
    'Ý nghĩa cho mô hình: lịch sử mua có đuôi phải, nhưng còn nhiều khách mua lần đầu được ghi nhận.',
    'Số tháng y_any=1 theo ID trong 02/2022–01/2023; không phải số lần mua hàng. 10.411 ID từng dương; 8.496 chỉ một tháng.',
    'aggregate_data.json → positive_month_histogram',
    'Targets chỉ có nhãn nhị phân theo tháng nên không đo được số đơn hoặc số lần mua trong một tháng. Max số tháng dương=8; không khẳng định phân phối power law.',True)

coh=rows('docs/propensity_feasibility/any_product/cohorts_2022_12.csv')
keys=['no_events_all_available_history','has_history_no_events_last_3_calendar_months','has_events_last_3_calendar_months','trx_only_all_history','all_three_sources_all_history']
lookup={r['cohort']:r for r in coh}
add('APPENDIX A2 · TARGET / COHORT','Cohort thiếu hành vi không đồng nghĩa xác suất mua bằng 0',
    bars(['Không có lịch sử trước T','Có cũ, trống 3 tháng gần','Có log 3 tháng gần','Chỉ trx trong quá khứ','Đủ 3 nguồn trong quá khứ'],[float(lookup[k]['positive_rate'])*100 for k in keys],maximum=2.3),
    'Ý nghĩa cho mô hình: giữ cold-start và thiếu nguồn; báo metric tách nhóm thay vì drop toàn bộ.',
    'Cutoff 01/12/2022, nhãn tháng 12; cohort chỉ dùng sự kiện trước cutoff. Các nhóm cuối có chồng lấn.',
    'docs/propensity_feasibility/any_product/cohorts_2022_12.csv',
    'Chỉ mô tả, không chọn feature từ calibration. Nhóm không lịch sử có 699 ID, 13 dương; nhóm cũ nhưng trống 3 tháng có 5.122 ID, 9 dương. Cỡ mẫu khác nhau, không diễn giải tỷ lệ cao là bằng chứng nhân quả.',True)

add('APPENDIX A3 · TRX / SPARSITY','Trx có đuôi rất dài: median 164, cực đại hơn một triệu dòng/ID',
    table(['Thống kê toàn lịch sử','Trx / ID','Hàm ý'],[
    ['Không có trx','1.503 ID (1,50%)','Giữ trong cohort scoring'],
    ['Median · P95 · P99','164 · 1.378 · 2.234','Dùng log1p và rate'],
    ['Trung bình · cực đại','386,8 · 1.020.773','Mean nhạy với đuôi lớn'],
    ['Ô tháng trống xen giữa first–last log','182.757 khách–tháng','Không tự gọi là mất dữ liệu'],
    ['Không có trx 3 tháng trước 01/12/2022','13.772 ID; 83 dương','Nhóm im lặng vẫn có nhãn dương']],[42,29,29]),
    'Ý nghĩa cho mô hình: mô tả inactivity theo cutoff; không dùng last-event tương lai để lọc khách.',
    'Quantile theo toàn bộ 100.224 ID, gồm count=0. Inactivity ở dòng cuối dùng ba tháng lịch trước cutoff, không phải 84 ngày.',
    'review/events_per_customer.csv; remaining_questions/absence_patterns.csv; any_product/cohorts_2022_12.csv',appendix=True)

add('APPENDIX A4 · TRX / LEAKAGE','Trx trước thời kỳ nhãn cung cấp lịch sử hợp lệ cho cutoff sau đó',
    table(['Kiểm tra','Quy tắc / bằng chứng'],[
    ['Sự kiện trước 02/2022','Được dùng cho feature tại cutoff muộn hơn'],
    ['Sự kiện trong / sau tháng T','Loại khỏi feature dự đoán tháng T'],
    ['trans_count nguồn','Không tái tạo được từ trx; chưa đưa vào X'],
    ['diff_trans_date nguồn','Chưa chốt ngữ nghĩa / cutoff; chưa đưa vào X'],
    ['Ngưỡng ngoại lai / vocabulary','Fit theo train; không lấy ngưỡng toàn kỳ cho model quá khứ']],[34,66]),
    'Ý nghĩa cho mô hình: leakage phụ thuộc thời điểm khả dụng và cách tạo feature, không phụ thuộc có nhãn hay chưa.',
    'Audit 612 cấu hình trans_count không có công thức khớp toàn bộ. Điều đó chưa tự chứng minh cột nguồn dùng tương lai.',
    'docs/propensity_feasibility/remaining_questions/assessment.md',appendix=True)

gr=aggregate['rfm']['geo']
add('APPENDIX A5 · GEO / LIÊN HỆ NHÃN','Đa dạng mã geo có liên hệ yếu; chưa kết luận đóng góp mô hình',
    bars([r['group'].split('. ',1)[1]+' mã geo / 30 ngày' for r in gr],[r['rate_pct'] for r in gr],maximum=max(r['rate_pct'] for r in gr)*1.3),
    'Ý nghĩa cho mô hình: kiểm tra H+trx+dialog so với thêm geo trước khi quyết định giữ hoặc bỏ.',
    'Train 02–08/2022; số mã geohash_6 phân biệt trong 30 ngày trước cutoff. Tỷ lệ dương từng nhóm, không đo khoảng cách.',
    'aggregate_data.json → rfm.geo; feature_target_associations.csv',
    'Không có null trong geo không có nghĩa dữ liệu sạch nhiễu. Chưa kiểm chứng tốc độ, khoảng cách hay vị trí thật. Không diễn giải cột mã số ẩn danh như tọa độ.',True)

add('APPENDIX A6 · ĐÓNG GÓP THEO NGUỒN','Chưa có ΔAP khi bỏ từng nguồn: cần ablation trên cùng nền H',
    table(['Đối chứng cần chạy','Câu hỏi','Trạng thái'],[
    ['H → H+trx','Trx tăng AP / Recall@K bao nhiêu?','Chưa chạy'],
    ['H+trx → thêm dialog metadata','Dialog bổ sung ngoài trx?','Chưa chạy'],
    ['H+trx+dialog → thêm geo','Geo bổ trợ hay không đáng kể?','Chưa chạy'],
    ['Metadata → metadata + embedding','Nội dung thêm gì ngoài tần suất?','Chưa chạy'],
    ['Full → bỏ lần lượt từng nguồn','ΔAP có ổn định theo tháng?','Chưa chạy']],[44,38,18]),
    'Ý nghĩa cho mô hình: giữ split/cohort/config; báo chênh AP, Recall@K và bootstrap theo khách.',
    'Bảng thiết kế thí nghiệm, không phải kết quả. Feature importance và tương quan không thay thế ablation có kiểm soát.',
    'deliverables/baseline-final/REPORT.md → Phần chưa thực hiện',appendix=True)

add('APPENDIX A7 · CÁCH TÍNH','Các chỉ số dùng mẫu số khác nhau và cần đọc đúng phạm vi',
    table(['Chỉ số / thiết kế','Cách tính / phạm vi'],[
    ['Prevalence · lift nhóm','Dương / số mẫu; lift = prevalence nhóm / prevalence toàn train'],
    ['Recall@1% · Precision@1%','TP trong top 1.003 / tổng dương; TP / 1.003'],
    ['AP · macro theo tháng','Average Precision trên thứ hạng; trung bình đều 09–11/2022'],
    ['Bootstrap','500 lần paired resample theo ID; các tháng cùng ID đi cùng nhau'],
    ['Baseline vs rule','Cùng ngân sách; rule ties phá bằng hash cố định; không dùng yT'],
    ['Cutoff / nhãn khả dụng','event_time < cutoff; lịch sử target chỉ dùng khi available_at ≤ cutoff']],[35,65]),
    'Ý nghĩa cho mô hình: không so metric khác cohort, khác ngân sách hoặc khác giả định nhãn.',
    'B+H 156 biến: CatBoost depth 4, 200 cây, learning rate 0,04, L2 5; không resampling/class weight.',
    'deliverables/baseline-final/REPORT.md; docs/propensity_baseline_design.md',appendix=True)

CSS='''
:root{--bg:#f7f9fb;--surface:#fff;--surface-2:#eaf0f5;--text-1:#102b46;--text-2:#496174;--text-3:#667d8e;--accent:#087e8b;--accent-2:#5477ad;--accent-3:#a06d4a;--accent-ink:#fff;--border:#dce5ec;--border-strong:#a7bac8;--good:#21734e;--good-soft:#e2f1e9;--warn:#976312;--warn-soft:#fff1d6;--bad:#a7463e;--bad-soft:#f8e6e2;--font-sans:'Segoe UI',Arial,sans-serif;--radius:10px;--shadow:none}
body{background:var(--surface-2)}.slide{padding:35px 54px 32px;justify-content:flex-start;transition:opacity .12s;transform:none;background:var(--bg)}
.slide:before{content:'';position:absolute;top:0;left:0;width:9px;height:100%;background:var(--accent)}
.kicker{font-size:14px;letter-spacing:.13em;color:var(--accent);font-weight:750;margin:0 0 12px;text-transform:uppercase}
h2{font-size:34px;line-height:1.18;letter-spacing:-.025em;margin:0;max-width:1150px;font-weight:720}
.chart{position:absolute;left:54px;right:54px;top:157px;height:350px}.chart svg{width:100%;height:100%;overflow:visible}.chart svg text{font-family:var(--font-sans);font-variant-numeric:tabular-nums}
.meaning{position:absolute;left:54px;right:54px;top:530px;padding:14px 18px;background:var(--surface-2);border-left:4px solid var(--accent);font-size:21px;font-weight:600;line-height:1.35;margin:0}
.method{position:absolute;left:54px;right:54px;top:598px;font-size:15px;color:var(--text-2);line-height:1.4;margin:0}
.source{position:absolute;left:54px;right:120px;bottom:24px;font-size:11px;line-height:1.35;color:var(--text-3);margin:0;overflow-wrap:anywhere}.page{position:absolute;right:54px;bottom:23px;font-size:14px;color:var(--text-3)}
.table-wrap{height:100%;display:flex;align-items:center}table{border-collapse:collapse;width:100%;table-layout:fixed;font-size:20px;line-height:1.35}th{text-align:left;font-size:15px;color:var(--text-3);font-weight:650;letter-spacing:.02em;padding:10px 16px;border-bottom:2px solid var(--accent)}td{padding:13px 16px;border-bottom:1px solid var(--border);vertical-align:middle}td:first-child{font-weight:650}tbody tr:nth-child(2n){background:var(--surface)}
.deck-nav{position:fixed;bottom:8px;left:50%;transform:translateX(-50%);display:flex;gap:5px;z-index:50;opacity:.2;transition:opacity .2s}.deck-nav:hover,.deck-nav:focus-within{opacity:1}.deck-nav button{background:var(--surface);color:var(--text-1);border:1px solid var(--border);border-radius:5px;padding:5px 10px;cursor:pointer;font-size:12px}
.progress-bar{height:3px!important}.notes{display:none}.hint-bar{display:none!important}
@media print{.deck-nav{display:none!important}.slide{break-after:page}.chart{height:350px}}
'''

parts=[]
for i,s in enumerate(slides,1):
    number=f'{i:02d} / 19' if not s['appendix'] else f'A{i-19:02d} / 07'
    parts.append(f'''<section class="slide{' is-active' if i==1 else ''}" data-title="{esc(s['title'])}" data-section="{esc(s['section'])}">
    <p class="kicker">MBD-MINI / {esc(s['section'])}</p><h2>{esc(s['title'])}</h2>
    <div class="chart">{s['chart']}</div><p class="meaning">{esc(s['meaning'])}</p>
    <p class="method">{esc(s['method'])}</p><p class="source">Nguồn: {esc(s['source'])} · cập nhật 11/11/2026</p>
    <span class="page">{number}</span><div class="notes">{esc(s['notes'] or s['method'])}</div></section>''')

template=(SKILL/'templates/full-decks/tech-sharing/index.html').read_text(encoding='utf-8') if SKILL.exists() else '<!doctype html>'
assert 'tpl-tech-sharing' in template or not SKILL.exists()
basecss=(ASSETS/'base.css').read_text(encoding='utf-8')
runtime=(ASSETS/'runtime.js').read_text(encoding='utf-8').replace('</script', '<\\/script')
page=f'''<!doctype html>
<!-- Adapted from html-ppt tech-sharing + chart/table/flow/gantt templates. html-ppt MIT © 2026 lewis. -->
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="MBD-mini: 19 slide chính và 7 phụ lục về tính phù hợp dữ liệu propensity."><title>MBD-mini · Dữ liệu có đủ tín hiệu cho propensity?</title>
<style>{basecss}\n{CSS}</style></head><body class="tpl-tech-sharing"><main class="deck" data-w="1280" data-h="720">{''.join(parts)}</main>
<nav class="deck-nav" aria-label="Điều hướng"><button onclick="location.hash='#/1'">Đầu</button><button onclick="document.dispatchEvent(new KeyboardEvent('keydown',{{key:'ArrowLeft'}}))">←</button><button onclick="document.dispatchEvent(new KeyboardEvent('keydown',{{key:'ArrowRight'}}))">→</button><button onclick="location.hash='#/20'">Phụ lục</button><button onclick="document.dispatchEvent(new KeyboardEvent('keydown',{{key:'o'}}))">Mục lục (O)</button><button onclick="document.dispatchEvent(new KeyboardEvent('keydown',{{key:'s'}}))">Trình chiếu (S)</button></nav>
<script>{runtime}</script></body></html>'''
(OUT/'index.html').write_text(page,encoding='utf-8',newline='\n')
(OUT/'aggregate_data.json').write_text(json.dumps(aggregate,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
(OUT/'outline.json').write_text(json.dumps([{k:v for k,v in s.items() if k!='chart'} for s in slides],ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'main_slides':sum(not s['appendix'] for s in slides),'appendix':sum(s['appendix'] for s in slides),'persistence_ratio':ratio,'trx_months':len(trx),'output':str(OUT/'index.html')},ensure_ascii=False))
