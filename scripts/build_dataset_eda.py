"""Build the standalone Vietnamese 30+10 HTML deck from aggregate EDA tables."""
from pathlib import Path
import html
import itertools
import json
import math
import shutil
import hashlib
import importlib.metadata
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables/mbd-mini-dataset-eda'
SKILL=Path.home()/'.codex/skills/html-ppt'
T=OUT/'tables'
slides=[]

def esc(s):return html.escape(str(s),quote=True)
def vn(x,d=0):
    if x is None or pd.isna(x):return '—'
    return f'{float(x):,.{d}f}'.replace(',','_').replace('.',',').replace('_','.')
def pct(x,d=2):return vn(float(x)*100,d)+'%'
def read(name):return pd.read_csv(T/(name+'.csv'))
def cell(x):return esc(x)
def table(headers,rows,widths=None,small=False):
    cols='' if widths is None else '<colgroup>'+''.join(f'<col style="width:{w}%">' for w in widths)+'</colgroup>'
    return '<div class="table-wrap'+(' small' if small else '')+'"><table>'+cols+'<thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def svg(body,w=1120,h=360,label='Biểu đồ dữ liệu MBD-mini'):
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}">{body}</svg>'
def tx(x,y,s,size=19,color='--text-1',anchor='start',weight=400):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="var({color})" text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>'
def rect(x,y,w,h,color='--accent-2',r=4):return f'<rect x="{x}" y="{y}" width="{max(0,w)}" height="{h}" rx="{r}" fill="var({color})"/>'
def line(x1,y1,x2,y2,color='--border-strong',width=1):return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="var({color})" stroke-width="{width}"/>'
def bars(labels,vals,lows=None,highs=None,fmt=None,colors=None,sub=None):
    fmt=fmt or (lambda v:vn(v,2));maximum=max(highs if highs is not None else vals)*1.12 or 1
    body='';n=len(labels);step=min(47,300/max(n,1));left=265;right=890
    for i,(label,val) in enumerate(zip(labels,vals)):
        y=25+i*step
        body+=tx(0,y+18,label,18)+rect(left,y, (right-left)*val/maximum,25,(colors or ['--accent-2']*n)[i])
        if lows is not None and pd.notna(lows[i]) and pd.notna(highs[i]):
            lo=left+(right-left)*lows[i]/maximum;hi=left+(right-left)*highs[i]/maximum
            body+=line(lo,y+12,hi,y+12,'--text-1',2)+line(lo,y+6,lo,y+18,'--text-1',2)+line(hi,y+6,hi,y+18,'--text-1',2)
        body+=tx(915,y+19,fmt(val),19,weight=650)
        if sub:body+=tx(915,y+36,sub[i],12,'--text-2')
    body+=tx(left,345,'Thanh ngang: ước lượng điểm'+(' · vạch đen: CI 95%' if lows is not None else ''),15,'--text-2')
    return svg(body)
def mini(feature,title):
    f=rates[(rates.kind=='feature')&(rates.metric==feature)&(rates.target=='y_any')].copy()
    # Keep the explicit zero group separate; never split equal values into deciles.
    b=tx(0,22,title,21,weight=700);maximum=max(1.2,float(f.lift.max())*1.08,float(f.lift_ci_high.max())*1.05)
    left=137;right=280
    for i,(_,r) in enumerate(f.iterrows()):
        y=55+i*46
        label=r['group'].replace('Unknown / no usable event','Thiếu / chưa thấy').replace('<= ','≤ ').replace(', ', '; ')
        b+=tx(0,y+13,label,15)+rect(left,y-3,(right-left)*r.lift/maximum,22,'--accent-2')
        if pd.notna(r.lift_ci_low) and pd.notna(r.lift_ci_high):
            lo=left+(right-left)*r.lift_ci_low/maximum;hi=left+(right-left)*r.lift_ci_high/maximum
            b+=line(lo,y+8,hi,y+8,'--text-1',1.5)+line(lo,y+3,lo,y+13,'--text-1')+line(hi,y+3,hi,y+13,'--text-1')
        b+=tx(323,y+13,vn(r.lift,2)+'×'+(' *' if r.small_positive else ''),14,anchor='end',weight=650)
        b+=tx(0,y+32,'n='+vn(r.n_total)+' · n+='+vn(r.n_positive),12.5,'--text-2')
    ref=left+(right-left)/maximum
    b+=f'<path d="M{ref},43 V{min(338,55+len(f)*46)}" stroke="var(--warn)" stroke-dasharray="4 4"/>'
    return svg(b,330,360,title)
def panels(items):return '<div class="panels">'+''.join('<div class="panel">'+x+'</div>' for x in items)+'</div>'
def linechart(frame,targets,unit='rate',count=False):
    data=frame.copy();months=sorted(data.month.unique());left,right,top,bottom=65,1085,30,276
    maximum=float(data[data.target.isin(targets)].ci_high.max())*1.13 if 'ci_high' in data else float(data[unit].max())*1.15
    maximum=max(maximum,1e-6);b=''
    for tick in np.linspace(0,maximum,5):
        y=bottom-(bottom-top)*tick/maximum;b+=line(left,y,right,y)+tx(left-10,y+5,pct(tick,1),15,'--text-2','end')
    palette=['--target-1','--target-2','--target-3','--target-4','--accent-2']
    for j,target in enumerate(targets):
        group=data[data.target==target].set_index('month');pts=[];color=palette[j] if len(targets)>1 else '--accent-2'
        for i,m in enumerate(months):
            r=group.loc[m];x=left+i*(right-left)/(len(months)-1);y=bottom-(bottom-top)*r[unit]/maximum;pts.append((x,y))
            if 'ci_high' in group:
                b+=line(x,bottom-(bottom-top)*r.ci_low/maximum,x,bottom-(bottom-top)*r.ci_high/maximum,color,1.4)
            b+=f'<circle cx="{x}" cy="{y}" r="3" fill="var({color})"/>'
            if count and target=='y_any':b+=tx(x,y-14,vn(r.n_positive),12,color,'middle')
        b+='<polyline points="'+' '.join(f'{x},{y}' for x,y in pts)+f'" fill="none" stroke="var({color})" stroke-width="2.5"/>'
        b+=rect(95+j*200,334,20,4,color,1)+tx(123+j*200,342,target,16,'--text-2')
    for i,m in enumerate(months):b+=tx(left+i*(right-left)/(len(months)-1),305,str(m)[5:7]+'/'+str(m)[2:4],15,'--text-2','middle')
    return svg(b)
def coverage(source):
    f=monthly[monthly.source==source];b='';left=65;right=1080
    for j,(col,title,scale,suffix) in enumerate([('events','Số sự kiện / tháng',1e6,' triệu'),('active_clients','Khách có log / tháng',1000,' nghìn')]):
        top=30+j*170;bottom=top+110;maximum=float(f[col].max())/scale*1.15
        b+=tx(0,top-5,title,17,weight=650)
        for tick in [0,maximum/2,maximum]:
            y=bottom-(bottom-top)*tick/maximum;b+=line(left,y,right,y)+tx(left-10,y+5,vn(tick,1),14,'--text-2','end')
        pts=[]
        for i,(_,r) in enumerate(f.iterrows()):
            x=left+i*(right-left)/(len(f)-1);y=bottom-(bottom-top)*(r[col]/scale)/maximum;pts.append((x,y))
            if j==1 and i%2==0:b+=tx(x,bottom+24,r.month[5:7]+'/'+r.month[2:4],13,'--text-2','middle')
        b+='<polyline points="'+' '.join(f'{x},{y}' for x,y in pts)+'" fill="none" stroke="var(--accent-2)" stroke-width="3"/>'
        b+=tx(right,top-5,'Đơn vị:'+suffix,13,'--text-2','end')
    return svg(b)
def add(section,title,chart,insight,method,sources,notes=''):
    n=len(slides)+1
    if isinstance(sources,str):sources=[sources]
    # Notes carry derivation and caveats; they never appear on the audience slide.
    narration=(notes+' '+insight+' '+method+' Khi trình bày, hãy chỉ rõ đơn vị quan sát và mẫu số trước khi đọc tỷ lệ. '
       'Một khách có thể xuất hiện ở nhiều tháng, vì vậy kết quả gộp không phải số người độc lập. '
       'Những khác biệt đang thấy mô tả tập dữ liệu này và không giải thích nguyên nhân hành vi. '
       'Không tìm thấy sự kiện trong file cũng chưa đủ để xác nhận khách không hoạt động ngoài phạm vi thu thập. '
       'Nếu người nghe muốn kiểm tra chi tiết, mở bảng số liệu nguồn của trang này: bảng giữ cả cỡ mẫu, định nghĩa và giới hạn tính toán. '
       'Với tỷ lệ hoặc lift, đọc khoảng tin cậy cùng số mẫu dương; nhóm gắn dấu sao là tín hiệu cần thận trọng. '
       'Từ phát hiện này, chuyển sang câu hỏi tiếp theo về thông tin quan sát được và điều còn chưa xác minh.')
    slides.append(dict(number=n,section=section,title=title,chart=chart,insight=insight,method=method,sources=sources,notes=narration))


manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
rates=read('rates_cluster_ci');target_month=read('target_monthly');monthly=read('source_monthly')
rates['ci_status']=np.where(rates.n_positive.eq(0),'not_estimable_zero_observed_positive',
                    np.where(rates.valid_replicates.lt(2000),'unstable_zero_bootstrap_denominator',
                          np.where(rates.n_positive.lt(30),'small_positive_sample','estimated')))
# Empirical bootstrap cannot create an unseen positive; a zero-width interval
# at zero would overstate what a sample with no positives can establish.
for col in ['ci_low','ci_high','lift_ci_low','lift_ci_high']:
    rates.loc[rates.n_positive.eq(0)|rates.valid_replicates.lt(2000),col]=np.nan
rates.to_csv(T/'rates_cluster_ci.csv',index=False,lineterminator='\n')
manifest['tables']['rates_cluster_ci'].update(columns=list(rates.columns),sha256=hashlib.sha256((T/'rates_cluster_ci.csv').read_bytes()).hexdigest())
manifest['bootstrap']['unstable_policy']='Point retained; no CI when no observed positive or bootstrap denominator can be zero; explicit ci_status and valid replicate counts'
manifest['versions']={name:importlib.metadata.version(name) for name in ['duckdb','pandas','numpy','pyarrow','scipy']}
overview=read('source_overview');activity=read('customer_activity');freq=read('target_frequency')
history_stats=[]
for target,g in freq.groupby('target'):
    values=np.repeat(g.positive_months.to_numpy(),g.customers.to_numpy())
    for scope,vals in [('all_customers',values),('ever_positive',values[values>0])]:
        q=np.quantile(vals,[.25,.5,.75,.9,.95,.99])
        history_stats.append(dict(target=target,population=scope,customers=len(vals),mean=float(vals.mean()),
            p25=q[0],median=q[1],p75=q[2],p90=q[3],p95=q[4],p99=q[5],maximum=int(vals.max()),
            top1_positive_months_share=float(np.sort(vals)[-math.ceil(len(vals)*.01):].sum()/vals.sum()),
            top5_positive_months_share=float(np.sort(vals)[-math.ceil(len(vals)*.05):].sum()/vals.sum())))
history_path=T/'target_purchase_history_summary.csv'
pd.DataFrame(history_stats).to_csv(history_path,index=False,lineterminator='\n')
manifest['tables']['target_purchase_history_summary']=dict(rows=len(history_stats),columns=list(history_stats[0]),sha256=hashlib.sha256(history_path.read_bytes()).hexdigest())
manifest['html_build']=dict(template='presenter-mode-reveal',theme='corporate-clean',main_slides=30,appendix_slides=10,standalone=True)
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
trans=read('target_transitions_monthly');composition=read('buyer_composition_monthly');transition=read('target_transition_comparison')
summaries=read('feature_target_summary');embedding=read('embedding_quality').iloc[0]
prevalence=rates[rates.kind=='prevalence'];anypre=prevalence[prevalence.target=='y_any'].iloc[0]
N=int(manifest['checks']['clients']);numrows=int(manifest['checks']['label_rows'])

# A — Context, five slides.
add('A / BỐI CẢNH','MBD-mini: hiểu dữ liệu trước khi đọc tín hiệu',
    '<div class="hero"><div class="hero-title">Khách hàng.<br>Hành vi.<br><span>Nhãn theo tháng.</span></div><div class="hero-stats">'
    f'<div><b>{vn(N)}</b><span>khách hàng doanh nghiệp</span></div><div><b>{vn(numrows)}</b><span>quan sát khách–tháng</span></div><div><b>3 nguồn · 4 nhãn</b><span>TRX / DIALOG / GEO / TARGETS</span></div></div></div>',
    'Dataset Understanding & Exploratory Data Analysis · dữ liệu ngân hàng đã ẩn danh.',
    'MBD-mini tại máy · 12 tháng nhãn 02/2022–01/2023 · cập nhật 11/11/2026.',
    ['source_overview','target_monthly'],
    'Mở đầu bằng ba cấp độ: một sự kiện, một khách và một khách trong một tháng. MBD gốc mô tả khách hàng doanh nghiệp; không suy rộng sang toàn bộ khách hàng cá nhân. Mục tiêu bài này là hiểu dữ liệu và giới hạn diễn giải.')
add('A / BỐI CẢNH','Dataset ghi nhận kết quả sản phẩm trong một tháng',
    '<div class="flow"><div><small>TRƯỚC CUTOFF</small><h3>Lịch sử quan sát</h3><p>Giao dịch<br>Tương tác ngân hàng<br>Mã vùng địa lý</p></div><b class="flow-arrow">→</b><div><small>ĐẦU THÁNG T</small><h3>Cùng thời điểm thông tin</h3><p>Chỉ dùng event_time &lt; cutoff<br>Cửa sổ chính: tháng T−1</p></div><b class="flow-arrow">→</b><div><small>THÁNG KẾT QUẢ T</small><h3>Bốn nhãn 0 / 1</h3><p>Khách có kết quả dương<br>ở từng sản phẩm?</p></div></div>',
    'Bốn target là bốn nhãn sản phẩm có thể đồng thời dương; tên thương mại chưa được xác nhận.',
    'Quy ước nghiên cứu: tháng chứa mon là T; không dịch mon thêm một tháng.',
    ['temporal_alignment','target_monthly'])
add('A / BỐI CẢNH','Hơn 105 triệu sự kiện, nhưng số khách là 100 nghìn',
    bars([r.source.upper() for _,r in overview.iterrows()],list(overview.rows/1e6),fmt=lambda v:vn(v,3)+' triệu',
        sub=[vn(r.clients)+' khách có log' for _,r in overview.iterrows()]),
    'Số dòng sự kiện, số khách và số khách–tháng là ba mẫu số khác nhau.',
    'Toàn lịch sử detail; không cộng dữ liệu ptls vào detail vì đây là hai dạng biểu diễn.',
    ['source_overview','sequence_reconciliation'])
add('A / BỐI CẢNH','client_id nối các bảng có grain khác nhau',
    table(['Bảng','Một dòng là gì?','Thông tin quan sát'],[
      ['CLIENT_SPLIT','Một khách','client_id + fold'],['TARGETS','Một khách × một tháng','mon + 4 target + trans_count + diff_trans_date'],
      ['TRX','Một giao dịch','event_time + amount + currency + các mã'],['DIALOG','Một tương tác','event_time + embedding 768 chiều'],
      ['GEO','Một quan sát mã vùng','event_time + geohash_4 / 5 / 6']],[20,26,54]),
    'Tổng hợp từng nguồn về khách–tháng trước khi đối chiếu nhãn; tránh nhân dòng khi join nhiều nguồn.',
    'fold là metadata chia khách; giữ cùng client_id và kiểm tra quan hệ một–nhiều.',
    ['raw_schema','target_schema'])
begin=pd.Timestamp('2020-12-01');end=pd.Timestamp('2023-02-01');span=(end-begin).days
b=''
for i,(_,r) in enumerate(overview.iterrows()):
    y=40+i*70;x=220+(pd.Timestamp(r.first_event)-begin).total_seconds()/86400/span*860
    xend=220+(pd.Timestamp(r.last_event)-begin).total_seconds()/86400/span*860
    b+=tx(0,y+20,r.source.upper(),23,weight=650)+rect(x,y,xend-x,26,'--accent-2')+tx(x,y-9,str(r.first_event)[:10],15,'--text-2')
b+=tx(0,270,'TARGETS',23,weight=650)+rect(220+(pd.Timestamp('2022-02-01')-begin).days/span*860,249,(pd.Timestamp('2023-02-01')-pd.Timestamp('2022-02-01')).days/span*860,26,'--target-3')+tx(700,310,'Nhãn: 02/2022 → 01/2023',18,'--text-2')
add('A / BỐI CẢNH','Các nguồn bắt đầu khác nhau; cutoff phải thống nhất',svg(b),
    'TRX từ cuối 2020; DIALOG từ 02/2021; GEO từ 12/2021. Nhãn bắt đầu 02/2022.',
    'Timestamp gốc không ghi timezone. UTC là quy ước nghiên cứu; tháng đầu/cuối có thể chỉ ghi nhận một phần.',
    ['source_overview','temporal_alignment'])

# B — Labels, nine slides.
add('B / TARGETS','Target có nhãn sản phẩm và hai cột phụ trợ',
    table(['Nhóm','Các cột thực tế','Cách đọc'],[
      ['Khóa / thời gian','client_id · mon','Một dòng / khách / tháng'],['Nhãn sản phẩm','target_1 · target_2 · target_3 · target_4','0 / 1; sản phẩm không loại trừ lẫn nhau'],
      ['Cột phụ trợ','trans_count · diff_trans_date','Cửa sổ và công thức chưa xác nhận đầy đủ'],['Phân chia khách','fold','Cột từ partition Parquet; 5 fold']],[20,45,35]),
    'Không dùng tên trans_count để suy ra số lần mua sản phẩm; không mặc định cột phụ là lịch sử trước cutoff.',
    'Kiểm kê tất cả cột từ Parquet; mon lưu dạng chuỗi ngày cuối tháng.',
    ['target_schema','raw_schema'])
nulls=read('target_schema');nulls=nulls[nulls.column.isin(['target_1','target_2','target_3','target_4','diff_trans_date'])]
add('B / TARGETS','Bốn nhãn đầy đủ; null nằm ở diff_trans_date',
    bars(list(nulls.column),list(nulls.null_rate),fmt=lambda v:pct(v,3),sub=[vn(v)+' ô null' for v in nulls.null_count]),
    'Không có nhãn null trong bản hiện có. 120.786 ô diff_trans_date null không làm mất nhãn của dòng đó.',
    'Mẫu số: 1.202.688 khách–tháng. Ngoài thời kỳ nhãn là chưa có bảng nhãn, không phải nhãn âm.',
    ['target_schema','target_null_patterns','raw_missing'])
add('B / TARGETS','Chỉ nhãn gộp có prevalence quanh 1%',
    bars(list(prevalence.target),list(prevalence.rate),list(prevalence.ci_low),list(prevalence.ci_high),fmt=lambda v:pct(v,4),
       sub=['n+='+vn(r.n_positive) for _,r in prevalence.iterrows()]),
    'target_2 chỉ có 577 khách–tháng dương trên toàn kỳ: việc chia nhỏ nhóm dễ thiếu mẫu.',
    'Mẫu số từng nhãn: 1.202.688 nhãn hợp lệ. CI 95%: cluster bootstrap khách, 2.000 lần.',
    ['rates_cluster_ci'])
add('B / TARGETS','y_any là phép OR bảo thủ của bốn nhãn',
    table(['Điều kiện','y_any','Kết quả thực tế'],[
      ['Ít nhất một target = 1','1',vn(anypre.n_positive)+' khách–tháng'],
      ['Đủ bốn target = 0','0',vn(anypre.n_negative)+' khách–tháng'],
      ['Không có 1, còn target null','Unknown','0 dòng trong phiên bản này']],[53,15,32]),
    'y_any = 1,0688% trên toàn kỳ. Không cộng tỷ lệ của bốn target vì có nhãn cùng dương.',
    'Đã đối soát union bốn target và tính duy nhất (client_id, mon).',
    ['rates_cluster_ci','target_null_patterns'])
add('B / TARGETS','Tỷ lệ dương thay đổi trong 12 tháng quan sát',linechart(target_month,['target_1','target_2','target_3','target_4','y_any'],count=True),
    'Đọc biến động từng target cùng số mẫu dương; mỗi tháng có đủ 100.224 khách.',
    'Trục Y: tỷ lệ dương. Vạch: Wilson CI 95%. Số cạnh đường y_any là số khách dương; nhãn không null.',
    ['target_monthly'])
ff=freq[(freq.target=='y_any')&(freq.positive_months>0)];ever=int(ff.customers.sum());zero=N-ever
add('B / TARGETS','Số tháng dương cho thấy mức độ lặp lại được quan sát',
    bars([str(int(r.positive_months))+' tháng dương' for _,r in ff.iterrows()],list(ff.customers),fmt=lambda v:vn(v)+' khách'),
    vn(ever)+' khách từng có y_any dương; '+pct(int(ff.iloc[0].customers)/ever)+' trong số này chỉ có một tháng dương.',
    'Biểu đồ chỉ gồm khách từng dương. Đây là số tháng dương, không phải số giao dịch hoặc số lần mua trong đời.',
    ['target_frequency','target_purchase_history_summary'])
tr=rates[rates.kind=='transition'];b=''
maximum=float(tr.ci_high.max())*1.2
for j,target in enumerate(['target_1','target_2','target_3','target_4','y_any']):
    b+=tx(10,40+j*59,target,20,weight=600)
    for p,color,off in [(0,'--target-4',0),(1,'--accent-2',23)]:
        r=tr[(tr.target==target)&(tr.metric=='previous_'+str(p))].iloc[0];y=22+j*59+off
        b+=rect(180,y,630*r.rate/maximum,18,color)+tx(850,y+15,pct(r.rate,3),17,weight=650)
        b+=line(180+630*r.ci_low/maximum,y+9,180+630*r.ci_high/maximum,y+9,'--text-1',2)
        b+=tx(960,y+14,'n+='+vn(r.n_positive),13,'--text-2')
b+=rect(180,332,20,5,'--target-4')+tx(210,340,'Không mua T−1',17)+rect(560,332,20,5,'--accent-2')+tx(590,340,'Có mua T−1',17)
rr=transition[transition.target=='y_any'].iloc[0]
add('B / TARGETS','Mua tháng trước liên hệ thế nào với tháng tiếp theo?',svg(b),
    'Với y_any: '+pct(rr.repeat_rate)+' sau tháng dương, so với '+pct(rr.nonbuyer_rate)+' sau tháng âm; tỷ số '+vn(rr.relative_rate,2)+'×.',
    '11 cặp tháng lịch liên tiếp; chỉ dùng nhãn hợp lệ ở cả hai tháng. Vạch: CI 95% bootstrap theo khách.',
    ['rates_cluster_ci','target_transition_comparison','target_transitions_monthly'])
co=composition[composition.target=='y_any'];b='';months=sorted(co.month.unique());palette={'returning':'--accent-2','first_observed':'--target-3','history_unknown':'--target-4'}
for i,m in enumerate(months):
    g=co[co.month==m].set_index('buyer_group');x=25+i*91;y=285
    for category in ['returning','first_observed','history_unknown']:
        value=float(g.loc[category,'share']);h=value*230;y-=h;b+=rect(x,y,55,h,palette[category],0)
        if value>.16:b+=tx(x+27,y+h/2+5,pct(value,0),13,'--accent-ink','middle',650)
    b+=tx(x+27,309,m[5:7]+'/'+m[2:4],14,'--text-2','middle')+tx(x+27,40,'n='+vn(g.all_current_buyers.iloc[0]),12,'--text-2','middle')
for j,(category,label) in enumerate([('returning','Đã từng dương'),('first_observed','Lần đầu quan sát'),('history_unknown','Chưa đủ lịch sử')]):
    b+=rect(30+j*360,336,18,5,palette[category])+tx(59+j*360,344,label,17)
add('B / TARGETS','Người mua hiện tại gồm ba nhóm lịch sử',svg(b),
    'Tháng đầu chưa có lịch sử nhãn trước đó nên thuộc Unknown; các tháng sau phân loại trong cửa sổ quan sát.',
    'Mẫu số mỗi cột: toàn bộ người mua tháng T; ba share cộng 100%. CI Wilson và số đếm nằm trong bảng nguồn.',
    ['buyer_composition_monthly'])
add('B / TARGETS','Target đầy đủ, nhưng hiếm và chỉ quan sát 12 tháng',
    table(['Kết luận','Bằng chứng','Giới hạn'],[
      ['Khung nhãn liên tục','100.224 khách × 12 tháng','Không có nhãn trước 02/2022'],
      ['Nhãn dương hiếm','y_any 1,0688%; target_2 0,0480%','Nhóm nhỏ khó kết luận chắc chắn'],
      ['Có thể mô tả mua lặp lại','11 cặp chuyển tiếp theo tháng','Không suy số lần mua trong một tháng'],
      ['First-time là lần đầu được thấy','Dựa trên lịch sử nhãn trước T','Không phải lịch sử mua trọn đời']],[28,37,35]),
    'Chất lượng nhãn tốt ở khung hiện có; giới hạn chính là độ hiếm và thời gian quan sát.',
    'Không sử dụng trans_count / diff_trans_date như bằng chứng đã có tại cutoff nếu chưa xác minh cửa sổ.',
    ['target_monthly','target_schema','target_transition_comparison'])

# C — Transactions, five slides.
add('C / TRX','TRX có lịch sử dài và độ phủ khách cao',coverage('trx'),
    '98.721 khách có ít nhất một giao dịch trong toàn lịch sử; 1.503 khách không có log TRX.',
    'Hai panel: triệu sự kiện và nghìn khách. TRX không null ID/time; currency có 118 null, giữ nguyên mã thuộc tính thiếu.',
    ['source_monthly','source_overview','customer_activity','raw_missing'])
a=activity[activity.source=='trx'].iloc[0]
add('C / TRX','Một số khách đóng góp lượng giao dịch rất lớn',
    table(['Thước đo / khách','Giá trị','Cách đọc'],[
      ['Median · P75',vn(a['median'])+' · '+vn(a.p75),'Đại diện phần giữa phân phối'],
      ['Mean · P95',vn(a['mean'],1)+' · '+vn(a.p95),'Mean nhạy với khách nhiều sự kiện'],
      ['P99 · Maximum',vn(a.p99)+' · '+vn(a.maximum),'Không tự coi đuôi lớn là lỗi'],
      ['Top 1% · Top 5%',pct(a.top1_share)+' · '+pct(a.top5_share),'Tỷ trọng tổng số giao dịch']],[32,31,37]),
    'Tổng giao dịch không phản ánh đều từng khách; luôn trình bày median và quantile cùng mean.',
    'Toàn lịch sử, mẫu số 100.224 khách, gồm cả count = 0; không loại hoặc cap đuôi.',
    ['customer_activity'])
add('C / TRX','RFM mô tả ba khía cạnh khác nhau của hoạt động',
    panels([mini('trx_recency_days','Recency · ngày'),mini('trx_events','Frequency · T−1'),mini('trx_amount_currency11','Monetary · mã 11')]),
    'Lift so với cùng baseline y_any 1,0688%; khác biệt là liên hệ mô tả trong tập quan sát.',
    'F/M: tháng T−1; R: gần nhất trước cutoff; M: currency 11. Vạch đen: CI 95%; nét đứt: lift = 1; *: n+ < 30.',
    ['rates_cluster_ci','feature_group_definitions'])
cat=read('trx_categories').head(5);catrows=[]
for _,r in cat.iterrows():
    z=rates[(rates.kind=='category')&(rates.metric=='category_'+str(int(r.event_type)))&(rates.target=='y_any')].iloc[0]
    catrows.append([str(int(r.event_type)),pct(r.events/38764930),vn(z.n_total),vn(z.n_positive),vn(z.lift,2)+'× ['+vn(z.lift_ci_low,2)+'; '+vn(z.lift_ci_high,2)+']'])
add('C / TRX','Mã giao dịch phổ biến không đồng nghĩa lift cao',
    table(['event_type','Tỷ trọng event','Khách–tháng có mã','n+','Lift y_any [CI 95%]'],catrows,[15,18,25,12,30]),
    'Popularity dùng số event; lift dùng mỗi khách–tháng có mã đúng một lần, dù phát sinh nhiều giao dịch.',
    'Top 5 mã theo tần suất toàn kỳ, chọn độc lập với nhãn. Nhóm có thể chồng lấn; baseline là toàn bộ khách–tháng.',
    ['trx_categories','rates_cluster_ci'])
def summary_rows(features):
    rows=[]
    for col,label in features:
        for y in [0,1]:
            r=summaries[(summaries.feature==col)&(summaries.target=='y_any')&(summaries.label==y)].iloc[0]
            digits=0 if col.endswith('_events') or col.endswith('_unique_regions') else 3
            rows.append([label,'Dương' if y else 'Âm',vn(r.n_observed),vn(r['median'],digits),'['+vn(r.p25,digits)+'; '+vn(r.p75,digits)+']'])
    return rows
add('C / TRX','Nhịp giao dịch cần đọc trên các khách có đủ cặp sự kiện',
    table(['Chỉ tiêu trong T−1','y_any','n có giá trị','Median','IQR'],summary_rows([('trx_gap_days','Median gap · ngày'),('trx_events','Số giao dịch')]),[31,13,20,16,20]),
    'Gap không xác định khi có dưới hai sự kiện trong tháng; không điền gap bằng 0 cho nhóm này.',
    'Gap giữa hai timestamp liên tiếp cùng khách và cùng tháng; không bắc cầu qua ranh giới tháng.',
    ['feature_target_summary'])

# D — Dialog, four slides.
add('D / DIALOG','DIALOG phủ ít khách hơn TRX và có hoạt động thưa',coverage('dialog'),
    '217.205 tương tác, 46.320 khách từng có DIALOG; thiếu log không cho biết lý do không tương tác.',
    'Độ phủ toàn kỳ khác độ phủ trong từng tháng. DIALOG bắt đầu 10/02/2021 theo timestamp lưu.',
    ['source_monthly','source_overview','customer_activity'])
add('D / DIALOG','Tần suất và Recency là hai góc nhìn về tương tác',
    panels([mini('dialog_events','Số DIALOG · T−1'),mini('dialog_recency_days','Recency · ngày')]),
    'Giữ riêng khách không có DIALOG trong T−1 và khách chưa từng có DIALOG trước cutoff.',
    'Count theo tháng T−1; Recency dùng toàn bộ quá khứ trước T. CI cluster bootstrap; *: n+ < 30.',
    ['rates_cluster_ci','feature_target_summary'])
add('D / DIALOG','Embedding hợp lệ về cấu trúc chưa xác nhận ý nghĩa nội dung',
    table(['Kiểm tra vector','Kết quả','Diễn giải'],[
      ['Tổng vector',vn(embedding['rows']),'Một vector / bản ghi DIALOG'],['Sai dimension / null vector',vn(embedding.wrong_dimension)+' / '+vn(embedding.null_vectors),'Dimension kỳ vọng: 768'],
      ['Phần tử null / không hữu hạn',vn(embedding.bad_elements),'Kiểm tra trên toàn bộ phần tử'],['Zero vector',vn(embedding.zero_vectors),'Không đồng nhất với hội thoại rỗng'],
      ['Norm: min · median · max',vn(embedding.minimum,2)+' · '+vn(embedding['median'],2)+' · '+vn(embedding.maximum,2),'Độ lớn vector số; không phải độ dài hội thoại']],[34,29,37]),
    'Không có raw text hoặc duration để đọc chủ đề, số token hay thời lượng; chưa công bố tín hiệu ngữ nghĩa bổ sung.',
    'Norm L2 = sqrt(sum(x²)); kiểm tra đầy đủ 217.205 vector. Không dùng PCA/UMAP để suy khả năng phân biệt nhãn.',
    ['embedding_quality'])
add('D / DIALOG','DIALOG cho biết mức độ tương tác, còn nội dung bị giới hạn',
    table(['Có thể mô tả','Bằng chứng','Chưa xác minh'],[
      ['Tần suất / Recency','Count T−1; lịch sử trước cutoff','Không có log có phải không liên hệ ngân hàng?'],
      ['Nhịp tương tác','Gap ở tháng có ≥2 bản ghi','Không có thời lượng cuộc hội thoại'],
      ['Biểu diễn nội dung','Vector 768 chiều, kiểm tra hữu hạn','Không gán chủ đề từ dimension hoặc norm'],
      ['Khác biệt theo nhãn','Tỷ lệ, lift, CI trên cùng cửa sổ','Không kết luận nội dung làm thay đổi hành vi']],[29,34,37]),
    'DIALOG bổ sung góc nhìn về tương tác; độ phủ thưa và thiếu văn bản giới hạn diễn giải.',
    'Kết quả mô tả trên dữ liệu có sẵn; các nhóm hiếm được giữ lại và gắn cờ.',
    ['source_overview','embedding_quality','rates_cluster_ci'])

# E — Geography, three slides.
add('E / GEO','GEO nhiều sự kiện nhưng chỉ bắt đầu từ cuối 2021',coverage('geo'),
    '66.295.724 quan sát ở 72.573 khách. Số sự kiện lớn không có nghĩa mọi khách đều có GEO.',
    'Nguồn bắt đầu 11/12/2021. Schema gồm ba mã integer; không có latitude/longitude.',
    ['source_overview','source_monthly','raw_schema'])
add('E / GEO','Mã vùng cho phép mô tả sự đa dạng và tập trung',
    table(['Chỉ tiêu trong T−1','y_any','n có giá trị','Median','IQR'],summary_rows([('geo_unique_regions','Số mã level 6'),('geo_dominant_share','Tỷ trọng mã trội · 0–1'),('geo_entropy','Entropy · nats')]),[31,13,20,16,20],small=True),
    'Số vùng và entropy phụ thuộc số sự kiện được ghi nhận; bảng nguồn có đối chiếu theo mức độ quan sát.',
    'Mỗi integer được xem như category. Entropy = −Σp·ln(p); không suy khoảng cách hoặc số chuyến đi.',
    ['feature_target_summary'])
add('E / GEO','Liên hệ GEO–nhãn cần đọc cùng độ dày quan sát',
    panels([mini('geo_unique_regions','Số mã level 6'),mini('geo_dominant_share','Tỷ trọng trội · 0–1'),mini('geo_transition_rate','Chuyển mã · tỷ lệ 0–1')]),
    'GEO mô tả vùng hoạt động được ghi nhận; chưa xác minh encoding để vẽ bản đồ hoặc tính quãng đường.',
    'Cùng T−1 và baseline y_any. Chuyển mã chỉ tính cặp timestamp có mã rõ ràng, cùng tháng; *: n+ < 30.',
    ['rates_cluster_ci','feature_target_summary'])

# F — Synthesis, four slides.
overlap=read('cross_modal_full_history');names={0:'Không nguồn nào',1:'Chỉ TRX',2:'Chỉ DIALOG',3:'TRX + DIALOG',4:'Chỉ GEO',5:'TRX + GEO',6:'DIALOG + GEO',7:'Cả ba nguồn'}
add('F / TỔNG HỢP','Ba nguồn giao nhau trên 35.424 khách',
    bars([names[int(r['mask'])] for _,r in overlap.iterrows()],list(overlap.clients),fmt=lambda v:vn(v)+' khách'),
    '468 khách không có log ở cả ba nguồn vẫn được giữ trong dân số và trong bảng nhãn.',
    '8 nhóm loại trừ nhau trên toàn lịch sử; không dùng độ phủ tương lai để lọc cohort tại cutoff quá khứ.',
    ['cross_modal_full_history','cross_modal_monthly'])
add('F / TỔNG HỢP','Mỗi bảng cung cấp một góc nhìn khác nhau',
    table(['Nguồn','Thông tin khai thác được','Giới hạn chính'],[
      ['TARGETS','Kết quả sản phẩm, chuyển tiếp, lần đầu quan sát','12 tháng nhãn; không biết số lần mua'],
      ['TRX','Mức độ hoạt động, Recency, amount, cơ cấu mã','Currency và mã ẩn danh; phân phối lệch'],
      ['DIALOG','Tần suất, nhịp tương tác, biểu diễn vector','Độ phủ thưa; không có raw text'],
      ['GEO','Đa dạng, tập trung, thay đổi mã vùng','Encoding chưa rõ; exposure bias']],[18,43,39]),
    '“Bổ sung góc nhìn” được hỗ trợ bởi schema và EDA; chưa phải bằng chứng về giá trị dự báo gia tăng.',
    'Các bảng tổng hợp cùng client_id, cùng cửa sổ T−1 và cutoff.',
    ['raw_schema','feature_target_summary','embedding_quality'])
add('F / TỔNG HỢP','MBD-mini phù hợp nghiên cứu propensity với điều kiện rõ',
    table(['Tiêu chí','Bằng chứng','Đánh giá'],[
      ['Quy mô','100.224 khách; 1.202.688 khách–tháng','Lớn tổng thể; n+ theo nhóm còn hạn chế'],
      ['Lịch sử trước nhãn','TRX dài hơn DIALOG và GEO','Có điều kiện theo nguồn / cutoff'],
      ['Prevalence','0,0480%–0,4458% từng sản phẩm','Nhãn hiếm; y_any = 1,0688%'],
      ['Độ phủ nguồn','98.721 / 46.320 / 72.573 khách','Không đồng đều; giữ khách thiếu nguồn'],
      ['Khả dụng nhãn','Đủ 12 tháng, 0 null ở bốn nhãn','Tốt trong khung; ETL timezone chưa chốt']],[24,40,36]),
    'Có nền tảng dữ liệu để nghiên cứu; kết luận phải giới hạn theo thời gian, nguồn quan sát và cỡ mẫu dương.',
    'Đánh giá mô tả theo đúng 5 tiêu chí; không chấm điểm tổng hợp hoặc suy hiệu quả dự báo.',
    ['source_overview','target_monthly','rates_cluster_ci','temporal_alignment'])
add('F / KẾT LUẬN','Hiểu rõ dữ liệu để đặt câu hỏi có thể trả lời',
    '<div class="conclusion-grid"><article><small>ĐIỂM MẠNH</small><h3>Khung nhãn đầy đủ</h3><p>Khách–tháng liên tục<br>Ba góc nhìn hành vi<br>Khóa nối và lịch sử sự kiện</p></article><article><small>GIỚI HẠN</small><h3>Quan sát có biên</h3><p>Nhãn hiếm, lịch sử hữu hạn<br>Độ phủ nguồn khác nhau<br>Timezone / mã hóa chưa rõ</p></article><article><small>CÓ THỂ KHAI THÁC</small><h3>Hành vi được ghi nhận</h3><p>Mua lặp lại theo tháng<br>Cường độ, nhịp độ, cơ cấu<br>Tập trung và đa dạng mã vùng</p></article></div>',
    'Mọi kết quả trong bài truy ngược được về bảng tổng hợp và nguồn dữ liệu; giới hạn đi cùng phát hiện.',
    'Phụ lục A1–A10: từ điển, công thức, số liệu theo tháng, temporal audit và phương pháp CI.',
    ['target_schema','source_overview','feature_target_summary'])
assert len(slides)==30

# Ten technical appendices, separate numbering.
schema=read('target_schema')
add('A1 / TỪ ĐIỂN','Toàn bộ cột của bảng targets',
    table(['Cột','Kiểu khi đọc','Null','Unique','Vai trò'],[[r['column'],r['dtype'],vn(r.null_count),vn(r.unique_count),
        'Nhãn OR bổ sung' if r['column']=='y_any' else ('Nhãn sản phẩm' if r['column'].startswith('target_') else 'Khóa / metadata / cột phụ')] for _,r in schema.iterrows()],[22,18,16,16,28],small=True),
    'fold đến từ partition; y_any là nhãn tổng hợp được tính trong EDA.',
    'Dtype ở bảng là kiểu sau đọc; raw_schema.csv lưu kiểu vật lý Parquet.',
    ['target_schema','raw_schema'])
add('A2 / NULL','Không có pattern thiếu nhãn trong bản hiện tại',
    table(['Phạm vi','Số lượng','Diễn giải'],[
      ['Đủ bốn nhãn',vn(numrows)+' dòng','100% dòng targets'],['Partial-null / all-null','0 / 0 dòng','Không tạo nhãn âm từ unknown'],
      ['diff_trans_date null','120.786 ô','Cột phụ trợ; không xóa dòng có nhãn hợp lệ'],['Trước 02/2022','Ngoài khung nhãn','Không tự thêm khách–tháng âm']],[33,27,40]),
    'Pattern thực nghiệm không đủ chứng minh nguyên nhân thu thập; giữ mức “chưa xác minh” khi thiếu ETL.',
    'Null rate dùng mẫu số toàn bộ dòng; prevalence chỉ dùng nhãn 0/1 hợp lệ.',
    ['target_null_patterns','target_schema'])
truth=list(itertools.product([0,1,None],repeat=4))
logic=[1 if 1 in t else 0 if all(v==0 for v in t) else None for t in truth]
assert logic.count(1)==65 and logic.count(0)==1 and logic.count(None)==15
add('A3 / LOGIC','Kiểm tra đủ 81 tổ hợp 0 / 1 / null của y_any',
    table(['Quy tắc lý thuyết','Số cấu hình','Ví dụ','Kết quả'],[
      ['Có ít nhất một 1','65','[0, 1, null, 0]','1'],['Bốn giá trị đều 0','1','[0, 0, 0, 0]','0'],
      ['Không có 1, còn null','15','[0, null, 0, 0]','Unknown']],[39,16,29,16]),
    '81 là số tổ hợp logic kiểm tra, không phải số quan sát dữ liệu. Các trường hợp partial-null hiện không xuất hiện.',
    'Dữ liệu thực đã đối soát y_any với union bốn target; miền kỳ vọng {0,1,null}.',
    ['target_schema','target_null_patterns'])
tm=target_month.pivot(index='month',columns='target',values='rate')
add('A4 / CHI TIẾT THÁNG','Prevalence theo từng tháng và từng nhãn',
    table(['Tháng','target_1','target_2','target_3','target_4','y_any'],[[m[:7]]+[pct(r[k],3) for k in ['target_1','target_2','target_3','target_4','y_any']] for m,r in tm.iterrows()],small=True),
    'Mỗi ô dùng 100.224 nhãn hợp lệ; bảng CSV kèm n+, n−, null, Wilson CI và cờ mẫu nhỏ.',
    '12 tháng: 02/2022–01/2023. Không trung bình phần trăm giữa nhóm có mẫu số khác nhau.',
    ['target_monthly'])
trows=[]
for target in ['target_1','target_2','target_3','target_4','y_any']:
    q=trans[trans.target==target];p0=q[q.previous==0];p1=q[q.previous==1]
    z=transition[transition.target==target].iloc[0]
    trows.append([target,vn(p0.n_negative.sum()),vn(p0.n_positive.sum()),vn(p1.n_negative.sum()),vn(p1.n_positive.sum()),
                  vn(z.relative_rate,2)+'× ['+vn(z.ratio_ci_low,2)+'; '+vn(z.ratio_ci_high,2)+']'])
add('A5 / CHUYỂN TIẾP','Ma trận 2×2 gộp 11 cặp tháng liên tiếp',
    table(['Nhãn','0 → 0','0 → 1','1 → 0','1 → 1','Tỷ số rate [CI]'],trows,[16,17,14,14,14,25],small=True),
    'Repeat = (1→1)/(1→0 + 1→1); nonbuyer = (0→1)/(0→0 + 0→1).',
    'Returning xét bất kỳ tháng dương trước T; repeat chỉ xét tháng T−1. Không đồng nhất hai khái niệm.',
    ['target_transitions_monthly','target_transition_comparison'])
curr=read('amount_currency');outlier=read('amount_outliers');c11=outlier[outlier.currency.astype(str).isin(['11','11.0'])].iloc[0]
add('A6 / TRX KỸ THUẬT','Amount giữ riêng currency; ngoại lai không đồng nghĩa lỗi',
    table(['Kiểm tra currency 11','Giá trị','Quy tắc'],[
      ['Số amount hợp lệ',vn(c11.valid_n),'Giữ đơn vị nguồn; không gán tên tiền'],['Log1p-IQR: đuôi thấp',vn(c11.log_below_n),'Tính quartile sau biến đổi'],
      ['Log1p-IQR: đuôi cao',vn(c11.log_above_n),'Đếm riêng hai đuôi'],['Tổng ngoài hàng rào',vn(c11.iqr_n)+' ('+vn(c11.iqr_pct,4)+'%)','Giữ nguyên; không tự xóa / cap'],
      ['Phương pháp đối chiếu','ln(amount) và thang gốc','Không thay kết quả chính log1p']],[35,28,37]),
    'Phần lớn cờ log-IQR currency 11 nằm ở đuôi thấp; không gọi toàn bộ là giao dịch quá lớn.',
    'Kế thừa bảng đính chính ngày 11/11/2026; chart RFM monetary chỉ dùng currency 11.',
    ['amount_outliers','amount_currency'])
add('A7 / DIALOG KỸ THUẬT','Gap và vector là hai loại phép đo khác nhau',
    table(['Chỉ tiêu trong T−1','y_any','n có giá trị','Median','IQR'],summary_rows([('dialog_gap_days','Median gap · ngày'),('dialog_events','Số tương tác')]),[31,13,20,16,20]),
    'Vector có 768 chiều; số chiều không đo độ dài cuộc hội thoại. Gap đo khoảng cách timestamp giữa các tương tác.',
    'Gap chỉ tính cặp cùng tháng. Chưa thực hiện PCA/UMAP hoặc suy chủ đề từ vector.',
    ['feature_target_summary','embedding_quality'])
add('A8 / GEO KỸ THUẬT','Chuyển mã vùng có điều kiện về thứ tự timestamp',
    table(['Hạng mục','Quy tắc / trạng thái'],[
      ['Integer encoding','Chưa xác minh; dùng category, không decode tọa độ'],['Unique / dominant / entropy','Tính theo geohash_6, từng khách trong T−1'],
      ['Timestamp nhiều mã khác nhau','Đánh dấu mơ hồ; loại cặp chạm timestamp này'],['Tỷ lệ chuyển mã','Số cặp đổi mã / số cặp hợp lệ trong cùng tháng'],
      ['Khách có <2 timestamp hợp lệ','Transition rate không xác định; không điền 0'],['Exposure bias','Bảng nguồn phân tầng số event khi đọc đa dạng mã']],[36,64],small=True),
    'Không lấy hiệu hai integer để tính khoảng cách; không gọi chuyển mã là chuyến đi.',
    'Giữ bản ghi raw. Việc gom timestamp chỉ phục vụ xác định thứ tự cặp chuyển mã.',
    ['feature_target_summary','rates_cluster_ci','raw_schema'])
audit=read('temporal_event_audit');aa=audit.pivot(index='outcome_month',columns='source',values='window_events')
add('A9 / TEMPORAL AUDIT','Sự kiện T−1 được đối chiếu trước từng cutoff',
    table(['Tháng T','Cửa sổ tháng T−1','TRX','DIALOG','GEO'],[[m[:7],str(pd.Timestamp(m)-pd.offsets.MonthBegin(1))[:7]]+[vn(r[s]) for s in ['trx','dialog','geo']] for m,r in aa.iterrows()],[17,24,22,17,20],small=True),
    'Mọi max_used_event đều < đầu tháng T; CSV audit lưu cả số sự kiện tích lũy và số bị loại sau cutoff.',
    'Cùng UTC và lịch tháng cho cả ba nguồn. Số đếm 0 nghĩa không có log trong file, không xác nhận không hoạt động.',
    ['temporal_event_audit','temporal_alignment'])
smalln=int(rates.small_positive.sum())
add('A10 / CỠ MẪU VÀ CI','Khoảng tin cậy cần đi cùng mẫu số và coverage',
    table(['Phép đo','Phương pháp / quy tắc'],[
      ['Tỷ lệ từng tháng','Wilson 95%; mỗi khách một quan sát'],['Tỷ lệ gộp nhiều tháng','Cluster bootstrap theo client_id, 2.000 lần, seed 42'],
      ['Lift / tỷ số chuyển tiếp','Tính lại cả tử và mẫu trong cùng lượt bootstrap'],['Nhóm n+ < 30',vn(smalln)+' dòng kết quả gắn cờ; không xếp hạng chắc chắn'],
      ['Mẫu số bằng 0','Không xác định; không tạo tỷ lệ hoặc lift giả'],['Giới hạn của CI','Không sửa bias do thiếu nguồn, encoding hoặc lựa chọn quan sát']],[32,68],small=True),
    'CI mô tả biến thiên khi resample khách trong dataset; không bảo đảm khái quát sang dân số ngân hàng khác.',
    'Bảng rate giữ n, n+, n− và phương pháp. Không có positive: không báo CI bootstrap suy biến tại 0.',
    ['rates_cluster_ci','target_transition_comparison'])
assert len(slides)==40

CSS='''
:root{--font-sans:'Segoe UI',Arial,sans-serif;--font-display:var(--font-sans);--accent-2:#087e8b;--target-1:#5477ad;--target-2:#996723;--target-3:#6652a5;--target-4:#a34e3f;--bg-soft:#f4f7fa;--text-3:#587084}
body{background:var(--bg-soft);font-family:var(--font-sans)}
.tpl-presenter-mode-reveal .slide{padding:34px 56px;justify-content:flex-start;transform:none;transition:opacity .1s;background:var(--bg)}
.slide:before{content:'';position:absolute;left:0;top:0;width:8px;height:100%;background:var(--accent-2)}
.tpl-presenter-mode-reveal .kicker{font-family:var(--font-sans);font-size:13px;font-weight:750;letter-spacing:.12em;color:var(--accent-2);margin:0 0 13px}
h2.title{font-size:35px!important;line-height:1.19;letter-spacing:-.025em;margin:0!important;max-width:1160px;font-weight:720}
.visual{position:absolute;left:56px;right:56px;top:154px;height:360px}
.visual>svg{width:100%;height:100%}svg text{font-family:var(--font-sans);font-variant-numeric:tabular-nums}
.insight{position:absolute;left:56px;right:56px;top:541px;padding:13px 17px;background:var(--surface-2);border-left:4px solid var(--accent-2);font-size:20px;line-height:1.35;font-weight:600;margin:0}
.method{position:absolute;left:56px;right:56px;top:617px;font-size:14px;line-height:1.4;color:var(--text-2);margin:0}
.source{position:absolute;left:56px;right:130px;bottom:22px;font-size:11px;line-height:1.3;color:var(--text-3);margin:0}
.source a{color:inherit}.page{position:absolute;right:56px;bottom:22px;font-size:14px;color:var(--text-3)}
.table-wrap{height:100%;display:flex;align-items:center}table{width:100%;border-collapse:collapse;table-layout:fixed;font-size:19px;line-height:1.35}
th{font-size:14px;text-align:left;padding:9px 12px;border-bottom:2px solid var(--accent-2);color:var(--text-2);font-weight:700}
td{padding:12px;border-bottom:1px solid var(--border);vertical-align:middle;overflow-wrap:anywhere}td:first-child{font-weight:650}tbody tr:nth-child(even){background:var(--bg-soft)}
.small table{font-size:15px;line-height:1.25}.small td{padding:3px 8px}.small th{padding:7px 8px;font-size:13px}
.panels{display:flex;gap:22px;width:100%;height:100%}.panel{flex:1;min-width:0;border-right:1px solid var(--border);padding-right:12px}.panel:last-child{border:0;padding-right:0}.panel svg{width:100%;height:100%}
.hero{display:flex;gap:70px;align-items:center;height:100%}.hero-title{font-size:55px;font-weight:750;line-height:1.13;letter-spacing:-.035em;width:53%;color:var(--accent)}.hero-title span{color:var(--accent-2)}.hero-stats{display:grid;gap:25px}.hero-stats b{display:block;font-size:40px;color:var(--accent)}.hero-stats span{font-size:18px;color:var(--text-2)}
.flow{display:flex;align-items:center;gap:20px;height:100%}.flow>div{flex:1;align-self:stretch;border:1px solid var(--border);border-top:4px solid var(--accent-2);padding:28px 22px;background:var(--bg-soft)}small{font-size:12px;letter-spacing:.08em;font-weight:700;color:var(--accent-2)}h3{font-size:26px;line-height:1.25;margin:19px 0}.flow p,.conclusion-grid p{font-size:20px;line-height:1.75;color:var(--text-2)}.flow-arrow{font-size:30px;color:var(--accent-2)}
.conclusion-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:25px;height:100%}.conclusion-grid article{padding:27px 23px;border-top:4px solid var(--accent-2);background:var(--bg-soft)}
.deck-nav{position:fixed;top:3px;left:50%;transform:translateX(-50%);z-index:50;display:flex;gap:4px;opacity:0;transition:opacity .2s;max-width:96vw;flex-wrap:wrap;justify-content:center}.deck-nav:hover,.deck-nav:focus-within{opacity:1}.deck-nav button{font:12px var(--font-sans);border:1px solid var(--border);background:var(--surface);color:var(--text-1);padding:5px 9px;border-radius:4px;cursor:pointer}.hint-bar{display:none!important}.progress-bar{height:3px!important}
.notes{display:none}
@page{size:1280px 720px;margin:0}
@media print{.deck-nav{display:none!important}.deck{width:1280px!important}.tpl-presenter-mode-reveal .slide{width:1280px!important;height:720px!important;max-height:720px!important;break-after:page}.visual{height:360px}}
'''

# Equivalent deterministic scaffold on Windows: inherit the exact template shell,
# replace all asset links with their inline contents, and assert no unresolved URL.
template=(SKILL/'templates/full-decks/presenter-mode-reveal/index.html').read_text(encoding='utf-8')
assert 'tpl-presenter-mode-reveal' in template and '<aside class="notes">' in template
base=(SKILL/'assets/base.css').read_text(encoding='utf-8')
theme=(SKILL/'assets/themes/corporate-clean.css').read_text(encoding='utf-8')
scoped=(SKILL/'templates/full-decks/presenter-mode-reveal/style.css').read_text(encoding='utf-8')
runtime=(SKILL/'assets/runtime.js').read_text(encoding='utf-8').replace('</script','<\\/script')
parts=[];ledger=[]
for s in slides:
    n=s['number'];number=f'{n:02d} / 30' if n<=30 else f'A{n-30:02d} / 10'
    sources=' · '.join(f'<a href="tables/{esc(k)}.csv">{esc(k)}</a>' for k in s['sources'])
    parts.append(f'<section class="slide{" is-active" if n==1 else ""}" data-title="{esc(s["title"])}" data-section="{esc(s["section"])}">'
      f'<p class="kicker">MBD-MINI / {esc(s["section"])}</p><h2 class="title">{esc(s["title"])}</h2>'
      f'<div class="visual">{s["chart"]}</div><p class="insight">{esc(s["insight"])}</p>'
      f'<p class="method">{esc(s["method"])}</p><p class="source">Nguồn: {sources}</p><span class="page">{number}</span>'
      f'<aside class="notes"><p><strong>{esc(s["title"])}</strong></p><p>{esc(s["notes"])}</p><p>Nguồn: {esc(", ".join(s["sources"]))}</p></aside></section>')
    for name in s['sources']:
        assert name in manifest['tables']
        ledger.append(dict(slide=n,section=s['section'],title=s['title'],chart='inline SVG / HTML table',
            table='tables/'+name+'.csv',method=s['method'],raw_source='data/MBD-mini',input_fingerprint=manifest['fingerprint']))
data={p.stem:pd.read_csv(p).where(pd.notna(pd.read_csv(p)),None).to_json(orient='records',force_ascii=False) for p in T.glob('*.csv')}
# Preserve numeric JSON, replacing pandas NaN with JSON null.
data={k:json.loads(v) for k,v in data.items()}
payload=json.dumps(dict(manifest=manifest,tables=data),ensure_ascii=False,allow_nan=False).replace('</','<\\/')
page='''<!doctype html>
<!-- html-ppt / presenter-mode-reveal / corporate-clean; MIT © 2026 lewis. -->
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="MBD-mini Dataset Understanding & EDA: 30 slide chính, 10 phụ lục, tiếng Việt, mở offline.">
<title>MBD-mini · Dataset Understanding & EDA</title><style>'''+base+'\n'+theme+'\n'+scoped+'\n'+CSS+'''</style></head>
<body class="tpl-presenter-mode-reveal" data-main-count="30"><main class="deck" data-w="1280" data-h="720">'''+''.join(parts)+'''</main>
<nav class="deck-nav" aria-label="Điều hướng"><button onclick="location.hash='#/1'">Đầu</button><button onclick="key('ArrowLeft')">←</button><button onclick="key('ArrowRight')">→</button><button onclick="location.hash='#/31'">Phụ lục</button><button onclick="key('o')">Mục lục (O)</button><button onclick="key('s')">Người trình bày (S)</button><button onclick="key('n')">Ghi chú (N)</button><button onclick="key('f')">Toàn màn hình (F)</button><button onclick="downloadData()">Tải số liệu</button></nav>
<script id="deck-data" type="application/json">'''+payload+'''</script><script>function key(k){document.dispatchEvent(new KeyboardEvent('keydown',{key:k}));}function downloadData(){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([document.getElementById('deck-data').textContent],{type:'application/json'}));a.download='MBD-mini_EDA_aggregate_data.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}</script><script>'''+runtime+'''</script></body></html>'''
assert '<link rel="stylesheet"' not in page and '<script src=' not in page
(OUT/'index.html').write_text(page,encoding='utf-8',newline='\n')
outline=[{k:v for k,v in s.items() if k!='chart'} for s in slides]
(OUT/'outline.json').write_text(json.dumps(outline,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
pd.DataFrame(ledger).to_csv(OUT/'coverage_ledger.csv',index=False,lineterminator='\n')
license_file=SKILL/'LICENSE'
if license_file.exists():shutil.copyfile(license_file,OUT/'LICENSE-html-ppt.txt')
else:(OUT/'LICENSE-html-ppt.txt').write_text('HTML PPT Studio · MIT License · Copyright (c) 2026 lewis\nSee https://github.com/lewislulu/html-ppt-skill\n',encoding='utf-8')
readme='''# MBD-mini — Dataset Understanding & EDA

Mở `index.html` trực tiếp bằng Microsoft Edge hoặc Chrome. Bản HTML gồm **30 slide chính + 10 phụ lục**, nhúng CSS, JavaScript, SVG và toàn bộ số liệu tổng hợp; không cần mạng hoặc máy chủ.

- `←` / `→` / Space / PageUp / PageDown: chuyển slide; Home / End: đầu / cuối.
- `O`: mục lục; `F`: toàn màn hình; `S`: cửa sổ người trình bày; `N`: ghi chú.
- `#/17`: liên kết thẳng slide 17. Nút **Phụ lục** mở A1.
- **Tải số liệu** xuất JSON tổng hợp ngay cả khi chỉ giữ một file HTML.
- Nguồn dưới mỗi slide liên kết CSV trong `tables/`; giữ cả thư mục để mở các CSV này.
- In bằng trình duyệt để xuất các trang. Font Segoe UI / Arial có tiếng Việt, không tải webfont.

## Phạm vi và quy ước

Toàn bộ 100.224 khách và 12 tháng nhãn 02/2022–01/2023. Chi tiết sự kiện lấy từ `detail/*`; không cộng `ptls/*`. Mapping T là tháng chứa mon; cutoff đầu tháng T và cửa sổ `[đầu T−1, đầu T)` theo quy ước nghiên cứu UTC. Timezone ETL nhãn gốc chưa xác nhận. Recency dùng sự kiện gần nhất trước cutoff. Số đếm 0 chỉ nghĩa không tìm thấy log, không chứng minh không hoạt động thực tế.

Wilson CI cho tỷ lệ từng tháng; cluster bootstrap theo client_id với 2.000 lần, seed 42 cho tỷ lệ gộp, lift và tỷ số chuyển tiếp. Lift tính lại baseline trên cùng lượt bootstrap. Nhóm n+ <30 gắn cờ. Dữ liệu chính không được lọc theo việc có đủ ba nguồn.

Returning / first-time observed / history unknown cùng chia cho toàn bộ người mua hiện tại; tháng nhãn đầu thuộc history unknown. Không suy số lần mua thực tế từ số tháng dương. Amount không cộng khác currency; ngoại lai giữ phương pháp log1p-IQR đã đính chính. Không huấn luyện mô hình.

## Tái dựng và đối soát

Chạy từ project root bằng Python `.venv`:

```powershell
.\\.venv\\Scripts\\python.exe -B -u -X utf8 scripts/analyze_dataset_eda.py
.\\.venv\\Scripts\\python.exe -B -X utf8 scripts/build_dataset_eda.py
.\\.venv\\Scripts\\python.exe -B -X utf8 scripts/validate_dataset_eda.py
node scripts/verify_dataset_eda.mjs NODE_MODULES_DIR
```

`manifest.json` lưu phiên bản đầu vào, checksum CSV, quy tắc và kiểm tra tính toán. `coverage_ledger.csv` nối slide với bảng và phương pháp. `outline.json` chứa nội dung và ghi chú. `verification.json` là kết quả kiểm tra 40 slide bằng Edge/Playwright.

Các snapshot cấp khách chỉ lưu trong `outputs/dataset_eda/`, không nằm trong gói bàn giao. Cache quét mới tách khỏi cache cũ; raw được đọc, giữ nguyên.

Nguồn dataset: https://huggingface.co/datasets/ai-lab/MBD-mini (CC BY 4.0); bài báo https://arxiv.org/abs/2409.17587. Thiết kế: HTML PPT Studio, template presenter-mode-reveal, theme corporate-clean (MIT).
'''
(OUT/'README.md').write_text(readme,encoding='utf-8',newline='\n')
print(json.dumps(dict(output=str(OUT/'index.html'),main=30,appendix=10,bytes=len(page.encode('utf-8')),sha256=hashlib.sha256(page.encode()).hexdigest()),ensure_ascii=False))
