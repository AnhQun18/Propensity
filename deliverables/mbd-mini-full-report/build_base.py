"""Professional Vietnamese report from verified full-history MBD-mini tables."""
from pathlib import Path
import json
import math
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4,landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle

ROOT=Path(__file__).resolve().parents[2]
BUILD=Path(__file__).resolve().parent
OUTPUT=ROOT/'output/pdf/MBD_mini_Bao_cao_phan_tich_toan_lich_su.pdf'
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'docs/mbd_mini_eda/tables/review'
for name,file in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf'),('Arial-Italic','ariali.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial-Italic',boldItalic='Arial-Bold')
W,H=landscape(A4);M=42;CW=W-2*M
NAVY=colors.HexColor('#17384e');TEAL=colors.HexColor('#177d83')
INK=colors.HexColor('#233747');GREY=colors.HexColor('#637482')
LIGHT=colors.HexColor('#edf3f5');LINE=colors.HexColor('#d6e0e5')
PALETTE=[TEAL,colors.HexColor('#c47748'),colors.HexColor('#7b66a8'),colors.HexColor('#326faf')]
C=canvas.Canvas(str(OUTPUT),pagesize=(W,H),pageCompression=1)
C.setTitle('Báo cáo phân tích dữ liệu MBD-mini - Toàn bộ lịch sử')
C.setAuthor('');C.setSubject('Độ phủ, chất lượng dữ liệu, đặc điểm hành vi và thiết kế mẫu EDA')
C.setCreator('MBD-mini research report');C.setKeywords('MBD-mini, EDA, full history, data quality')
PAGES=[];PAGE=0


def read(name):return pd.read_csv(DATA/(name+'.csv'))
def f(value,d=0):
    if value is None or pd.isna(value):return 'null'
    return f'{float(value):,.{d}f}'.replace(',','~').replace('.',',').replace('~','.')
def pct(value,d=2):return f(value,d)+'%'
def text(x,y,value,size=11,font='Arial',color=INK,align='left'):
    C.setFont(font,size);C.setFillColor(color)
    getattr(C,{'left':'drawString','right':'drawRightString','center':'drawCentredString'}[align])(x,y,str(value))
def para(value,y,x=M,width=CW,size=11.3,leading=16,color=INK,bold=False):
    style=ParagraphStyle('p',fontName='Arial-Bold' if bold else 'Arial',fontSize=size,leading=leading,textColor=color)
    p=Paragraph(value,style);_,height=p.wrap(width,1000)
    if y-height<(25 if size<=8 else 42):raise ValueError(f'Text exceeds page {PAGE}: {value[:70]}')
    p.drawOn(C,x,y-height)
    return y-height
def section_title(value,y):
    text(M,y,value,12.2,'Arial-Bold',NAVY)
    return y-20
def table(headers,rows,y,widths=None,font=10.1,row_height=None,align_right=()):
    widths=widths or [CW/len(headers)]*len(headers)
    ps=ParagraphStyle('table',fontName='Arial',fontSize=font,leading=font*1.3,textColor=INK)
    hs=ParagraphStyle('head',parent=ps,fontName='Arial-Bold',textColor=NAVY)
    data=[[Paragraph(escape(str(v)),hs) for v in headers]]+[[Paragraph(escape(str(v)),ps) for v in row] for row in rows]
    tb=Table(data,colWidths=widths,rowHeights=row_height,repeatRows=1)
    commands=[('BACKGROUND',(0,0),(-1,0),LIGHT),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
      ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
      ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
      ('LINEBELOW',(0,0),(-1,0),.8,LINE),('LINEBELOW',(0,1),(-1,-1),.35,LINE)]
    for col in align_right:commands.append(('ALIGN',(col,1),(col,-1),'RIGHT'))
    tb.setStyle(TableStyle(commands));_,height=tb.wrap(CW,H)
    if y-height<48:raise ValueError(f'Table exceeds page {PAGE}: {headers}, bottom={y-height}')
    tb.drawOn(C,M,y-height)
    return y-height
def note(value,y):return para(value,y,size=9.5,leading=13,color=GREY)
def source(value):
    para('Nguồn: '+value,45,size=7.8,leading=9.6,color=GREY)
def page(title,part='PHÂN TÍCH DỮ LIỆU'):
    global PAGE
    if PAGE:C.showPage()
    PAGE+=1;PAGES.append(title)
    C.setFillColor(colors.white);C.rect(0,0,W,H,fill=1,stroke=0)
    text(M,H-27,'MBD-MINI',9,'Arial-Bold',TEAL)
    text(W-M,H-27,part,8.2,'Arial',GREY,'right')
    C.setStrokeColor(LINE);C.setLineWidth(.7);C.line(M,H-37,W-M,H-37)
    para(title,H-53,size=22,leading=27,color=NAVY,bold=True)
    C.setStrokeColor(LINE);C.line(M,24,W-M,24)
    text(M,12,'Báo cáo toàn lịch sử | 06/11/2026',8,'Arial',GREY)
    text(W-M,12,f'{PAGE:02d}',8,'Arial',GREY,'right')
    return H-98


def matrix(headers,values,y,row_labels,maximum):
    widths=[60]+[(CW-60)/len(headers)]*len(headers)
    rows=[['Khách hàng']+headers]+[[label]+[f(v) for v in row] for label,row in zip(row_labels,values)]
    tb=Table(rows,colWidths=widths,rowHeights=[20]+[18.5]*len(values))
    cmds=[('FONTNAME',(0,0),(-1,-1),'Arial'),('FONTNAME',(0,0),(-1,0),'Arial-Bold'),
          ('FONTSIZE',(0,0),(-1,-1),8.6),('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
          ('BACKGROUND',(0,0),(-1,0),LIGHT),('TEXTCOLOR',(0,0),(-1,0),NAVY),
          ('GRID',(0,0),(-1,-1),.4,colors.white)]
    for r,row in enumerate(values,1):
        for col,value in enumerate(row,1):
            ratio=float(value)/maximum if maximum else 0
            bg=colors.Color(.96-.86*ratio,.98-.52*ratio,.96-.59*ratio)
            cmds.extend([('BACKGROUND',(col,r),(col,r),bg),('TEXTCOLOR',(col,r),(col,r),colors.white if ratio>.56 else INK)])
    tb.setStyle(TableStyle(cmds));_,height=tb.wrap(CW,H)
    if y-height<55:raise ValueError('Matrix overflow')
    tb.drawOn(C,M,y-height);return y-height


def line_chart(x,y,w,h,values,labels,title,color=TEAL,percent=False,zoom=False,reference=None,annotate=False):
    text(x,y+h-4,title,11.3,'Arial-Bold',NAVY)
    left=x+58;right=x+w-12;bottom=y+42;top=y+h-27
    a=np.asarray(values,dtype=float);valid=a[np.isfinite(a)]
    lo=0.;hi=float(valid.max())*1.12 if len(valid) else 1.
    if zoom and len(valid):
        pad=max(float(np.ptp(valid))*.18,float(valid.mean())*.05,.0005)
        lo=max(0.,float(valid.min())-pad);hi=float(valid.max())+pad
    if reference is not None:
        lo=min(lo,reference);hi=max(hi,reference*1.05)
    if hi<=lo:hi=lo+1
    for v in np.linspace(lo,hi,4):
        py=bottom+(v-lo)/(hi-lo)*(top-bottom)
        C.setStrokeColor(LINE);C.setLineWidth(.35);C.line(left,py,right,py)
        text(left-7,py-3,pct(v,3) if percent else f(v),8.5,'Arial',GREY,'right')
    if reference is not None:
        py=bottom+(reference-lo)/(hi-lo)*(top-bottom)
        C.setStrokeColor(GREY);C.setDash(3,2);C.line(left,py,right,py);C.setDash()
    C.setStrokeColor(GREY);C.setLineWidth(.5);C.line(left,bottom,right,bottom)
    px=lambda i:left+i/max(1,len(a)-1)*(right-left)
    C.setStrokeColor(color);C.setLineWidth(1)
    path=C.beginPath();active=False
    for i,v in enumerate(a):
        if not np.isfinite(v):active=False;continue
        py=bottom+(v-lo)/(hi-lo)*(top-bottom)
        if active:path.lineTo(px(i),py)
        else:path.moveTo(px(i),py);active=True
    C.drawPath(path)
    for i,label in labels:
        C.saveState();C.translate(px(i),bottom-9);C.rotate(35)
        text(0,-4,label,8.2,'Arial',GREY,'right');C.restoreState()
    if annotate:
        for i,v in enumerate(a):
            py=bottom+(v-lo)/(hi-lo)*(top-bottom)
            C.setFillColor(color);C.circle(px(i),py,2.6,fill=1,stroke=0)
            text(px(i),py+8,str(int(annotate[i])),8.4,'Arial',INK,'center')
    return top,bottom


def bar_chart(x,y,w,h,counts,labels,title,color=TEAL):
    text(x,y+h-4,title,11.3,'Arial-Bold',NAVY)
    left=x+49;right=x+w-10;bottom=y+61;top=y+h-26
    hi=max(counts)*1.16 or 1
    for v in np.linspace(0,hi,4):
        py=bottom+v/hi*(top-bottom)
        C.setStrokeColor(LINE);C.setLineWidth(.4);C.line(left,py,right,py)
        text(left-7,py-3,f(v/1e6,1),8.3,'Arial',GREY,'right')
    step=(right-left)/len(counts)
    for i,(n,label) in enumerate(zip(counts,labels)):
        height=n/hi*(top-bottom);C.setFillColor(color)
        C.rect(left+i*step+2,bottom,step-4,height,fill=1,stroke=0)
        C.saveState();C.translate(left+(i+.5)*step,bottom-8);C.rotate(36)
        text(0,-3,label,7.7,'Arial',GREY,'right');C.restoreState()
    text(x,y+8,'Trục Y: triệu giao dịch. Mỗi cột biểu diễn số giao dịch trong một khoảng giá trị.',8.8,'Arial',GREY)


# 01. Cover
y=page('Báo cáo phân tích dữ liệu MBD-mini','BÁO CÁO NGHIÊN CỨU')
y=para('Độ phủ lịch sử, chất lượng dữ liệu<br/>và đặc điểm hành vi khách hàng',y-24,size=26,leading=35,color=NAVY)
y=para('Phân tích trên toàn bộ dữ liệu giao dịch, vị trí và hội thoại',y-23,size=14,leading=21,color=GREY)
C.setStrokeColor(TEAL);C.setLineWidth(2);C.line(M,y-24,M+120,y-24)
y=para('<b>100.224 khách hàng</b><br/>105.277.859 bản ghi sự kiện<br/>1.202.688 dòng nhãn khách hàng - tháng',y-51,size=14,leading=25)
para('Phạm vi sự kiện: 31/12/2020 - 31/12/2022<br/>Tháng nhãn: 02/2022 - 01/2023<br/>Ngày hoàn thiện: 06/11/2026',y-32,size=11,leading=18,color=GREY)

# 02. Summary
y=page('Tổng quan kết quả phân tích','TÓM TẮT ĐIỀU HÀNH')
for title,body in [
 ('Dữ liệu giao dịch có độ phủ khách hàng cao nhất','98,50% khách có giao dịch, 72,41% có log vị trí và 46,22% có hội thoại trong toàn lịch sử. Chỉ 35,34% có log ở cả ba nguồn. Phân tích tiếp tục giữ 468 khách không có log.'),
 ('Thiếu dữ liệu tập trung ở các thuộc tính phụ trợ','Toàn bộ 120.786 giá trị null của diff_trans_date xuất hiện khi trans_count bằng 0. Các cặp thuộc tính src/dst thường thiếu đồng thời. Chưa có bằng chứng xác nhận nguyên nhân thu thập của các mã bị thiếu.'),
 ('Dạng chuỗi giao dịch không bảo toàn toàn bộ sự kiện','ptls/trx ít hơn detail/trx 2.020.020 bản ghi tại 13 khách. Báo cáo sử dụng detail làm nguồn chuẩn để bảo toàn lịch sử.'),
 ('Phân phối hành vi lệch phải và nhãn dương hiếm','Trung bình toàn kỳ là 1.050,43 sự kiện mỗi khách, trong khi trung vị là 646. Tỷ lệ dương từng nhãn dao động từ 0,0480% đến 0,4458%. Các kết quả này cần được xem xét khi thiết kế mô hình và đánh giá theo thời gian.')]:
    y=section_title(title,y);y=para(body,y,size=11.5,leading=17)-21
source('Các bảng review/behavior_rates, behavior_combinations, null_target_context, sequence_reconciliation, events_per_customer và descriptive.targets/target_prevalence.')

# 03. Raw table inventory
y=page('Cấu trúc và quy mô các bảng nguồn','01  CẤU TRÚC DỮ LIỆU')
schema=read('source_schema')
names={'clients':'Khách hàng','targets':'Khách hàng - tháng','trx':'Sự kiện giao dịch','geo':'Sự kiện vị trí','dialog':'Sự kiện hội thoại',
       'ptls_trx':'Chuỗi giao dịch / khách','ptls_geo':'Chuỗi vị trí / khách','ptls_dialog':'Chuỗi hội thoại / khách'}
rows=[]
for s,p in schema.groupby('source',sort=False):
    rows.append([p.path.iloc[0],names[s],f(p.rows.iloc[0]),'client_id, mon' if s=='targets' else 'client_id'])
y=table(['Bảng','Đơn vị quan sát','Số dòng','Khóa liên kết'],rows,y,[147,219,115,CW-481],font=11)-20
y=para('Các bảng detail lưu từng sự kiện theo dòng. Các bảng ptls tổ chức dữ liệu theo chuỗi của khách hàng. Hai dạng này có quan hệ biểu diễn và không được cộng để xác định tổng số sự kiện.',y)
note('client_id là mã khách hàng ẩn danh. fold xác định nhóm chia dữ liệu theo khách hàng. Không suy diễn tên sản phẩm hoặc ý nghĩa mã giao dịch khi chưa có từ điển nghiệp vụ.',y-15)
source('review/source_schema.csv. Tổng sự kiện chỉ tính từ detail/trx, detail/geo và detail/dialog.')

# 04. Variables
y=page('Các biến trong bảng khách hàng, nhãn và giao dịch','01  CẤU TRÚC DỮ LIỆU')
rows=[['client_split','client_id; fold','Mã khách hàng và nhóm chia dữ liệu'],
 ['targets','client_id; mon; fold','Khóa khách hàng - tháng và phân chia dữ liệu'],
 ['targets','target_1; target_2; target_3; target_4','Bốn nhãn nhị phân. Chưa xác định tên sản phẩm nghiệp vụ'],
 ['targets','trans_count; diff_trans_date','Trường phụ trợ liên quan giao dịch. Chưa xác minh đầy đủ công thức, đơn vị và thời điểm tạo'],
 ['detail/trx','client_id; event_time; fold','Khách hàng, thời điểm sự kiện và phân chia dữ liệu'],
 ['detail/trx','amount','Số thực float32 như lưu trong dữ liệu gốc. Chưa xác nhận đơn vị hoặc phép biến đổi trước khi công bố'],
 ['detail/trx','event_type; event_subtype; currency','Mã loại giao dịch, loại phụ và tiền tệ đã mã hóa'],
 ['detail/trx','src_type11; src_type12; dst_type11; dst_type12','Các cặp thuộc tính phía nguồn và phía đích đã mã hóa'],
 ['detail/trx','src_type21; src_type22; src_type31; src_type32','Các thuộc tính nguồn bổ sung đã mã hóa']]
y=table(['Bảng','Biến','Nội dung và cách diễn giải'],rows,y,[90,248,CW-338],font=10.4)-18
note('Bốn target có thể đồng thời nhận giá trị 1. trans_count và diff_trans_date chưa được mặc định dùng làm đặc trưng dự báo vì cần xác minh thời điểm và phạm vi tính.',y)
source('docs/dict/mbd_mini.md và review/source_schema.csv.')

# 05. Time and other variables
y=page('Dữ liệu vị trí, hội thoại và phạm vi thời gian','01  CẤU TRÚC DỮ LIỆU')
rows=[['detail/geo','client_id; event_time; geohash_4; geohash_5; geohash_6; fold','Sự kiện vị trí với ba cấp mã vị trí'],
 ['detail/dialog','client_id; event_time; embedding; fold','Vector float32 có 768 chiều, không cung cấp văn bản hội thoại gốc'],
 ['ptls/trx; ptls/geo; ptls/dialog','Các biến tương ứng của detail tổ chức thành mảng. client_id và fold vẫn ở cấp khách','event_time ở ptls có kiểu int64. Không tự suy ra đơn vị trước khi chuyển đổi thời gian']]
y=table(['Bảng','Biến','Nội dung'],rows,y,[155,337,CW-492],font=10.5)-24
y=section_title('Thời gian sự kiện đã ghi nhận',y)
ranges=pd.read_csv(ROOT/'docs/mbd_mini_eda/tables/descriptive.temporal/source_time_ranges.csv')
rows=[[{'transactions':'Giao dịch','geo':'Vị trí','dialog':'Hội thoại'}[r.source],str(r.first_event)[:19],str(r.last_event)[:19]] for r in ranges.itertuples()]
y=table(['Nguồn','Thời điểm đầu','Thời điểm cuối'],rows,y,[155,(CW-155)/2,(CW-155)/2],font=11)-17
note('Timestamp trong detail không kèm múi giờ. Báo cáo giữ nguyên lịch thời gian như trong file. Các tháng và ngày biên có thể chỉ có một phần dữ liệu.',y)
source('review/source_schema.csv và descriptive.temporal/source_time_ranges.csv.')

# 06. Null attributes
y=page('Thiếu thuộc tính trong các dòng đã ghi nhận','02  CHẤT LƯỢNG DỮ LIỆU')
y=para('Giá trị null thể hiện thuộc tính chưa được cung cấp trong một dòng đã tồn tại. Trường hợp một khách không có bản ghi trong một tháng được xử lý riêng ở lưới khách hàng - tháng.',y)-17
miss=read('missing_values');miss=miss[miss.missing_n>0]
rows=[[r.source,r.column,f(r.missing_n),pct(r.missing_rate*100,6)] for r in miss.itertuples()]
y=table(['Nguồn','Biến','Số dòng null','Tỷ lệ trên số dòng của nguồn'],rows,y,[98,229,144,CW-471],font=10.2)-15
note('ID, thời gian và Amount của detail đều đầy đủ. Các dòng geo/dialog được ghi nhận không có null ở các cột. Bốn nhãn target đầy đủ trên toàn bộ 1.202.688 dòng.',y)
source('review/missing_values.csv. Mẫu số targets: 1.202.688 dòng. Mẫu số giao dịch: 38.764.930 dòng.')

# 07. Evidence for null
y=page('Bằng chứng về sự xuất hiện của giá trị null','02  CHẤT LƯỢNG DỮ LIỆU')
y=section_title('diff_trans_date gắn với nhóm không có giao dịch',y)
ctx=read('null_target_context')
mapping={'zero':'trans_count = 0','one':'trans_count = 1','two_or_more':'trans_count từ 2 trở lên'}
y=table(['Nhóm','Số dòng','diff_trans_date null'],[[mapping[r.trans_count_group],f(r.rows),f(r.missing_diff)] for r in ctx.itertuples()],y,[CW*.47,CW*.26,CW*.27],font=11)-14
y=para('Toàn bộ 120.786 null xuất hiện khi trans_count bằng 0. Tất cả dòng có trans_count lớn hơn 0 đều có diff_trans_date. Quan hệ này hỗ trợ cách hiểu thiếu do không có giao dịch trong phạm vi đếm, nhưng chưa xác định công thức của diff_trans_date.',y,size=11,leading=16)-19
y=section_title('Các cặp thuộc tính thiếu đồng thời',y)
pairs=read('null_pairs')
y=table(['Cặp biến','Cùng null','Chỉ một biến null'],[[r.pair,f(r.both_null),f(r.only_one_null)] for r in pairs.itertuples()],y,[CW*.47,CW*.26,CW*.27],font=10.5)-13
note('Các cặp có thể chia sẻ cơ chế ghi nhận hoặc quan hệ cấp cha/con. Dữ liệu chưa cho phép xác định là trường không áp dụng hay lỗi thu thập. Báo cáo giữ nguyên null và không tự điền 0.',y)
source('review/null_target_context.csv, null_pairs.csv và null_by_event_type.csv.')

# 08. Duplicate
y=page('Kết quả kiểm tra bản ghi trùng lặp','02  CHẤT LƯỢNG DỮ LIỆU')
y=para('Kiểm tra trên toàn bộ dữ liệu, xuyên các shard và batch. Với mỗi nhóm trùng, số dòng dư được tính sau lần xuất hiện đầu tiên.',y)-15
dups=read('duplicates');rows=[]
for s in ['clients','targets','trx','geo','dialog','ptls_trx','ptls_geo','ptls_dialog']:
    p=dups[dups.source.eq(s)]
    key=p[p.kind.eq('unique_key')];full=p[p.kind.eq('full_row')];candidate=p[p.kind.eq('same_client_time_candidate')]
    rows.append([s,f(p.analyzed_rows.iloc[0]),f(key.duplicate_n.iloc[0]) if len(key) else 'Không có khóa sự kiện',
                 f(full.duplicate_n.iloc[0]) if len(full) else 'Khóa khách duy nhất',f(candidate.duplicate_n.iloc[0]) if len(candidate) else 'Không áp dụng'])
y=table(['Bảng','Dòng kiểm tra','Trùng khóa','Trùng nguyên dòng','Dư theo khách + thời gian'],rows,y,[99,110,150,141,CW-500],font=10.3)-22
y=para('<b>Không phát hiện bản ghi trùng hoàn toàn.</b> Có một cặp giao dịch cùng client_id và event_time nhưng khác nội dung. Cặp này chỉ là ứng viên cần đối chiếu nghiệp vụ và tiếp tục được giữ trong dữ liệu.',y)-14
note('Trùng nguyên dòng so sánh mọi cột và fold, gồm nội dung vector. Trùng khóa dùng client_id cho khách/ptls và client_id + mon cho targets. Không có event_id nghiệp vụ để tự kết luận hai giao dịch khác thuộc tính là một giao dịch.',y)
source('review/duplicates.csv. Các phép kiểm tra đều có phạm vi full.')

# 09. Representation reconciliation
y=page('Dạng chuỗi giao dịch có mức độ đầy đủ khác detail','02  CHẤT LƯỢNG DỮ LIỆU')
rec=read('sequence_reconciliation')
rows=[[r.source,f(r.detail_events),f(r.ptls_events),f(r.customers_shorter_sequence),f(r.max_sequence_events)] for r in rec.itertuples()]
y=table(['Nguồn','Log detail','Phần tử ptls','Khách có chuỗi ngắn hơn','Độ dài chuỗi tối đa'],rows,y,[90,159,159,169,CW-577],font=10.5)-27
y=para('<b>ptls/trx có ít hơn 2.020.020 sự kiện tại 13 khách hàng.</b> Chuỗi dài nhất gồm 10.000 phần tử, trong khi một khách có tới 1.020.773 giao dịch trong detail. Kết quả gợi ý giới hạn độ dài chuỗi, nhưng chưa xác minh quy tắc chọn các sự kiện được giữ.',y,size=12,leading=19)-20
y=para('Tổng số phần tử và số lượng theo khách của ptls/geo và ptls/dialog khớp với detail. Tuy nhiên, sự tương ứng về số lượng không tự xác nhận ý nghĩa hay quy tắc mã hóa của mọi thuộc tính.',y)-20
y=para('<b>Quy tắc của báo cáo:</b> detail là nguồn chuẩn cho tổng hợp hành vi, biểu đồ và lịch sử trong mẫu. Các bảng ptls được giữ làm nguồn đối chiếu và không được cộng thêm vào tổng sự kiện.',y)-16
note('Không phát hiện null phần tử hoặc giá trị không hữu hạn trong embedding detail/ptls. Việc ptls không có null phần tử ở mảng một cấp không chứng minh detail/trx không thiếu thuộc tính.',y)
source('review/sequence_reconciliation.csv, sequence_quality.csv và embedding_element_quality.csv.')

# 10. Behavior rate matrices
y=page('Độ phủ khách hàng theo nguồn dữ liệu','03  ĐỘ PHỦ HÀNH VI')
br=read('behavior_rates')
y=table(['Nguồn','Khách có log','Khách không có log','Tỷ lệ có log'],[[r.source,f(r.with_log),f(r.without_log),pct(r.coverage_percent)] for r in br.itertuples()],y,[140,190,210,CW-540],font=11)-21
y=section_title('Ma trận tổ hợp nguồn trong toàn lịch sử',y)
co=read('behavior_combinations')
y=table(['Giao dịch','Vị trí','Hội thoại','Số khách','Tỷ lệ'],[[int(r.trx),int(r.geo),int(r.dialog),f(r.customers),pct(r.percent)] for r in co.itertuples()],y,[120,120,120,200,CW-560],font=10)-12
note('Mẫu số là 100.224 khách. 1 biểu thị có ít nhất một log, 0 biểu thị không ghi nhận log trong toàn lịch sử. Không cộng số khách giữa các nguồn. Nhóm có đủ ba nguồn gồm 35.424 khách.',y)
source('review/behavior_rates.csv và behavior_combinations.csv.')

# 11. Monthly coverage matrix, native tables
y=page('Độ phủ theo tháng và chính sách null nguồn sớm','03  ĐỘ PHỦ HÀNH VI')
y=para('Giữ mọi sự kiện trước 01/2022. Trong lưới khách hàng - tháng, geo/dialog không có log trước mốc này được đặt null kèm lý do thiếu quan sát. Các ô có log vẫn giữ số đếm gốc.',y,size=11,leading=16)-18
mc=read('monthly_coverage');months=sorted(mc.month.unique())
for part in [months[:13],months[13:]]:
    rows=[]
    for s in ['trx','geo','dialog']:
        vals=mc[mc.source.eq(s)].set_index('month').loc[part].coverage_percent
        rows.append([s]+['null' if pd.isna(v) else pct(v,1) for v in vals])
    y=table(['Nguồn']+[m[5:]+'/'+m[2:4] for m in part],rows,y,[54]+[(CW-54)/len(part)]*len(part),font=8.5)-24
y=para('Tỷ lệ theo tháng bằng số khách có ít nhất một log chia cho 100.224. Khi toàn nguồn chưa có log trong một tháng sớm, bảng hiển thị null. Nếu đã có log ở một phần khách, tỷ lệ thể hiện độ phủ đã quan sát trên toàn dân số.',y,size=11,leading=16)-14
note('Từ 01/2022, số đếm 0 có nghĩa không ghi nhận log. Giá trị này không chứng minh khách thực sự không hoạt động hoặc hệ thống thu thập liên tục. Tháng 12/2020 chỉ có dữ liệu ngày biên.',y)
source('review/monthly_coverage.csv. Nhãn tháng định dạng MM/YY. Mỗi nguồn sử dụng cùng mẫu số khách hàng.')

# 12-16. Purposefully selected exact customer-month counts.
pc=read('purposeful_coverage')
group_descriptions=[
 ('Nhiều nhất','Nhóm khách có tổng số log cao nhất trong toàn lịch sử'),
 ('Gần trung bình','Nhóm khách có tổng số log gần mức trung bình 1.050,43'),
 ('Gần trung vị','Nhóm khách có tổng số log gần trung vị 646'),
 ('Ít nhất có log','Nhóm khách có tổng số log nhỏ nhất trong số khách đã có log'),
 ('Không có log','Nhóm khách không ghi nhận log trong cả ba nguồn')]
for group,title in group_descriptions:
    y=page(title,'03  MA TRẬN THEO NHÓM CÓ CHỦ ĐÍCH')
    p=pc[pc.group.eq(group)];ids=sorted(p.display_id.unique(),key=lambda s:int(s.split('-')[1]))
    pivot=p.pivot(index='display_id',columns='month',values='events').reindex(ids)[months]
    maxv=int(pivot.to_numpy().max());full=p.drop_duplicates('display_id').full_history_events
    y=para(f'Mỗi hàng là một khách, mỗi ô là tổng số log giao dịch + vị trí + hội thoại trong tháng. '
           f'Tổng log toàn kỳ trong nhóm: {f(full.min())} - {f(full.max())}.',y,size=10.7,leading=15)-13
    for part in [months[:13],months[13:]]:
        y=matrix([m[5:]+'/'+m[2:4] for m in part],pivot[part].to_numpy(),y,ids,maxv)-17
    note('Chọn 8 khách thực tế trong mỗi nhóm, không lặp giữa các nhóm. Đây là minh họa có chủ đích, không dùng để ước lượng tỷ lệ dân số. Mỗi nhóm dùng một thang màu tuyến tính riêng, giữ chung thang cho hai bảng trên trang.',y+3)
    source('review/purposeful_coverage.csv. KH là mã hiển thị trong nhóm. 0 là không ghi nhận log của bất kỳ nguồn nào.')

# 17. Descriptive activity counts
y=page('Số sự kiện phát sinh trên mỗi khách hàng','04  ĐẶC ĐIỂM HÀNH VI')
stats=read('events_per_customer');whole=stats[stats.population.eq('all_customers')]
y=para('Thống kê tổng số log của từng khách trong toàn lịch sử, giữ cùng mẫu số 100.224 khách và bao gồm khách không có log.',y)-17
y=table(['Nguồn','Trung bình','Min','P25','Trung vị','P75','P95','Max'],
    [[r.source,f(r.mean,2),f(r.min),f(r.p25),f(r.median),f(r.p75),f(r.p95),f(r.max)] for r in whole.itertuples()],
    y,[98,101,55,65,87,81,83,CW-570],font=10.6)-24
y=section_title('Phân phối trên riêng khách có log',y)
active=stats[stats.population.eq('customers_with_log')]
y=table(['Nguồn','Số khách','Trung bình','Trung vị','Max'],[[r.source,f(r.rows),f(r.mean,2),f(r.median),f(r.max)] for r in active.itertuples()],y,[120,150,160,140,CW-570],font=10.6)-18
note('Tổng mọi nguồn có trung bình 1.050,43 và trung vị 646 sự kiện/khách. Chênh lệch phản ánh phân phối lệch phải. Các nguồn có phạm vi quan sát khác nhau, nên đây không phải tần suất đã chuẩn hóa theo ngày thu thập.',y)
source('review/events_per_customer.csv. any_source là tổng số log của ba nguồn, không gộp sự kiện khác nguồn thành một giao dịch nghiệp vụ.')

# 18. Daily behavior, native vector evidence
y=page('Nhịp hành vi theo ngày trên toàn lịch sử','04  ĐẶC ĐIỂM HÀNH VI')
daily=read('daily_behavior')
for i,s in enumerate(['trx','geo','dialog']):
    p=daily[daily.source.eq(s)].copy();days=pd.to_datetime(p.day)
    label_idx=[j for j,d in enumerate(days) if d.day==1 and d.month in [1,4,7,10]]
    line_chart(M,H-240-i*129,CW,142,p.events.to_numpy(),[(j,days.iloc[j].strftime('%m/%Y')) for j in label_idx],
               {'trx':'Giao dịch: số log/ngày','geo':'Vị trí: số log/ngày','dialog':'Hội thoại: số log/ngày'}[s],PALETTE[i])
source('review/daily_behavior.csv. Ba trục Y tuyến tính riêng. Ngoài mốc đầu/cuối nguồn để null. Các ngày biên có thể chỉ có một phần dữ liệu.')

# 19. Amount original scale
y=page('Phân phối Amount theo giá trị gốc','04  ĐẶC ĐIỂM HÀNH VI')
a=read('amount_full').iloc[0]
y=table(['Trung bình','Trung vị','P95','P99','Min','Max'],[[f(a['mean'],2),f(a['median'],2),f(a.p95,2),f(a.p99,2),'3,02182 x 10^-6',f(a['max'])]],y,[119,119,119,119,125,CW-601],font=10.1)
h=read('amount_raw_bins');labels=['<0','0-1k','1-5k','5-10k','10-25k','25-50k','50-100k','100-250k','250-500k','0,5-1tr','1-5tr','5-10tr','10-100tr','100tr-1tỷ','>=1tỷ']
bar_chart(M,116,CW,284,h.n.tolist(),labels,'Số lượng giao dịch theo bin Amount gốc')
note('Các bin có độ rộng không bằng nhau, chiều cao biểu diễn số lượng và không biểu diễn mật độ. k = nghìn, tr = triệu. Không quy đổi currency và không gắn đơn vị VND.',103)
source('review/amount_raw_bins.csv và amount_full.csv. Tổng số Amount hữu hạn: 38.764.930. Giữ toàn bộ đuôi phân phối, không biến đổi log.')

# 20. Zoom amount
y=page('Phân phối Amount trong khoảng từ 0 đến P99','04  ĐẶC ĐIỂM HÀNH VI')
zoom=read('amount_zoom');included=int(zoom.n.sum());excluded=int(a.finite_n-included)
y=para(f'Hình tập trung vào {f(included)} giao dịch có Amount không vượt P99 = {f(a.p99,2)}. '
       f'{f(excluded)} giao dịch còn lại vẫn nằm trong bảng và biểu đồ toàn miền.',y)-17
labels=[f(v/1e6,1) if i%3==0 else '' for i,v in enumerate(zoom.left)]
bar_chart(M,105,CW,338,zoom.n.tolist(),labels,'30 bin đều theo giá trị Amount gốc; nhãn trục X theo triệu')
note('Cả trục X và Y đều tuyến tính. Zoom là lựa chọn hiển thị, không phải bước loại quan sát. Các thống kê pooled mô tả số lưu trong file và cần đối chiếu theo từng mã currency trước khi diễn giải nghiệp vụ.',92)
source('review/amount_zoom.csv. 99% quan sát đầu không đại diện cho toàn bộ đuôi phải; hình toàn miền nằm ở trang trước.')

# 21-22. Targets, readable panels
rates=read('target_monthly')
for group in [[1,2],[3,4]]:
    y=page('Biến động tỷ lệ nhãn dương: '+', '.join('target_'+str(i) for i in group),'05  PHÂN PHỐI NHÃN')
    y=para('Tỷ lệ từng tháng bằng tổng số nhãn dương chia cho tổng dòng có nhãn của tất cả fold. Mỗi tháng gồm 100.224 khách. Số cạnh điểm là số mẫu dương thực tế.',y,size=10.7,leading=15)
    for k,i in enumerate(group):
        target='target_'+str(i);p=rates[rates.target.eq(target)].sort_values('mon')
        pooled=p.positive_n.sum()/p.labelled_n.sum()*100
        line_chart(M,269-k*190,CW,180,p.positive_percent.to_numpy(),
                   [(j,pd.to_datetime(v).strftime('%m/%Y')) for j,v in enumerate(p.mon)],
                   f'{target} - tỷ lệ toàn kỳ {pct(pooled,4)}',PALETTE[i-1],True,True,pooled,p.positive_n.tolist())
    source('review/target_monthly.csv. Y zoom riêng cho từng target và không nhất thiết bắt đầu từ 0. Đường đứt: tỷ lệ toàn kỳ. Không so biên độ thị giác giữa hai trục.')

# 23. Sample method and linked population
y=page('Thiết kế mẫu theo phân phối hoạt động','06  LẤY MẪU DỮ LIỆU')
y=para('<b>Quy mô mẫu: 1.000 khách hàng không lặp.</b> Phân tầng đồng thời theo fold, tổ hợp nguồn có log và band tổng số log toàn lịch sử. Seed bằng 42.',y,size=12,leading=18)-15
y=para('Phân bổ số khách theo tỷ trọng dân số của từng tầng bằng phương pháp phần nguyên và phần dư lớn nhất (Hamilton). Sau đó lấy ngẫu nhiên không hoàn lại trong từng tầng. Các band dùng nhóm 0 log và các mốc P25, P50, P75, P95, P99 của khách có log.',y,size=11,leading=16)-18
inventory=read('sample_inventory')
y=table(['Bảng trong mẫu','Số dòng','Phạm vi được giữ'],[[r.source,f(r.rows),'Toàn bộ dòng của khách được chọn'] for r in inventory.itertuples()],y,[170,140,CW-310],font=10)-13
note('Giữ đủ 12 tháng nhãn và mọi sự kiện detail của khách đã chọn. Lưới customer_month có đủ 25 tháng, bảo toàn số đếm raw và cờ null nguồn sớm. Các tầng hiếm có thể nhận 0 mẫu và được công khai trong bảng phân bổ.',y)
source('review/sample_inventory.csv, sampling_strata.csv và sampledata/README.md. Các bảng ptls trong mẫu vẫn giữ giới hạn của ptls gốc.')

# 24. Sample diagnostics
y=page('Đối chiếu mẫu với phân phối dân số','06  LẤY MẪU DỮ LIỆU')
sv=read('sampling_validation');rows=[]
for s in ['trx','geo','dialog','any_source']:
    p=sv[sv.source.eq(s)].set_index('metric')
    rows.append([s,f(p.loc['mean','population_value'],2),f(p.loc['mean','sample_value'],2),
                 f(p.loc['median','population_value'],1),f(p.loc['median','sample_value'],1),f(p.loc['KS_distance','sample_value'],4)])
y=table(['Nguồn','Mean dân số','Mean mẫu','Median dân số','Median mẫu','Khoảng cách KS'],rows,y,[112,128,119,130,130,CW-619],font=10.4)-19
y=note('KS là chênh lệch lớn nhất giữa hai hàm phân phối thực nghiệm của tổng số log; giá trị càng gần 0 thì hai phân phối càng gần nhau. Cực trị và mean có thể khác dù phân phối tổng thể tương đối gần.',y)-20
y=section_title('Tỷ lệ nhãn dương trong mẫu',y)
sl=read('sampling_labels')
y=table(['Nhãn','Tỷ lệ dân số','Tỷ lệ mẫu','Số dương trong mẫu'],[[r.target,pct(r.population_percent,4),pct(r.sample_percent,4),f(r.sample_positive_n)] for r in sl.itertuples()],y,[150,190,190,CW-530],font=10.5)-16
note('Mẫu không bảo toàn chính xác cực trị hoặc tỷ lệ nhãn hiếm. Trọng số tầng bằng population_n/sample_n không khôi phục tầng vắng. Đây là mẫu EDA theo toàn lịch sử; khi thiết kế mô hình quá khứ cần lấy mẫu chỉ từ thông tin trước cutoff.',y)
source('review/sampling_validation.csv và sampling_labels.csv. Các kết luận chính của báo cáo sử dụng toàn bộ raw, không suy rộng từ mẫu 1.000 khách.')

# 25. Outliers by customer and day
y=page('Các quan sát cực trị về tần suất hành vi','07  KIỂM TRA GIÁ TRỊ NGOẠI LAI')
y=para('Sử dụng hàng rào Tukey: [Q1 - 1,5 x IQR; Q3 + 1,5 x IQR], với IQR = Q3 - Q1. Cờ thống kê xác định điểm cần đối chiếu, không tự xác nhận lỗi dữ liệu.',y)-18
oc=read('customer_count_outliers')
y=table(['Nguồn','Ngưỡng trên / khách','Khách vượt IQR','Khách trên P99','Max / khách'],[[r.source,f(r.upper_fence,1),f(r.outlier_n),f(r.above_p99_n),f(r.max)] for r in oc.itertuples()],y,[125,190,150,140,CW-605],font=10.5)-22
y=section_title('Cực trị số log theo ngày trong khoảng đã ghi nhận của nguồn',y)
od=read('daily_count_outliers')
y=table(['Nguồn','Ngày trong phạm vi','Ngày vượt IQR','Ngày có nhiều log nhất','Max log / ngày'],[[r.source,f(r.observed_days),f(r.outlier_days),r.max_day,f(r.max_events)] for r in od.itertuples()],y,[110,155,145,201,CW-611],font=10.2)-16
note('Các phân phối theo khách giữ cả khách 0 log. Đối với hội thoại thưa, IQR nhỏ có thể gắn cờ nhiều khách. Báo cáo giữ toàn bộ quan sát và so sánh thêm P99 cùng bối cảnh sự kiện.',y)
source('review/customer_count_outliers.csv và daily_count_outliers.csv. Không tự xóa, winsorize hoặc clip quan sát cực trị.')

# 26. Amount outliers by currency
y=page('Giá trị ngoại lai của Amount theo từng mã currency','07  KIỂM TRA GIÁ TRỊ NGOẠI LAI')
ao=read('amount_outliers_currency')
rows=[]
for r in ao.itertuples():
    rows.append(['null' if pd.isna(r.currency) else str(int(r.currency)),f(r.finite_n),f(r.upper_fence,2),f(r.outlier_n),f(r.max)])
y=table(['Currency','Amount hữu hạn','Hàng rào Tukey trên','Ngoài hàng rào IQR','Max Amount'],rows,y,[72,150,184,163,CW-569],font=9.5)-17
note('Tính các ngưỡng riêng từng currency, gồm nhóm currency null. Không có Amount không dương trong dữ liệu đã quét. Các nhóm chỉ có 2, 4 hoặc vài quan sát cho ngưỡng phân vị không ổn định. Chưa xác minh đơn vị và ý nghĩa mã tiền tệ nên chưa đặt ngưỡng nghiệp vụ.',y)
source('review/amount_outliers_currency.csv. Ngưỡng là công cụ rà soát, không phải tiêu chuẩn loại bỏ giao dịch.')

# 27. Conclusions and evidence
y=page('Kết luận và cơ sở sử dụng dữ liệu','KẾT LUẬN')
for title,body in [
 ('Phạm vi phân tích','Giữ 100.224 khách và toàn bộ lịch sử detail. Các nguồn có thời gian quan sát khác nhau, do đó không giới hạn EDA về phần giao của ba nguồn.'),
 ('Chính sách dữ liệu','Bảo toàn null thuộc tính, đánh dấu khung geo/dialog thiếu quan sát trước 01/2022 và giữ các điểm cực trị. Không tự loại ứng viên giao dịch cùng thời điểm.'),
 ('Hàm ý cho thử nghiệm propensity','Cần kiểm soát cutoff khi tạo đặc trưng và đánh giá riêng từng nhãn theo tháng. Chưa có kết quả mô hình hoặc hiệu quả kinh doanh để kết luận giá trị triển khai.')]:
    y=section_title(title,y);y=para(body,y)-19
y=section_title('Nguồn và tài liệu đối chiếu',y)
for value in [
 'MBD-mini, dữ liệu cục bộ và README đi kèm. Bộ dữ liệu do ai-lab công bố, giấy phép CC BY 4.0.',
 'Báo cáo EDA toàn lịch sử và các bảng review trong repository Propensity, phiên bản 3e4b95d.',
 'Từ điển MBD-mini và danh mục đầy đủ các bảng, biến trong docs/mbd_mini_eda/data_catalog.md.']:
    y=para(value,y,size=10.2,leading=14)-9
text(M,y-3,'Repository và dữ liệu nguồn',10,'Arial-Bold',TEAL)
C.linkURL('https://github.com/AnhQun18/Propensity/tree/3e4b95d',(M,y-7,M+260,y+9),relative=0)
C.linkURL('https://huggingface.co/datasets/ai-lab/MBD-mini',(M,y-31,M+260,y-14),relative=0)
text(M,y-27,'ai-lab/MBD-mini trên Hugging Face',10,'Arial',TEAL)

C.save()
(BUILD/'build_summary.json').write_text(json.dumps({'pdf':str(OUTPUT),'pages':PAGE,'page_titles':PAGES,
    'source_commit':'3e4b95d','language':'vi','format':'A4 landscape','chart_transform':'none',
    'editable_text_and_tables':True,'native_vector_charts':True},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'pdf':str(OUTPUT),'pages':PAGE},ensure_ascii=False))
