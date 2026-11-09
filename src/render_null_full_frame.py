"""PNG tables with explicit, non-interchangeable missing-data denominators."""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/canva_red_notes_20261107'
data=json.loads((OUT/'null_full_frame_evidence.json').read_text(encoding='utf-8'))
WHITE='#ffffff';INK='#173d38';MUTED='#617a71';GREEN='#267969';LINE='#dbe5df';PALE='#edf4ef'
def font(n,bold=False):return ImageFont.truetype('C:/Windows/Fonts/'+('arialbd.ttf' if bold else 'arial.ttf'),n)
def vi(v,dp=0):return f'{v:,.{dp}f}'.replace(',','~').replace('.',',').replace('~','.')
def canvas(title,subtitle):
    im=Image.new('RGB',(1920,1080),WHITE);d=ImageDraw.Draw(im)
    d.text((90,55),'MBD / PROPENSITY LAB',font=font(19),fill=MUTED)
    d.text((90,118),title,font=font(56,True),fill=INK)
    d.text((90,204),subtitle,font=font(27),fill=MUTED)
    return im,d
def cell(d,x,y,value,right=False,bold=False,size=30,color=INK):
    d.text((x,y),str(value),font=font(size,bold),fill=color,anchor='ra' if right else 'la')

im,d=canvas('Null trên toàn bộ ô dữ liệu gốc','Toàn lịch sử · tỷ lệ null = số ô null / (số dòng × số cột gốc)')
xs=[90,600,810,1170,1480,1830]
headers=['Bảng','Tổng dòng','Cột gốc','Tổng ô dữ liệu','Ô null','Tỷ lệ null']
for j,h in enumerate(headers):cell(d,xs[j],286,h,right=j>0,size=25,color=MUTED)
d.line((90,334,1830,334),fill=LINE,width=2)
names={'clients':'Khách hàng','targets':'targets','trx':'Giao dịch','geo':'Vị trí','dialog':'Hội thoại'}
for i,r in enumerate(data['raw_cells']):
    y=365+i*81
    vals=[names[r['source']],vi(r['rows']),r['cols'],vi(r['cells']),vi(r['null_cells']),vi(r['percent'],3)+'%']
    for j,v in enumerate(vals):cell(d,xs[j],y,v,right=j>0,bold=j==4)
    d.line((90,y+59,1830,y+59),fill=LINE,width=1)
total_rows=sum(r['rows'] for r in data['raw_cells']);total_cols=sum(r['cols'] for r in data['raw_cells'])
total_cells=sum(r['cells'] for r in data['raw_cells']);total_null=sum(r['null_cells'] for r in data['raw_cells'])
d.rectangle((70,780,1850,865),fill=PALE)
vals=['Tổng 5 bảng',vi(total_rows),total_cols,vi(total_cells),vi(total_null),vi(total_null/total_cells*100,3)+'%']
for j,v in enumerate(vals):cell(d,xs[j],806,v,right=j>0,bold=True,color=GREEN,size=30)
d.text((90,903),'targets: 10,043% là tỷ lệ dòng có null; 1,255% là tỷ lệ ô null trên toàn bảng.',font=font(29),fill=INK)
d.text((90,956),'Không tính fold từ partition; embedding tính là 1 ô vector. Phạm vi: bản ghi gốc đang tồn tại.',font=font(24),fill=MUTED)
d.text((90,1022),'Nguồn: review/missing_values.csv · 5 bảng gốc · không cộng ptls',font=font(19),fill=MUTED)
im.save(OUT/'null_toan_bo_o_du_lieu.png')

im,d=canvas('Null trên toàn khung khách hàng–tháng','100.224 khách × 25 tháng (12/2020–12/2022) = 2.505.600 ô cho mỗi nguồn')
xs=[90,720,1330,1830]
headers=['Nguồn','Ô có ≥1 log','Ô trống (null khung)','Tỷ lệ null / toàn khung']
for j,h in enumerate(headers):cell(d,xs[j],292,h,right=j>0,size=25,color=MUTED)
d.line((90,342,1830,342),fill=LINE,width=2)
names={'trx':'Giao dịch','geo':'Vị trí','dialog':'Hội thoại'}
for i,r in enumerate(data['monthly_frame']):
    y=390+i*112
    assert r['with_log']+r['no_log']==r['frame']==2505600
    vals=[names[r['source']],vi(r['with_log']),vi(r['no_log']),vi(r['missing_percent'],2)+'%']
    for j,v in enumerate(vals):cell(d,xs[j],y,v,right=j>0,bold=j in [2,3],size=36)
    d.line((90,y+73,1830,y+73),fill=LINE,width=1)
d.rectangle((90,789,96,968),fill=GREEN)
d.text((120,790),'Null khung = không có bản ghi tại nguồn tương ứng trong ô khách–tháng.',font=font(30,True),fill=INK)
d.text((120,848),'Tỷ lệ null = số ô trống / 2.505.600; bao gồm toàn bộ các khoảng trước năm 2022.',font=font(27),fill=INK)
d.text((120,904),'Đây là độ thiếu log trên khung độ phủ, khác với null thuộc tính trong bảng sự kiện.',font=font(26),fill=MUTED)
d.text((90,1022),'Nguồn: review/monthly_coverage.csv · mẫu số cố định 2.505.600 ô/nguồn · 12/2020 chỉ có ngày 31',font=font(19),fill=MUTED)
im.save(OUT/'null_toan_khung_khach_thang.png')
im.save(OUT/'null_khung_khach_thang_da_chot.png')
(OUT/'null_render_summary.json').write_text(json.dumps({'raw_total_cells':total_cells,'raw_null_cells':total_null,'raw_null_percent':total_null/total_cells*100,'customer_month_frame_per_source':2505600},indent=2),encoding='utf-8')
print('PNG tables rendered with explicit denominators.')
