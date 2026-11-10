"""Exact-data HTML presentation, using html-ppt weekly-report/table layouts."""
from pathlib import Path
import csv
import html
import json


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables/baseline-final/presentation'
SKILL=Path.home()/'.codex/skills/html-ppt'


def build():
    root=OUT.parent
    with (root/'summary.csv').open(encoding='utf-8',newline='') as stream:
        summary={r['model']:r for r in csv.DictReader(stream)}
    monthly=json.loads((root/'monthly_metrics.json').read_text(encoding='utf-8'))
    comparisons=json.loads((root/'paired_comparisons.json').read_text(encoding='utf-8'))
    esc=lambda value:html.escape(str(value),quote=True)
    pct=lambda value:f'{100*float(value):.2f}%'.replace('.',',')
    pp=lambda value:f'{100*value:+.2f}'.replace('.',',')
    names={'previous_month_purchase':'Rule mua tháng trước','behavior_original':'Hành vi gốc',
           'behavior':'Hành vi bỏ biến lịch','purchase':'Lịch sử mua','combined':'Kết hợp'}
    def table(headers,rows,cls=''):
        return '<table class="t '+cls+'"><thead><tr>'+''.join('<th>'+esc(c)+'</th>' for c in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>'
    pages=[]
    def page(title,content,notes,cover=False):
        pages.append(dict(index=len(pages)+1,title=title,content=content,notes=notes,cover=cover))
    page('Baseline propensity',
         '<p class="date">MBD-mini · 11/11/2026</p><h1>Baseline<br>propensity</h1><div class="cover-result"><strong>'+pct(summary['combined']['mean_recall_at_1pct'])+'</strong><p>Recall@1%<br>hành vi + lịch sử mua</p></div><p class="lead">Chọn 1.003 khách mỗi tháng<br>Validation cuốn chiếu 09–11/2022</p>',
         'Đây là báo cáo kết quả nghiên cứu offline. Mô hình kết hợp 156 biến đạt Recall@1% trung bình ba tháng 9,74%. Không diễn giải kết quả như hiệu quả chiến dịch. Test 01/2023 chưa được đánh giá.',True)
    page('Một khách hàng, một tháng',
         '<div class="metrics"><div><strong>100.224</strong><span>khách mỗi tháng</span></div><div><strong>1,20 triệu</strong><span>dòng khách–tháng</span></div><div><strong>1,07%</strong><span>nhãn dương</span></div></div>'
         '<div class="flow"><div>Trx<br>Geo<br>Dialog</div><div>Bảng biến<br>trước cutoff</div><div>GBDT<br>xác suất mua tháng M</div></div>'
         '<p class="takeaway">y_any = mua ít nhất một trong bốn sản phẩm có nhãn</p><p class="muted">Giữ khách thiếu nguồn. Count 0 khác thống kê chưa xác định.</p>',
         'Có 100.224 khách qua 12 tháng, 1.202.688 dòng khách-tháng và 12.854 dòng dương. Sự kiện trước đầu tháng M tạo feature, nhãn mô tả mua trong tháng M. Không đưa client_id hoặc fold vào predictor. Nhãn không đủ căn cứ xác định được giữ unknown.')
    page('Đánh giá bằng ba tháng tương lai',
         table(['Fold','Nhãn train','Validation','Dòng train'],[['1','02–08/2022','09/2022','701.568'],['2','02–09/2022','10/2022','801.792'],['3','02–10/2022','11/2022','902.016']])+
         '<p class="takeaway">Cùng cohort, cùng ngân sách 1.003 khách/tháng</p><p class="muted">UTC và độ trễ nhãn 0 là giả định nghiên cứu.<br>Calibration 12/2022 và test 01/2023 không mở trong ablation mới.</p>',
         'Khách được phép lặp qua tháng train và validation vì câu hỏi là khách hiện hữu ở tháng tương lai. Nhãn lịch sử chỉ dùng khi đã khả dụng tại cutoff. Năm fold theo client là thí nghiệm riêng cho khách chưa thấy. Trainer đã kiểm tra protocol từ preparation config.')
    page('Bốn bộ biến, cùng một cấu hình',
         table(['Biến','Đầu vào','Vai trò'],[['153','Hành vi gốc','Tái lập mốc đã công bố'],['141','Bỏ 12 biến độ phủ nguồn theo lịch','Kiểm tra biến đại diện thời gian'],['15','Lịch sử mua đã khả dụng','Đối chứng mạnh hơn rule một lag'],['156','141 hành vi + 15 lịch sử mua','Đo giá trị tăng thêm']])+
         '<p class="takeaway">12 lần fit CatBoost trên dữ liệu thật</p><p class="muted">Depth 4 · 200 cây · learning rate 0,04 · L2 5<br>Không class weight, resampling hoặc early stopping</p>',
         'Các bộ biến giữ cùng cấu hình seed 20261110 và CPU 2 threads. Hành vi gồm trx, geo và dialog metadata trong tháng gần nhất và lịch sử cũ. Lịch sử mua có lag y_any, target từng sản phẩm, tần suất và recency. Độ dài lịch sử có nhãn cũng có thể gợi thời gian. Tái lập baseline gốc khớp metric cả ba tháng.')
    bars=[]
    for name in names:
        value=float(summary[name]['mean_recall_at_1pct'])*100
        bars.append(f'<div class="bar-row"><span>{names[name]}</span><div class="bar-track"><div class="bar-fill {"best" if name=="combined" else ""}" style="width:{value/12*100:.6f}%"></div></div><b>{pct(summary[name]["mean_recall_at_1pct"])}</b></div>')
    page('Kết hợp đạt Recall@1% '+pct(summary['combined']['mean_recall_at_1pct']),
         '<p class="muted">Trung bình ba tháng · Cùng ngân sách 1.003 khách/tháng</p><div class="bars">'+''.join(bars)+'<div class="axis"><span>0%</span><span>12%</span></div></div>',
         'Các giá trị lần lượt là rule 6,18%, hành vi gốc 8,70%, hành vi bỏ biến lịch 8,89%, lịch sử mua 6,44% và kết hợp 9,74%. Đây là tỷ lệ tìm được trong tổng người mua, không phải precision, accuracy hoặc tỷ lệ tăng mua do chiến dịch.')
    delta=next(c for c in comparisons if c['variant']=='combined' and c['reference']=='purchase')
    page('Giá trị tăng thêm của hành vi',
         '<p class="delta"><strong>'+pp(delta['estimate'])+'</strong> điểm phần trăm so với lịch sử mua</p>'+
         table(['So sánh Recall@1%','Chênh lệch','Khoảng bootstrap 95%'],
               [[names[c['variant']]+' − '+names[c['reference']],pp(c['estimate']),
                 '['+pp(c['interval_95'][0])+'; '+pp(c['interval_95'][1])+']'] for c in comparisons],'intervals')+
         '<p class="muted">Đơn vị: điểm phần trăm. 500 lần bootstrap theo khách.<br>Khoảng có điều kiện theo model và ba tháng đã quan sát.</p>',
         'Kết hợp so với lịch sử mua có chênh 3,2986 điểm phần trăm, khoảng 2,1207 đến 4,2780. So với hành vi 141 biến, chênh 0,8507 điểm, khoảng 0,0230 đến 1,6545. Hai khoảng khác còn chứa 0. Validation đã được dùng phát triển nên đây là so sánh khám phá, không phải xác nhận độc lập hay hiệu quả can thiệp.')
    months=sorted({r['month'] for r in monthly})
    page('Kết quả theo từng tháng',
         table(['Tháng','Rule','Hành vi 141','Lịch sử mua','Kết hợp'],
               [[m,*[pct(next(r['top_k']['0.01']['recall'] for r in monthly if r['model']==n and r['month']==m))
                      for n in ['previous_month_purchase','behavior','purchase','combined']]] for m in months],'monthly')+
         '<p class="takeaway">Kết hợp dẫn đầu Recall@1% trong cả ba tháng</p><p class="muted">AP kết hợp: 0,0455 · ROC-AUC: 0,7924<br>Chỉ số xác suất dùng model từng fold, trước calibration.</p>',
         'Recall kết hợp theo tháng 09, 10 và 11 là 8,5884%, 9,6774% và 10,9573%. Không gộp thứ hạng giữa các tháng. AP và ROC-AUC là metric bổ sung trung bình đều ba tháng. Nhãn được xác định trên cùng 100.224 khách mỗi tháng.')
    page('Bước tiếp theo và giới hạn',
         '<div class="next-row"><strong>01</strong><p>Ablation từng nguồn<br>trên cùng nền lịch sử mua</p></div>'
         '<div class="next-row"><strong>02</strong><p>Cửa sổ 7/14/30/90 ngày<br>và dialog PCA</p></div>'
         '<div class="next-row"><strong>03</strong><p>Khóa mô hình, kiểm tra calibration<br>trước test độc lập</p></div>'
         '<p class="muted">Các nâng cấp này chưa chạy. CoLES là thí nghiệm sau.<br>Chưa đánh giá test độc lập hoặc tác động tăng mua do chiến dịch.</p>'
         '<p class="sources">Nguồn: github.com/AnhQun18/Propensity · arxiv.org/abs/2409.17587</p>',
         'Giữ hướng bảng phẳng và GBDT. Số chiều PCA hay tổng số biến được chốt bằng validation thay vì quy tắc theo số dòng dương. PCA, vocabulary và encoder chỉ fit trên phần train và thời gian được phép. Các cửa sổ mới, dialog PCA và CoLES chưa được triển khai trong lần báo cáo này. Không chọn lại mô hình từ test.')
    base=(SKILL/'assets/base.css').read_text(encoding='utf-8')
    theme=(SKILL/'assets/themes/corporate-clean.css').read_text(encoding='utf-8')
    template=(SKILL/'templates/full-decks/weekly-report/style.css').read_text(encoding='utf-8')
    custom='''
    .tpl-weekly-report{--bg:#fff;--text-1:#142c48;--text-2:#536779;--text-3:#647889;--accent:#142c48;--accent-2:#087f8c;--accent-3:#8093a8;--font-sans:'Segoe UI',Arial,sans-serif;--font-display:'Segoe UI',Arial,sans-serif;font-family:var(--font-sans)}
    .tpl-weekly-report .slide{padding:50px 78px 78px;background:var(--bg);justify-content:flex-start;gap:0}
    .slide:before{content:'';position:absolute;left:78px;top:28px;width:58px;height:5px;background:var(--accent-2)}
    .slide h1{font-size:84px;line-height:1.03;letter-spacing:-2px;margin:26px 0 20px;font-weight:700}
    .slide h2{font-size:42px;line-height:1.16;letter-spacing:-.8px;margin:15px 0 28px;font-weight:700;color:var(--accent)}
    .slide p{font-size:25px;line-height:1.4;margin:0 0 20px}.slide .date{font-size:20px;color:var(--text-2);margin:8px 0 0}
    .slide .lead{font-size:27px;line-height:1.4;margin-top:24px;color:var(--text-2)}
    .cover-result{display:flex;align-items:center;gap:32px;margin-top:0}.cover-result strong{font-size:96px;line-height:1.05;color:var(--accent-2);letter-spacing:-2px}.cover-result p{font-size:28px;margin:0}
    .metrics{display:flex;gap:40px;margin:15px 0 36px}.metrics>div{flex:1}.metrics strong{display:block;font-size:57px;letter-spacing:-1px;color:var(--accent-2);line-height:1.2}.metrics span{display:block;font-size:23px;margin-top:8px;color:var(--text-2)}
    .flow{display:flex;margin:12px 0 33px;border-top:2px solid var(--border);border-bottom:2px solid var(--border);padding:24px 0}.flow>div{flex:1;font-size:27px;font-weight:600;line-height:1.4;border-right:1px solid var(--border);padding-left:28px}.flow>div:first-child{padding-left:0}.flow>div:last-child{border:0}
    .t{width:100%;border-collapse:collapse;margin:5px 0 24px;table-layout:fixed}.t th,.t td{text-align:left;padding:16px 16px 16px 0;border-bottom:1px solid var(--border);font-size:24px;line-height:1.3;vertical-align:middle}.t th{font-size:20px;color:var(--text-2);font-weight:500;border-bottom:2px solid var(--accent)}.t td:first-child{font-weight:600}.t th:first-child,.t td:first-child{width:19%}
    .intervals th:first-child,.intervals td:first-child{width:49%}.intervals th:nth-child(2),.intervals td:nth-child(2){width:18%}.intervals td{font-size:22px;padding:15px 16px 15px 0}.intervals{margin:0 0 21px}
    .monthly td{font-size:29px;padding:27px 16px 27px 0}.monthly th{font-size:21px}.monthly td:last-child{color:var(--accent-2);font-weight:700}
    .slide .takeaway{font-size:28px;line-height:1.35;font-weight:700;color:var(--accent);margin:20px 0 16px}.slide .muted{font-size:22px;line-height:1.45;color:var(--text-2);margin:10px 0 15px}
    .bars{margin:24px 0 28px}.bar-row{display:grid;grid-template-columns:335px 1fr 100px;align-items:center;gap:20px;height:65px}.bar-row>span{font-size:24px}.bar-row>b{font-size:28px;text-align:right}.bar-track{height:36px;border-left:1px solid var(--text-2);background:var(--surface-2)}.bar-fill{height:100%;background:var(--accent-3)}.bar-fill.best{background:var(--accent-2)}.axis{display:flex;justify-content:space-between;font-size:18px;color:var(--text-2);margin:10px 120px 0 355px}
    .slide .delta{font-size:24px;margin:0 0 20px}.delta strong{font-size:56px;line-height:1;color:var(--accent-2);margin-right:12px}
    .next-row{display:flex;gap:32px;border-bottom:1px solid var(--border);padding:14px 0 20px;margin-bottom:9px}.next-row strong{font-size:47px;line-height:1.2;color:var(--accent-2);min-width:80px}.next-row p{font-size:28px;line-height:1.3;margin:0}.slide .sources{font-size:17px;line-height:1.3;color:var(--text-3);margin-top:18px}
    .slide .deck-footer{position:absolute;left:78px;right:78px;bottom:24px;display:flex;justify-content:space-between;color:var(--text-3);font-size:16px;line-height:1.2}.notes{display:none}
    '''
    css=base+'\n'+theme+'\n'+template+'\n'+custom
    def sections(static):
        values=[]
        for page in pages:
            attrs=f' data-document-role="page" data-speaker-notes="{esc(page["notes"])}"' if static else ''
            title='' if page['cover'] else '<h2>'+esc(page['title'])+'</h2>'
            values.append(f'<section class="slide" data-title="{esc(page["title"])}" data-label="{esc(page["title"])}"{attrs}>{title}{page["content"]}<div class="deck-footer"><span>Validation 09–11/2022</span><span>{page["index"]:02d} / 08</span></div><aside class="notes">{esc(page["notes"])}</aside></section>')
        return '\n'.join(values)
    head='<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Baseline propensity · 11/11/2026</title><style>'+css+'</style></head>'
    OUT.mkdir(parents=True,exist_ok=True)
    runtime=(SKILL/'assets/runtime.js').read_text(encoding='utf-8').replace('</script','<\\/script')
    runtime='\n'.join(line.rstrip() for line in runtime.splitlines())+'\n'
    (OUT/'index.html').write_text(head+'<body class="tpl-weekly-report"><div class="deck" data-w="1280" data-h="720">'+sections(False)+'</div><script>'+runtime+'</script></body></html>',encoding='utf-8',newline='\n')
    static='body{display:block;width:1280px;height:auto;overflow:visible}.slide,.tpl-weekly-report .slide{position:relative;inset:auto;opacity:1;transform:none;pointer-events:auto;width:1280px;height:720px;margin:0;page-break-after:always}'
    (OUT/'canva_import.html').write_text(head.replace('</style>',static+'</style>')+'<body class="tpl-weekly-report">'+sections(True)+'</body></html>',encoding='utf-8',newline='\n')
    (OUT/'outline.json').write_text(json.dumps(dict(title='Baseline propensity',date='2026-11-11',language='vi',canvas=[1280,720],
                                                  template='html-ppt weekly-report, cover, table, stat-highlight',slides=pages),ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'slides':len(pages),'output':str(OUT)},ensure_ascii=False))


if __name__=='__main__':
    build()
