"""Build the concise Vietnamese PDF and the exact HTML deck imported into Canva.

Requires reportlab. Charts and text are generated from the published EDA CSVs.
The html-ppt weekly-report template supplies the deck structure and keyboard runtime.
"""
from pathlib import Path
import csv
import html
import json
import math
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/eda"
SLIDES = OUT / "presentation"
SLIDES.mkdir(exist_ok=True)
SKILL = Path.home() / ".codex/skills/html-ppt"


def read(name):
    with (OUT / "tables" / (name + ".csv")).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def vn(x, digits=0):
    return f"{float(x):,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
SOURCE_DATE=datetime.fromisoformat(manifest["created_utc"]).strftime("%d/%m/%Y")
overview = {r["table"]: r for r in read("overview")}
monthly = read("target_monthly")
stats = read("feature_statistics_train")
assoc = read("feature_target_associations")
currency = next(r for r in read("amount_by_currency") if float(r["currency"]) == 11)
outlier = next(r for r in read("amount_outliers_by_currency") if float(r["currency"]) == 11)
assert outlier["analysis_scale"] == "log1p", "Re-execute notebook with log-IQR before publishing"
stat_by_name = {r["feature"]: r for r in stats}
n, positive = sum(float(r["rows"]) for r in monthly), sum(float(r["positive_n"]) for r in monthly)
rate = 100 * positive / n
missing_n = sum(float(r["missing_pct"]) > 0 for r in stats)
constant_n = sum(r["constant_observed"] == "True" for r in stats)
missing_cols = [("dialog_30d_days_since_last_event", "Hội thoại 30 ngày"),
                ("geo_30d_days_since_last_event", "Vị trí 30 ngày"),
                ("trx_30d_days_since_last_event", "Giao dịch 30 ngày")]
top_labels = ["Số subtype trong 30 ngày", "Số loại giao dịch trong 30 ngày", "Số mã đích type12 trong 30 ngày"]
top3 = assoc[:3]
pair_count = len(read("high_correlation_pairs"))


def esc(s):
    return html.escape(str(s), quote=True)


def line_svg():
    w,h,left,bottom = 1100,250,52,204
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Tỷ lệ nhãn dương theo tháng">']
    for v in (0,.5,1,1.5,2):
        y=bottom-v*86
        parts.append(f'<line x1="{left}" y1="{y}" x2="1070" y2="{y}" stroke="#DEE7ED"/><text x="36" y="{y+5}" text-anchor="end" font-size="15" fill="#546575">{vn(v,1)}%</text>')
    points=[]
    for i,r in enumerate(monthly):
        x=left+i*92.0; y=bottom-float(r["positive_pct"])*86
        points.append(f"{x},{y}")
    parts.append('<polyline fill="none" stroke="#0E8A91" stroke-width="4" points="'+" ".join(points)+'"/>')
    for i,r in enumerate(monthly):
        x=left+i*92.0; y=bottom-float(r["positive_pct"])*86
        label=r["cutoff"][5:7]+"/"+r["cutoff"][2:4]
        parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#0E8A91"/><text x="{x}" y="238" text-anchor="middle" font-size="15" fill="#546575">{label}</text>')
    parts.append('</svg>')
    return "".join(parts)


def bar_svg(rows, maxval, suffix="", digits=1, width=1100, color="#0E8A91"):
    parts=[f'<svg viewBox="0 0 {width} {len(rows)*66+30}" role="img" aria-label="Biểu đồ so sánh">']
    for i,(label,value) in enumerate(rows):
        y=20+i*66
        parts.append(f'<text x="0" y="{y+23}" font-size="22" fill="#102B46">{esc(label)}</text>')
        parts.append(f'<rect x="375" y="{y}" width="600" height="30" fill="#EAF0F3"/><rect x="375" y="{y}" width="{600*value/maxval}" height="30" fill="{color}"/>')
        parts.append(f'<text x="1000" y="{y+24}" font-size="23" fill="#102B46">{vn(value,digits)}{suffix}</text>')
    parts.append('</svg>')
    return "".join(parts)


pages=[]


def page(title, content, source, notes, cover=False):
    index=len(pages)+1
    pages.append({"index":index,"title":title,"content":content,"source":source,"notes":notes,"cover":cover})


page("EDA MBD-mini",
     '<p class="date">11/11/2026</p><h1>EDA MBD-mini</h1><p class="cover-lead">Chất lượng dữ liệu và định hướng đặc trưng<br>cho bài toán propensity</p>'
     '<div class="metrics cover-metrics"><div><strong>100.224</strong><span>khách hàng</span></div><div><strong>1.202.688</strong><span>dòng khách-tháng</span></div><div><strong>221</strong><span>đặc trưng được phân tích</span></div></div>',
     f"Dữ liệu MBD-mini. Kết quả EDA cập nhật ngày {SOURCE_DATE}.",
     "Mục tiêu là dự đoán khách mua ít nhất một trong bốn sản phẩm trong tháng đích. Bài trình bày tóm tắt bảy bước EDA, tập trung vào các phát hiện có ý nghĩa cho xử lý dữ liệu và thiết kế đặc trưng. Nguồn: overview.csv, manifest.json và notebook eda.ipynb.",True)
sources=[("Giao dịch",float(overview["trx"]["rows"])),("Vị trí",float(overview["geo"]["rows"])),("Hội thoại",float(overview["dialog"]["rows"]))]
rows="".join(f'<tr><td>{label}</td><td class="number">{vn(count)}</td><td>sự kiện</td></tr>' for label,count in sources)
page("Phạm vi và cấu trúc dữ liệu",'<table class="source-table">'+rows+'</table><p class="takeaway">Phân tích đặc trưng trên <b>701.568 dòng train</b>, từ 02 đến 08/2022</p><p class="muted">Detail là nguồn sự kiện. PTLS là biểu diễn chuỗi của cùng nguồn.<br>Lịch sử tạo đặc trưng luôn trước cutoff.</p>',
     "Nguồn: overview.csv. Không cộng detail và ptls vào cùng tổng log.",
     "Raw được thống kê toàn lịch sử ở cả tám bảng. Chỉ dùng giai đoạn train 02-08/2022 cho liên hệ giữa đặc trưng và nhãn. Overview và tỷ lệ nhãn theo tháng được trình bày đủ 12 tháng. Các mốc cutoff được diễn giải theo quy ước UTC của dự án.")
page("Tỷ lệ mua chỉ "+vn(rate,4)+"%",'<div class="metric-line"><strong>'+vn(positive)+'</strong><span>dòng dương trong '+vn(n)+' dòng khách-tháng</span></div>'+line_svg()+'<p class="takeaway">Ưu tiên AP/PR-AUC và Precision, Recall, Lift theo ngân sách</p><p class="muted">Tỷ lệ mua thay đổi theo tháng. Accuracy đơn thuần không đủ để đánh giá.</p>',
     "Nguồn: target_monthly.csv. Tỷ lệ theo tháng trên toàn cohort.",
     "Nhãn y_any là 1 khi có ít nhất một nhãn sản phẩm dương. Toàn bộ 1.202.688 dòng hiện có nhãn và có 12.854 dòng dương. Tỷ lệ thấp nhất 0,673% và cao nhất 1,694%. Đây là thống kê mô tả của dữ liệu, chưa phải kết quả dự đoán.")
page("Thiếu nguồn cần được giữ đúng nghĩa",f'<div class="metrics compact"><div><strong>{missing_n}/221</strong><span>đặc trưng có giá trị thiếu</span></div><div><strong>{constant_n}/221</strong><span>đặc trưng hằng trong phần quan sát</span></div></div>'+bar_svg([(label,float(stat_by_name[col]["missing_pct"])) for col,label in missing_cols],100,"%",2)+'<p class="takeaway">Giữ khách thiếu nguồn, thêm cờ missing và coverage</p><p class="muted">Biểu đồ: tỷ lệ thiếu của recency theo nguồn trong cửa sổ 30 ngày.</p>',
     "Nguồn: feature_statistics_train.csv. Toàn bộ train 02-08/2022.",
     "Recency chỉ xác định khi có sự kiện nguồn phù hợp trước cutoff. Thiếu recency không đồng nghĩa mất nhãn hay lỗi thu thập đã được xác minh. Trường phụ trợ diff_trans_date thiếu 120.786 dòng (10,043%), tất cả cùng trans_count=0. Imputation nếu cần phải fit trên train.")
page("Không phát hiện dòng trùng hoàn toàn",'<div class="finding"><strong>0</strong><div><h3>Dòng trùng hoàn toàn trong raw</h3><p>Khóa khách-tháng của nhãn cũng không trùng.</p></div></div><div class="finding"><strong>1</strong><div><h3>Cặp giao dịch cùng khách và timestamp</h3><p>Khác thuộc tính, giữ lại để tránh xóa nhầm.</p></div></div><div class="finding"><strong class="smaller">2.020.020</strong><div><h3>Sự kiện thiếu trong PTLS/trx so với detail/trx</h3><p>Dùng detail làm nguồn lịch sử giao dịch đầy đủ.</p></div></div>',
     "Nguồn: duplicates.csv và detail_sequence_reconciliation.csv. Quét toàn bộ raw.",
     "Full-row duplicate được quét chính xác, không dùng hash gần đúng để kết luận. Cùng khách và timestamp chỉ là ứng viên kiểm tra. PTLS/trx có 36.744.910 sự kiện so với 38.764.930 ở detail/trx. Không cộng hai biểu diễn và không tự động loại các giao dịch đồng thời.")
page("Ngoại lai cần tính trên thang log",'<p class="lead">Currency mã 11, '+vn(float(currency["valid_n"]))+' giao dịch</p><div class="metrics outlier-metrics"><div><strong>'+vn(float(outlier["raw_iqr_pct"]),4)+'%</strong><span>IQR thang gốc, chỉ đối chiếu</span><small>'+vn(float(outlier["raw_iqr_n"]))+' dòng</small></div><div><strong>'+vn(float(outlier["iqr_pct"]),4)+'%</strong><span>IQR trên log1p, kết quả chính</span><small>'+vn(float(outlier["iqr_n"]))+' dòng</small></div></div><p class="takeaway">Log-IQR xét cả hai đuôi, không chỉ giao dịch lớn</p><p class="muted">Đuôi thấp '+vn(float(outlier["log_below_n"]))+' dòng, đuôi cao '+vn(float(outlier["log_above_n"]))+' dòng.<br>Đối chiếu ln(amount): '+vn(float(outlier["ln_iqr_pct"]),4)+'%. Giữ raw, không tự động xóa/cap.</p>',
     "Nguồn: amount_outliers_by_currency.csv. Tính log trước quartile, theo từng currency.",
     "Đính chính ngày 11/11/2026 theo rà soát log-IQR ngày 09/11/2026. Biến đổi từng giá trị DOUBLE sang log1p rồi mới tính quartile và hàng rào 1,5*IQR. Với mã 11: "+vn(float(outlier["iqr_n"]))+" dòng ngoài hàng rào, gồm "+vn(float(outlier["log_below_n"]))+" thấp và "+vn(float(outlier["log_above_n"]))+" cao. ln(amount) là phân tích độ nhạy, raw chỉ đối chiếu. Không kết luận ngoại lai cho nhóm dưới 30 giá trị, currency thiếu hoặc IQR log bằng 0. Ngưỡng toàn kỳ chỉ phục vụ EDA, không dùng nguyên cho mô hình quá khứ. Gắn cờ chưa chứng minh lỗi.")
page("Độ đa dạng giao dịch liên hệ với nhãn",bar_svg([(label,float(r["pearson_target"])) for label,r in zip(top_labels,top3)],.10,"",4)+'<p class="takeaway">Liên hệ đơn biến còn thấp, cần kiểm chứng bằng validation</p><p class="muted"><b>'+vn(pair_count)+'</b> cặp đặc trưng có |r| ≥ 0,90 trong mẫu 30.000 dòng train.<br>Rà soát dư thừa giữa cửa sổ thời gian, số đếm và biến biến đổi.</p>',
     "Pearson với target: toàn train. Ma trận nhiều biến: mẫu hash 30.000 dòng, seed 42.",
     "Các biến trong biểu đồ lần lượt là trx_30d_event_subtype_nunique, trx_30d_event_type_nunique và trx_30d_dst_type12_nunique. Tương quan không phải nhân quả hoặc feature importance của mô hình. Ma trận nhiều biến dùng pairwise-complete, cỡ mẫu từng cặp được giữ trong CSV. Nguồn: feature_target_associations.csv, high_correlation_pairs.csv.")
page("Ưu tiên Feature Engineering",'<table class="action-table"><tr><th>Nhóm biến</th><th>Hướng xử lý</th></tr><tr><td>Thiếu nguồn</td><td>Giữ NaN, thêm cờ missing và coverage</td></tr><tr><td>Số đếm, amount</td><td>Log1p/signed-log, tách currency</td></tr><tr><td>Mã phân loại</td><td>Encoding phù hợp, nhóm unknown riêng</td></tr><tr><td>Biến hằng, tương quan cao</td><td>Rà soát trên train, xác nhận qua validation</td></tr></table><p class="muted">Scaling khi thuật toán cần. Fit imputation, encoding và scaler trên train.<br>Loại ID, fold, nhãn và hai trường phụ trợ nhãn khỏi predictor.</p>',
     "Nguồn: feature_engineering_plan.csv. Mọi sự kiện tạo feature phải trước cutoff.",
     "Hai trường phụ trợ nhãn bị loại là trans_count và diff_trans_date do chưa xác minh công thức và cutoff. Không đưa client_id hoặc fold vào X. Đề xuất mới có thể gồm recency, frequency, active-day rate, tỷ trọng loại giao dịch và xu hướng giữa cửa sổ 30/84 ngày. Chỉ chốt biến sau đối chứng theo thời gian.")
page("Bước tiếp theo",'<div class="timeline"><div><b>Train</b><strong>02-08/2022</strong><span>Tạo và rà soát biến</span></div><div><b>Validation</b><strong>09-11/2022</strong><span>So sánh đối chứng</span></div><div><b>Calibration</b><strong>12/2022</strong><span>Hiệu chỉnh xác suất</span></div><div><b>Test khóa</b><strong>01/2023</strong><span>Đánh giá cuối</span></div></div><ol class="next"><li>Chốt schema và quy tắc missing</li><li>Đối chứng đặc trưng theo nguồn và lịch sử</li><li>Đánh giá theo thời gian và ngân sách liên hệ</li></ol><p class="closing">Bảo toàn ý nghĩa thiếu nguồn và ngoại lai trước khi tối ưu mô hình</p>',
     "Nguồn: feature_engineering_plan.csv. Bài trình bày này báo cáo EDA, không báo cáo mô hình.",
     "Thiết kế chia thời gian này dành cho giai đoạn đánh giá tiếp theo. EDA chưa huấn luyện hoặc mở kết quả dự đoán test để chọn biến. Khách có thể xuất hiện nhiều tháng nên khi ước lượng bất định cần tôn trọng nhóm khách. Dữ liệu không cung cấp hiệu quả chiến dịch hay ROI. Toàn bộ bảng và định nghĩa ở deliverables/eda.")

base=(SKILL/"assets/base.css").read_text(encoding="utf-8")
theme=(SKILL/"assets/themes/corporate-clean.css").read_text(encoding="utf-8")
template=(SKILL/"templates/full-decks/weekly-report/style.css").read_text(encoding="utf-8")
runtime=(SKILL/"assets/runtime.js").read_text(encoding="utf-8")
# The runtime includes a closing-script token in a comment. Escape it before
# embedding so the HTML parser cannot terminate the inline script early.
runtime=runtime.replace("</script", "<\\/script")
runtime="\n".join(line.rstrip() for line in runtime.splitlines())+"\n"
custom="""
.tpl-weekly-report{--bg:#fff;--text-1:#102B46;--text-2:#536474;--text-3:#647789;--accent:#102B46;--accent-2:#0E8A91;--warn:#C97529;--font-sans:Arial,'Segoe UI',sans-serif;--font-display:Arial,'Segoe UI',sans-serif;font-family:var(--font-sans)}
.tpl-weekly-report .slide{padding:52px 78px 76px;justify-content:flex-start;background:var(--bg);gap:0}
.slide:before{content:'';width:58px;height:5px;background:var(--accent-2);position:absolute;top:34px;left:78px}
.slide h1{font-size:76px;line-height:1.08;margin:60px 0 22px;letter-spacing:-2px;font-weight:700}
.slide h2{font-size:42px;line-height:1.16;margin:22px 0 28px;font-weight:700;max-width:1100px;letter-spacing:-.9px}
.slide h3{font-size:25px;margin:0 0 7px;font-weight:700}.slide p{font-size:22px;line-height:1.45;margin:0 0 17px}
.slide .date{font-size:18px;margin:24px 0 0;color:var(--text-2)}.slide .cover-lead{font-size:31px;line-height:1.4;color:var(--text-2)}
.metrics{display:flex;gap:54px;margin:20px 0 30px;align-items:start}.metrics>div{flex:1}.metrics strong{display:block;font-size:62px;line-height:1.12;font-weight:700;letter-spacing:-1.5px;color:var(--accent)}.metrics span{display:block;font-size:23px;line-height:1.35;margin-top:12px;color:var(--text-2)}.metrics small{display:block;font-size:21px;margin-top:12px;color:var(--text-2)}
.cover-metrics{margin-top:55px}.cover-metrics strong{font-size:52px}.compact{margin:3px 0 22px}.compact strong{font-size:48px}.compact span{font-size:21px}
.slide .source-table{border-collapse:collapse;width:100%;margin:8px 0 28px}.source-table td{font-size:29px;padding:20px 0;border-bottom:1px solid var(--border)}.source-table td.number{font-size:39px;font-weight:700;text-align:right;padding-right:35px}.source-table td:last-child{font-size:22px;color:var(--text-2);width:140px}
.slide .takeaway{font-size:27px;font-weight:700;line-height:1.35;color:var(--accent);margin-top:22px}.slide .muted{font-size:21px;line-height:1.5;color:var(--text-2)}
.metric-line{display:flex;gap:22px;align-items:center;margin-bottom:16px}.metric-line strong{font-size:50px;letter-spacing:-1px}.metric-line span{font-size:23px;color:var(--text-2);max-width:580px}
.slide svg{width:100%;font-family:var(--font-sans);flex-shrink:0;margin:6px 0 4px}.finding{display:flex;align-items:center;gap:34px;border-bottom:1px solid var(--border);padding:24px 0}.finding>strong{font-size:62px;width:230px;flex:none;color:var(--accent-2)}.finding>strong.smaller{font-size:37px;letter-spacing:-1px}.finding p{font-size:22px;color:var(--text-2);margin:0}.lead{color:var(--text-2)}.outlier-metrics{margin-top:24px;margin-bottom:32px}.outlier-metrics strong{font-size:66px}
.action-table{width:100%;border-collapse:collapse;margin:0 0 25px}.action-table th{text-align:left;font-size:20px;color:var(--text-2);font-weight:400;padding:4px 0 14px;border-bottom:2px solid var(--accent)}.action-table td{font-size:23px;padding:19px 0;border-bottom:1px solid var(--border);line-height:1.3}.action-table td:first-child{width:37%;font-weight:700;padding-right:25px}
.timeline{display:flex;gap:28px;margin:20px 0 32px;padding-bottom:27px;border-bottom:2px solid var(--accent-2)}.timeline>div{flex:1}.timeline b{font-size:22px;color:var(--accent-2);display:block}.timeline strong{display:block;font-size:27px;margin:12px 0}.timeline span{font-size:19px;color:var(--text-2)}.next{margin:5px 0 26px;padding-left:31px}.next li{font-size:27px;padding:10px 0 10px 6px}.slide .closing{font-size:26px;color:var(--accent);font-weight:700;line-height:1.4}
.slide .deck-footer{position:absolute;left:78px;right:78px;bottom:27px;font-size:14px;line-height:1.3;color:var(--text-3);display:flex;justify-content:space-between;gap:30px;align-items:end}.deck-footer .source{max-width:970px}.deck-footer .page{font-size:17px;color:var(--text-2);white-space:nowrap}
"""
sections=[]
for p in pages:
    title="" if p["cover"] else f'<h2>{esc(p["title"])}</h2>'
    sections.append(f'<section class="slide" data-document-role="page" data-label="{esc(p["title"])}" data-title="{esc(p["title"])}" data-speaker-notes="{esc(p["notes"])}">{title}{p["content"]}<div class="deck-footer"><span class="source">{esc(p["source"])}</span><span class="page">{p["index"]:02d} / 09</span></div><aside class="notes">{esc(p["notes"])}</aside></section>')
css=base+"\n"+theme+"\n"+template+"\n"+custom
head='<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EDA MBD-mini - Báo cáo kết quả</title><style>'+css+'</style></head>'
(SLIDES/"index.html").write_text(head+'<body class="tpl-weekly-report"><div class="deck" data-w="1280" data-h="720">'+"\n".join(sections)+'</div><script>'+runtime+'</script></body></html>',encoding="utf-8")
import_css='body{display:block;width:1280px;height:auto;overflow:visible}.slide,.tpl-weekly-report .slide{position:relative;inset:auto;opacity:1;transform:none;pointer-events:auto;width:1280px;height:720px;margin:0;page-break-after:always}aside.notes{display:none}'
(SLIDES/"canva_import.html").write_text(head.replace('</style>','\n'+import_css+'</style>')+'<body class="tpl-weekly-report">'+"\n".join(sections)+'</body></html>',encoding="utf-8")
outline={"title":"EDA MBD-mini","language":"vi","slides":pages,"source_manifest_created_utc":manifest["created_utc"],"design":{"template":"html-ppt/weekly-report","theme":"corporate-clean adapted navy/teal","canvas":[1280,720]},"notes":"Values derived from the executed EDA CSVs. HTML contains editable text and SVG charts."}
(SLIDES/"outline.json").write_text(json.dumps(outline,ensure_ascii=False,indent=2),encoding="utf-8")
brief="# EDA MBD-mini - Báo cáo tóm tắt\n\nĐính chính log-IQR ngày 11/11/2026 theo rà soát 09/11/2026; notebook đã được thực thi lại.\n\n"
for p in pages:
    brief+=f"## {p['index']}. {p['title']}\n\n{p['notes']}\n\n{p['source']}\n\n"
(OUT/"executive_summary.md").write_text(brief.rstrip()+"\n",encoding="utf-8")


def build_pdf():
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import ParagraphStyle
    pdfmetrics.registerFont(TTFont("Arial", "C:/Windows/Fonts/arial.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-Bold", "C:/Windows/Fonts/arialbd.ttf"))
    pdfmetrics.registerFontFamily("Arial",normal="Arial",bold="Arial-Bold",italic="Arial",boldItalic="Arial-Bold")
    W,H=A4; M=43; CW=W-2*M
    c=canvas.Canvas(str(OUT/"executive_report.pdf"),pagesize=A4)
    c.setTitle("EDA MBD-mini - Báo cáo tóm tắt")
    c.setAuthor("Propensity Research")
    navy,teal,muted,light=(HexColor(x) for x in ("#102B46","#0E8A91","#536474","#DFE7EC"))
    def txt(x,y,s,size=10,font="Arial",color=navy):
        c.setFillColor(color);c.setFont(font,size);c.drawString(x,y,str(s))
    def para(s,y,size=10.4,leading=15,color=navy,width=CW):
        p=Paragraph(s,ParagraphStyle("body",fontName="Arial",fontSize=size,leading=leading,textColor=color))
        _,height=p.wrap(width,800);p.drawOn(c,M,y-height)
        return y-height
    def rule(y):
        c.setStrokeColor(light);c.setLineWidth(.7);c.line(M,y,W-M,y)
    def footer(num):
        rule(48);txt(M,31,"EDA MBD-mini. Đính chính log-IQR ngày 11/11/2026.",8,color=muted)
        txt(W-M-19,31,f"{num}/3",8,color=muted)
    def start(title,num):
        c.setFillColor(teal);c.rect(M,H-40,34,3,fill=1,stroke=0)
        txt(M,H-72,title,24,"Arial-Bold")
        txt(M,H-95,"MBD-mini / Nghiên cứu propensity",9,color=muted)
    # Page 1: scope, imbalance and source scale.
    start("EDA MBD-mini",1)
    y=H-135
    for x,value,label in [(M,"100.224","khách hàng"),(M+178,"1.202.688","dòng khách-tháng"),(M+365,"221","đặc trưng")]:
        txt(x,y,value,25,"Arial-Bold");txt(x,y-22,label,10,color=muted)
    rule(y-41)
    txt(M,y-73,"Nhãn đầy đủ, tỷ lệ mua chỉ "+vn(rate,4)+"%",17,"Arial-Bold")
    para("Mục tiêu <b>y_any</b> là mua ít nhất một trong bốn sản phẩm trong tháng đích. "
         "Có <b>12.854</b> dòng dương trong <b>1.202.688</b> dòng nhãn. Tỷ lệ theo tháng từ "
         "<b>0,673%</b> đến <b>1,694%</b>. Cần đánh giá bằng AP/PR-AUC và Precision, Recall, Lift theo ngân sách.",y-91)
    chart_bottom=385;chart_left=M+30;chart_w=CW-44
    for v in (0,.5,1,1.5,2):
        yy=chart_bottom+v*62;c.setStrokeColor(light);c.line(chart_left,yy,chart_left+chart_w,yy)
        txt(M-2,yy-3,vn(v,1)+"%",8,color=muted)
    pts=[]
    for i,r in enumerate(monthly):
        x=chart_left+i*chart_w/11;yy=chart_bottom+float(r["positive_pct"])*62;pts.append((x,yy))
        txt(x-10,chart_bottom-18,r["cutoff"][5:7]+"/"+r["cutoff"][2:4],7,color=muted)
    c.setStrokeColor(teal);c.setLineWidth(2)
    for a,b in zip(pts,pts[1:]):c.line(*a,*b)
    for x,yy in pts:c.setFillColor(teal);c.circle(x,yy,2.5,fill=1,stroke=0)
    txt(M,chart_bottom-39,"Tỷ lệ nhãn dương theo tháng, toàn bộ cohort",8.5,color=muted)
    txt(M,chart_bottom-70,"Quy mô ba nguồn sự kiện",14,"Arial-Bold")
    yy=chart_bottom-96
    for label,count in sources:
        txt(M,yy,label,10.5);txt(M+230,yy,vn(count),11,"Arial-Bold");txt(M+365,yy,"sự kiện",10,color=muted);rule(yy-9);yy-=31
    yy=para("<b>Phạm vi:</b> raw toàn lịch sử ở 8 bảng. Thống kê đặc trưng và liên hệ nhãn chỉ dùng "
            "<b>701.568 dòng train 02-08/2022</b>. Detail là sự kiện, PTLS là chuỗi tương ứng và không cộng hai dạng.",yy-7)
    para(f"Nguồn: overview.csv, target_monthly.csv, manifest.json. Báo cáo kỹ thuật kèm 12 biểu đồ và {len(manifest['tables'])} bảng CSV.",yy-14,8.5,12,muted)
    footer(1);c.showPage()
    # Page 2: missingness, duplicates and tail flags.
    start("Chất lượng dữ liệu và ngoại lai",2)
    yy=H-140
    for x,value,label in [(M,"0","dòng raw trùng hoàn toàn"),(M+184,str(missing_n),"đặc trưng có giá trị thiếu"),(M+369,str(constant_n),"đặc trưng hằng quan sát")]:
        txt(x,yy,value,30,"Arial-Bold");txt(x,yy-20,label,8.1,color=muted)
    txt(M,yy-61,"Tỷ lệ thiếu recency trong cửa sổ 30 ngày",13,"Arial-Bold")
    yy-=91
    for col,label in missing_cols:
        val=float(stat_by_name[col]["missing_pct"])
        txt(M,yy,label,9);c.setFillColor(light);c.rect(M+147,yy-2,269,12,fill=1,stroke=0)
        c.setFillColor(teal);c.rect(M+147,yy-2,269*val/100,12,fill=1,stroke=0)
        txt(M+430,yy,vn(val,2)+"%",9,"Arial-Bold");yy-=31
    yy=para("<b>Giữ khách thiếu nguồn.</b> Thêm cờ missing/coverage và chỉ fit imputation trên train khi thuật toán cần. "
            "<b>diff_trans_date</b> thiếu 120.786 dòng (10,043%), đều có trans_count=0. Đây là thiếu thuộc tính phụ trợ, không phải mất nhãn.",yy-2,10.2,14.6)
    yy-=31;txt(M,yy,"Amount theo currency mã 11",14,"Arial-Bold");yy-=27
    for label,value in [("Số giao dịch",vn(currency["valid_n"])),("Trung vị (đơn vị dữ liệu)",vn(currency["median"],3)),
                        ("IQR gốc (đối chiếu)",vn(outlier["raw_iqr_n"])+" dòng / "+vn(outlier["raw_iqr_pct"],4)+"%"),
                        ("Log1p-IQR (kết quả chính)",vn(outlier["iqr_n"])+" dòng / "+vn(outlier["iqr_pct"],4)+"%")]:
        txt(M,yy,label,10);txt(M+237,yy,value,10.5,"Arial-Bold");rule(yy-9);yy-=30
    yy=para("<b>Tính log trước quartile, xét cả hai đuôi:</b> "+vn(outlier["log_below_n"])+" dòng thấp và "+vn(outlier["log_above_n"])+" dòng cao. "
            "ln(amount) cho "+vn(outlier["ln_iqr_pct"],4)+"% ngoài hàng rào. <b>Giữ raw, không tự động xóa/cap.</b> "
            "Không kết luận cho IQR log=0, nhóm dưới 30 giá trị hoặc currency thiếu. Không gộp amount khác currency.",yy-1,10.2,14.6)
    yy=para("<b>Đối soát biểu diễn:</b> PTLS/trx ít hơn detail/trx <b>2.020.020 sự kiện</b>. Có một cặp giao dịch cùng khách/timestamp "
            "nhưng khác thuộc tính, được giữ lại. Dùng detail làm lịch sử giao dịch đầy đủ.",yy-14,10.2,14.6)
    para("Nguồn: feature_statistics_train.csv, missing_diff_context.csv, duplicates.csv, amount_outliers_by_currency.csv, detail_sequence_reconciliation.csv.",yy-12,8.2,11.5,muted)
    footer(2);c.showPage()
    # Page 3: target associations and modelling decisions.
    start("Đặc trưng và bước tiếp theo",3)
    txt(M,H-137,"Liên hệ mạnh nhất với y_any trong train",14,"Arial-Bold")
    yy=H-170
    for label,r in zip(top_labels,top3):
        value=float(r["pearson_target"])
        txt(M,yy,label,9.5);c.setFillColor(light);c.rect(M+217,yy-2,218,12,fill=1,stroke=0)
        c.setFillColor(teal);c.rect(M+217,yy-2,218*value/.1,12,fill=1,stroke=0)
        txt(M+447,yy,vn(value,4),9,"Arial-Bold");yy-=34
    yy=para("Pearson dùng toàn bộ train. Liên hệ đơn biến còn thấp và không phải feature importance hay nhân quả. "
            "Mẫu 30.000 dòng có <b>259 cặp |r| &gt;= 0,90</b>, cần kiểm tra dư thừa.",yy-1,10.2,14.5)
    yy-=28;txt(M,yy,"Ưu tiên triển khai",14,"Arial-Bold");yy-=23
    actions=[("Thiếu nguồn", "Giữ NaN, thêm cờ missing và coverage."),
             ("Số đếm và amount", "Log1p/signed-log, tách currency; scaling khi cần."),
             ("Mã phân loại", "Encoding phù hợp, có nhóm unknown riêng."),
             ("Biến hằng / tương quan cao", "Rà soát trên train và kiểm chứng qua validation.")]
    for a,b in actions:
        yy=para(f"<b>{a}.</b> {b}",yy,10.3,14.6);yy-=10
    yy=para("<b>Tránh leakage:</b> event_time &lt; cutoff. Loại client_id, fold, nhãn, trans_count và diff_trans_date khỏi predictor. "
            "Fit mọi bước imputation/encoding/scaling trên train.",yy-3,10.2,14.5)
    yy-=29;txt(M,yy,"Lịch đánh giá theo thời gian",14,"Arial-Bold");yy-=29
    for i,(a,b) in enumerate([("Train","02-08/2022"),("Validation","09-11/2022"),("Calibration","12/2022"),("Test khóa","01/2023")]):
        x=M+i*128;txt(x,yy,a,10,"Arial-Bold",teal);txt(x,yy-20,b,10)
    yy=para("<b>Giới hạn:</b> khách lặp qua tháng nên quan sát không độc lập. Một số biểu đồ, Spearman và ma trận nhiều biến dùng "
            "mẫu hash 30.000 dòng, seed 42; tỷ lệ thiếu, Pearson với nhãn và số đếm outlier dùng toàn bộ phạm vi tương ứng. "
            "Báo cáo này không huấn luyện mô hình, không đánh giá test và không suy ra ROI chiến dịch.",yy-51,9.4,13.2)
    para("Nguồn: feature_target_associations.csv, high_correlation_pairs.csv, feature_engineering_plan.csv. "
         "Chi tiết, định nghĩa và bảng đầy đủ: deliverables/eda/report.html và notebooks/eda.ipynb.",yy-14,8.5,11.8,muted)
    footer(3);c.save()


if "--html-only" not in sys.argv:
    build_pdf()
print("Built publication sources (9 slides)" + (" and executive_report.pdf (3 pages)" if "--html-only" not in sys.argv else ""))
