"""Create a concise three-page baseline PDF from published aggregate evidence."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                               Image, PageBreak)


NAMES = {'previous_month_purchase':'Rule tháng trước', 'behavior_original':'Hành vi gốc',
         'behavior':'Hành vi bỏ biến lịch', 'purchase':'Lịch sử mua', 'combined':'Kết hợp'}
ORDER = list(NAMES)


def build(root):
    load = lambda name: json.loads((root/name).read_text(encoding='utf-8'))
    # CSV has only aggregate values and no client identifiers.
    import csv
    with (root/'summary.csv').open(encoding='utf-8',newline='') as stream:
        summary = list(csv.DictReader(stream))
    by = {r['model']:r for r in summary}
    monthly, comparisons = load('monthly_metrics.json'), load('paired_comparisons.json')
    pdfmetrics.registerFont(TTFont('Segoe','C:/Windows/Fonts/segoeui.ttf'))
    pdfmetrics.registerFont(TTFont('SegoeBold','C:/Windows/Fonts/segoeuib.ttf'))
    pdfmetrics.registerFontFamily('Segoe',normal='Segoe',bold='SegoeBold',italic='Segoe',boldItalic='SegoeBold')
    navy, teal, muted = colors.HexColor('#142c48'), colors.HexColor('#087f8c'), colors.HexColor('#536779')
    styles = {
        'title':ParagraphStyle('title',fontName='SegoeBold',fontSize=27,leading=33,textColor=navy,spaceAfter=14),
        'h1':ParagraphStyle('h1',fontName='SegoeBold',fontSize=17,leading=22,textColor=navy,spaceBefore=8,spaceAfter=11),
        'body':ParagraphStyle('body',fontName='Segoe',fontSize=10.5,leading=15.5,textColor=navy,spaceAfter=9),
        'small':ParagraphStyle('small',fontName='Segoe',fontSize=8.7,leading=12.3,textColor=muted,spaceAfter=7),
        'cell':ParagraphStyle('cell',fontName='Segoe',fontSize=8.4,leading=11.5,textColor=navy),
        'head':ParagraphStyle('head',fontName='SegoeBold',fontSize=8.2,leading=11,textColor=colors.white),
    }
    P = lambda text, kind='body': Paragraph(text,styles[kind])
    pct = lambda value:f'{100*float(value):.2f}%'.replace('.',',')
    pp = lambda value:f'{100*value:+.2f}'.replace('.',',')
    width = A4[0]-86
    def T(headers,rows,widths):
        data = [[P(escape(str(c)),'head') for c in headers],
                *[[P(escape(str(c)),'cell') for c in row] for row in rows]]
        table = Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('VALIGN',(0,0),(-1,-1),'TOP'),
                                   ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
                                   ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
                                   ('LINEBELOW',(0,1),(-1,-1),.4,colors.HexColor('#dce5eb')),
                                   ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f2f7f8')])]))
        return table
    story = [P('Baseline propensity<br/>Kết quả và bước tiếp theo','title'),
             P('MBD-mini · Báo cáo nghiên cứu offline · 11/11/2026','small'),
             P('Mỗi tháng, chọn <b>1.003 trong 100.224 khách</b> để tìm người sẽ mua ít nhất một trong bốn sản phẩm có nhãn. Đánh giá cuốn chiếu trên 09–11/2022.'),
             P('Kết quả trên cùng ngân sách','h1'),
             Image(str(root/'recall_comparison.png'),width=width,height=width*4.5/11.5),Spacer(1,9),
             T(['Đầu vào','Biến','Recall@1%','AP','ROC-AUC'],
               [[NAMES[n],by[n]['predictors'],pct(by[n]['mean_recall_at_1pct']),f"{float(by[n]['mean_ap']):.4f}",f"{float(by[n]['mean_roc_auc']):.4f}"] for n in ORDER],
               [width*.39,width*.10,width*.19,width*.15,width*.17]),Spacer(1,13),
             P(f"Kết hợp hành vi và lịch sử mua đạt <b>{pct(by['combined']['mean_recall_at_1pct'])}</b>, so với <b>{pct(by['purchase']['mean_recall_at_1pct'])}</b> khi chỉ dùng lịch sử mua. Các số liệu là trung bình đều ba tháng validation."),
             P('<b>Giới hạn:</b> validation đã được dùng để phát triển mô hình. Chưa có đánh giá test độc lập, chưa đo tác động tăng mua do liên hệ khách.','small'),PageBreak(),
             P('Thiết kế thí nghiệm','title'),
             P('<b>12 lần fit trên dữ liệu thật.</b> Cùng CatBoost depth 4, 200 cây, learning rate 0,04, L2 leaf 5, seed 20261110 và CPU 2 threads. Giữ phân bố tự nhiên, không class weight, resampling hoặc early stopping.'),
             T(['Nhãn train','Validation','Số dòng train'],[['02–08/2022','09/2022','701.568'],['02–09/2022','10/2022','801.792'],['02–10/2022','11/2022','902.016']],
               [width*.38,width*.30,width*.32]),Spacer(1,12),
             P('Bốn bộ biến đối chứng','h1'),
             P('<b>Hành vi gốc (153):</b> trx, geo, dialog metadata trong tháng gần nhất và lịch sử cũ. <b>Hành vi (141):</b> bỏ 12 biến độ phủ nguồn theo lịch.'),
             P('<b>Lịch sử mua (15):</b> lag đúng tháng, nhãn từng sản phẩm, tần suất và recency mua. <b>Kết hợp (156):</b> 141 biến hành vi cộng 15 biến lịch sử mua.'),
             P('Recall@1% theo từng tháng','h1'),
             T(['Tháng','Dương','Rule','Hành vi','Lịch sử mua','Kết hợp'],
               [[m,next(r['positives'] for r in monthly if r['month']==m),
                 *[pct(next(r['top_k']['0.01']['recall'] for r in monthly if r['month']==m and r['model']==n))
                   for n in ['previous_month_purchase','behavior','purchase','combined']]]
                for m in sorted({r['month'] for r in monthly})],
               [width*.15,width*.11,width*.16,width*.18,width*.20,width*.20]),Spacer(1,12),
             P('Mỗi dòng chỉ dùng sự kiện trước cutoff và nhãn quá khứ đã khả dụng. Giữ khách thiếu geo/dialog. UTC và độ trễ nhãn 0 là giả định nghiên cứu, chưa xác nhận SLA nguồn.','small'),
             P('Tháng 12/2022 dành cho calibration trong protocol cũ. Lần ablation này không mở calibration hoặc test 01/2023. Run mới lưu model từng fold; artifact cũ giữ nguyên.','small'),PageBreak(),
             P('Độ tin cậy và định hướng','title'),
             P('Giá trị tăng thêm của từng bộ biến','h1'),
             T(['So sánh Recall@1%','Chênh lệch','Khoảng 95%'],
               [[NAMES[c['variant']]+' − '+NAMES[c['reference']],pp(c['estimate'])+' đpt',
                 '['+pp(c['interval_95'][0])+'; '+pp(c['interval_95'][1])+'] đpt'] for c in comparisons],
               [width*.52,width*.19,width*.29]),Spacer(1,11),
             P('500 lần paired bootstrap theo khách, giữ các tháng của khách đi cùng nhau và tính lại top-K. Khoảng có điều kiện theo model và tháng đã quan sát, chưa đo bất định của những tháng tương lai hoặc toàn bộ quá trình chọn mô hình.','small'),
             P('Quy tắc feature đã chốt','h1'),
             P('<b>Thời gian:</b> ghép đúng tháng lịch. Thiếu hoặc chưa có nhãn thì giữ unknown. Vocabulary, PCA và encoder tuân thủ train/cutoff của từng thí nghiệm.'),
             P('<b>Giá trị:</b> amount tách theo currency. Count 0 khác với mean/recency/embedding không xác định. Geo dùng ba cột mã có sẵn, chưa suy khoảng cách địa lý từ số mã.'),
             P('<b>Độ phức tạp:</b> chọn số biến và số chiều bằng validation. Gain không tự chứng minh một nhóm biến tương quan vô ích. Độ dài lịch sử nhãn cũng có thể đại diện thời gian.'),
             P('Ưu tiên tiếp theo','h1'),
             P('Ablation từng nguồn trên cùng nền lịch sử mua, sau đó cửa sổ 7/14/30/90 ngày và dialog PCA. CoLES là thí nghiệm nâng cấp sau khi các đối chứng đơn giản ổn định. Khóa mô hình và kiểm tra calibration trước khi mở test.'),
             P('Chưa thực hiện các nâng cấp embedding, tuning rộng, inference vận hành hay đánh giá tác động chiến dịch. Thay đổi code baseline làm run cũ không còn khớp code hash tại checkout mới; không sửa manifest để bỏ qua guard.','small'),
             P('Nguồn và tái lập','h1'),
             P('Báo cáo đầy đủ, CSV/JSON và hợp đồng feature: deliverables/baseline-final. Cấu hình: configs/propensity_ablation.json. Script: run_purchase_ablation.py và publish_baseline_ablation.py. Chỉ số liệu tổng hợp được công bố.','small'),
             P('MBD: arxiv.org/abs/2409.17587. Phương pháp đánh giá thời gian và tránh leakage: tài liệu scikit-learn. Tham số boosting: tài liệu XGBoost.','small')]
    output = root/'Baseline_Propensity_Report.pdf'
    if output.exists():
        raise FileExistsError(output)
    def footer(canvas,doc):
        canvas.setStrokeColor(teal);canvas.setLineWidth(1);canvas.line(43,39,A4[0]-43,39)
        canvas.setFont('Segoe',8);canvas.setFillColor(muted)
        canvas.drawString(43,25,'MBD-mini · Validation 09–11/2022 · Test giữ kín')
        canvas.drawRightString(A4[0]-43,25,str(doc.page))
    SimpleDocTemplate(str(output),pagesize=A4,rightMargin=43,leftMargin=43,topMargin=43,bottomMargin=52,
                      title='Baseline propensity — kết quả và định hướng',author='Propensity research').build(story,onFirstPage=footer,onLaterPages=footer)
    from pypdf import PdfReader
    reader=PdfReader(output)
    if len(reader.pages) != 3:
        raise ValueError(f'Expected three pages, got {len(reader.pages)}')
    text='\n'.join(p.extract_text() for p in reader.pages)
    for n in ('combined','purchase'):
        if pct(by[n]['mean_recall_at_1pct']) not in text:
            raise ValueError('Missing headline metric in PDF')
    print(json.dumps({'pdf':str(output),'pages':len(reader.pages),'metric_text_check':'passed'},ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('deliverables/baseline-final'))
    build(parser.parse_args().root)
