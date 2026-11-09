"""Build a five-page Canva supplement from exported aggregate evidence."""
from pathlib import Path
import csv
import html
import json
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/behavior_coverage/data_scale_canva"
TABLES = ROOT / "docs/mbd_mini_eda/tables"


def read(section, name):
    return list(csv.DictReader((TABLES / section / f"{name}.csv").open(encoding="utf-8-sig")))


def vi(n):
    return f"{n:,}".replace(",", ".")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = read("descriptive.structure", "table_inventory")
    roles = read("descriptive.structure", "column_roles")
    by_source = {r["source"]: r for r in inventory}
    monthly = read("descriptive.temporal", "monthly_activity")
    sources = [("transactions", "Giao dịch", "teal"), ("geo", "Vị trí", "mint"), ("dialog", "Hội thoại", "coral")]
    total_events = sum(int(by_source[s]["rows"]) for s, _, _ in sources)
    year_events = {s: sum(int(r["events"]) for r in monthly if r["source"] == s and r["month"].startswith("2022-")) for s, _, _ in sources}
    year_total = sum(year_events.values())
    logical = sum(int(r["logical_columns"]) for r in inventory)
    unique = len({r["column"] for r in roles})
    assert total_events == 105277859 and year_total == 85059894
    assert logical == len(roles) == 36 and unique == 26
    # Client presence in 2022 is computed from the complete cache, never summed across sources.
    import numpy as np
    with np.load(ROOT / "outputs/behavior_coverage/matrix.npz", allow_pickle=False) as data:
        months = data["months"]
        selected = data["counts"][:, (months >= "2022-01") & (months < "2023-01")]
    assert int(selected.sum()) == year_total
    customers_2022 = int((selected.sum(axis=1) > 0).sum())
    pages = []
    def add(title, subtitle, body, source, notes):
        number = len(pages) + 1
        pages.append(f'<section data-document-role="page" data-label="{html.escape(title)}" data-speaker-notes="{html.escape(notes, quote=True)}"><div class="brand">MBD / PROPENSITY LAB</div><h1>{title}</h1><p class="subtitle">{subtitle}</p><div class="content">{body}</div><footer><span>{source}</span><span>0{number} / 05</span></footer></section>')
    def stat(value, label, detail=""):
        return f'<div class="stat"><strong>{value}</strong><h3>{label}</h3><p>{detail}</p></div>'

    scale_bars = ''
    for source, name, color in sources:
        n = int(by_source[source]["rows"])
        scale_bars += f'<div class="source-row"><b>{name}</b><div class="track"><div class="fill {color}" style="width:{n/total_events*100}%"></div></div><span>{vi(n)} log</span></div>'
    add("Quy mô dữ liệu gốc", "Bản MBD-mini hiện có trên máy · toàn lịch sử nguồn",
        '<div class="stats">'+stat("100.224", "khách hàng", "Danh sách client_split")+stat("1.202.688", "dòng khách hàng–tháng", "100.224 khách × 12 tháng nhãn")+stat("105,28 triệu", "log hành vi", "Tổng detail/trx + geo + dialog")+'</div><div class="source-bars">'+scale_bars+'</div><p class="takeaway">Số log sự kiện khác với số mẫu huấn luyện propensity.</p>',
        "Nguồn: table_inventory.csv · phạm vi dữ liệu gốc",
        "Nguồn: docs/mbd_mini_eda/tables/descriptive.structure/table_inventory.csv. "
        "Ba nguồn detail có 105.277.859 dòng. Không cộng ptls vì là biểu diễn chuỗi của dữ liệu sự kiện. "
        "1.202.688 dòng nhãn thuộc 12 marker 02/2022–01/2023, không phải cùng phạm vi 01–12/2022 của feature. "
        "Số mẫu mô hình sau lọc cohort/chia thời gian chưa được xác định. Các thanh có cùng thang 0–105.277.859 sự kiện.")

    retention = ''
    for source, name, color in sources:
        full = int(by_source[source]["rows"])
        n = year_events[source]
        ratio = n / full * 100
        retention += f'<div class="history-row"><div class="history-label"><b>{name}</b><span>{vi(n)} log trong 2022</span></div><div class="track"><div class="fill {color}" style="width:{ratio}%"></div></div><strong>{ratio:.1f}%</strong></div>'
    add("Quy mô lịch sử 12 tháng", "01/01/2022 ≤ event_time &lt; 01/01/2023 · cutoff dự đoán tháng 01/2023",
        '<div class="stats two">'+stat("85.059.894", "log trong lịch sử đầu vào", "80,8% số log của toàn dữ liệu gốc")+stat(vi(customers_2022), "khách có ≥1 log", "Trên 100.224 khách ban đầu")+'</div><div class="history-bars">'+retention+'</div><p class="caption">Mỗi thanh chuẩn hóa theo tổng log toàn lịch sử của chính nguồn đó.</p>',
        "Nguồn: monthly_activity.csv + matrix.npz · lịch sử 01–12/2022",
        "Nguồn: docs/mbd_mini_eda/tables/descriptive.temporal/monthly_activity.csv, cộng events theo nguồn trong năm 2022; "
        "đối chiếu tổng bằng outputs/behavior_coverage/matrix.npz. Giao dịch 19.376.661, vị trí 65.564.488, hội thoại 118.745; tổng 85.059.894. "
        "Có 99.447 khách có ít nhất một log trong 12 tháng; không cộng số khách giữa nguồn hoặc tháng. "
        "Tháng 12/2022 kết thúc ở timestamp ngày 31/12 đã lưu; không khẳng định mỗi khách có lịch sử liên tục hoặc đủ ba nguồn. "
        "Tỷ lệ 80,8% được làm tròn từ 85.059.894/105.277.859. Timestamp naive theo lịch nguồn. "
        "Chính sách giữ ID/timestamp hợp lệ, không drop sự kiện chỉ vì thuộc tính khác null.")

    names = {"clients":"Khách hàng", "targets":"Nhãn", "transactions":"Giao dịch", "geo":"Vị trí", "dialog":"Hội thoại"}
    units = {"clients":"Khách", "targets":"Khách–tháng", "transactions":"Sự kiện", "geo":"Sự kiện", "dialog":"Sự kiện"}
    body_rows = ''
    for r in inventory:
        s = r["source"]
        ncols = int(r["logical_columns"])
        body_rows += f'<tr><td>{names[s]}</td><td>{units[s]}</td><td>{vi(int(r["rows"]))}</td><td>{ncols-1}</td><td><b>{ncols}</b></td></tr>'
    add("Số cột hiện có theo từng bảng", "5 bảng tham chiếu và detail · fold được bổ sung từ partition",
        '<div class="schema-summary"><div><strong>36</strong><span>cột logic cộng theo bảng</span></div><div><strong>26</strong><span>tên cột khác nhau</span></div></div><table><thead><tr><th>Bảng</th><th>Đơn vị dòng</th><th>Số dòng</th><th>Cột vật lý</th><th>Cột logic</th></tr></thead><tbody>'+body_rows+'</tbody></table><p class="takeaway">36 cột có tính tên lặp giữa các bảng như client_id, event_time và fold.</p>',
        "Nguồn: table_inventory.csv + column_roles.csv · schema gốc",
        "Cột logic: clients 2, targets 9, transactions 15, geo 6, dialog 4; tổng 36. "
        "Mỗi bảng có fold là partition thay vì cột vật lý trong shard; cột vật lý lần lượt 1,8,14,5,3, tổng 31. "
        "Có 26 tên cột phân biệt. Đây là số cột schema theo bảng, không phải 36 feature độc lập hay đã sẵn sàng cho mô hình. "
        "Nguồn: docs/mbd_mini_eda/tables/descriptive.structure/table_inventory.csv và column_roles.csv.")

    attr_rows = [r for r in roles if r["source"] in {s for s,_,_ in sources} and r["column"] not in {"client_id","event_time","fold"}]
    assert len(attr_rows) == 16
    assert sum(r["role"] == "categorical" for r in attr_rows) == 14
    features = '<div class="attribute-grid"><div><span class="label">GIAO DỊCH</span><strong>12</strong><h3>thuộc tính sự kiện</h3><p>1 amount<br>11 mã loại, tiền tệ, nguồn/đích</p></div><div><span class="label">VỊ TRÍ</span><strong>3</strong><h3>mã vị trí</h3><p>geohash_4<br>geohash_5 · geohash_6</p></div><div><span class="label">HỘI THOẠI</span><strong>1</strong><h3>cột embedding</h3><p>Vector 768 chiều<br>Không có văn bản gốc trong bảng</p></div></div>'
    add("Nhóm biến hành vi hiện có", "16 trường thuộc tính sự kiện · đã tách ID, thời gian và fold",
        features+'<div class="role-strip"><div><b>16</b><span>thuộc tính hành vi</span></div><div><b>14</b><span>ID / thời gian / fold</span></div><div><b>4</b><span>nhãn target</span></div><div><b>2</b><span>cột phụ trợ nhãn</span></div></div><p class="takeaway">768 chiều nằm trong 1 cột embedding; số feature khách hàng–tháng chưa được chốt.</p>',
        "Nguồn: column_roles.csv + embedding_quality.csv · thuộc tính gốc, chưa feature engineering",
        "Trong detail, bỏ client_id, event_time, fold: giao dịch có 12 thuộc tính (amount và 11 categorical), "
        "geo có 3 categorical, dialog có 1 vector. Tổng 16 = 14 categorical + 1 numeric + 1 vector. "
        "14 trường metadata cộng theo bảng gồm client_id 5 lần, fold 5 lần, event_time 3 lần, mon 1 lần. "
        "Bốn target là nhãn, không làm feature của cùng mẫu; hai auxiliary là trans_count và diff_trans_date chưa xác minh thời điểm tạo. "
        "Embedding 768 chiều đã kiểm tra trên 217.205 hội thoại, vẫn là một cột list trong schema. "
        "Nếu mở rộng vector thành các số đầu vào thì số chiều tensor khác số cột schema. Chưa có bảng feature khách–tháng hoàn chỉnh. "
        "Nguồn: docs/mbd_mini_eda/tables/descriptive.structure/column_roles.csv và descriptive.modalities/embedding_quality.csv.")

    matrix_html = (ROOT / "outputs/behavior_coverage/quartet_12months/canva_page.html").read_text(encoding="utf-8")
    matrix_section = re.search(r'<section data-document-role="page".*?</section>', matrix_html, re.S).group(0)
    matrix_section = matrix_section.replace('<section ', '<section class="matrix-page" ', 1)
    matrix_section = matrix_section.replace('<footer>', '<footer class="matrix-footer">')
    pages.append(matrix_section)
    css = '''
    :root{--bg:#f6f4ec;--paper:#fffcf6;--ink:#173d38;--dim:#526963;--line:#d6ded4;--teal:#267969;--mint:#9fc8ae;--coral:#db8e6e}
    *{box-sizing:border-box}html,body{margin:0;padding:0;width:1920px;background:var(--bg);font-family:Arial,sans-serif;color:var(--ink)}
    section{position:relative;width:1920px;height:1080px;padding:74px 100px;overflow:hidden}
    .brand{font-size:18px;color:var(--dim);letter-spacing:3px;margin-bottom:38px}h1{font-size:62px;line-height:1.1;letter-spacing:-1.8px;font-weight:600;margin:0 0 22px}
    .subtitle{font-size:27px;line-height:1.5;color:var(--dim);margin:0 0 40px}.content{position:relative}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:34px;margin-bottom:36px}.stats.two{grid-template-columns:1fr 1fr;margin-bottom:30px}
    .stat{border-top:3px solid var(--teal);padding:24px 0}.stat strong{font-size:66px;letter-spacing:-2px;font-weight:500;color:var(--teal)}.stat h3{font-size:27px;margin:14px 0 10px}.stat p{font-size:23px;color:var(--dim);margin:0}
    .source-bars{background:var(--paper);border-radius:26px;padding:22px 30px}.source-row{display:grid;grid-template-columns:210px 1fr 310px;align-items:center;gap:24px;margin:22px 0;font-size:25px}.source-row>b{font-weight:500}.source-row>span{text-align:right}.track{height:28px;background:#e5e9df;border-radius:30px;overflow:hidden}.fill{height:100%;border-radius:30px;min-width:4px}.teal{background:var(--teal)}.mint{background:var(--mint)}.coral{background:var(--coral)}
    .takeaway{font-size:25px;line-height:1.5;color:var(--ink);padding:21px 28px;background:#e6eee6;border-radius:20px;margin:28px 0 0}.history-row{display:grid;grid-template-columns:480px 1fr 110px;gap:26px;align-items:center;margin:28px 0}.history-label b{display:block;font-size:27px;margin-bottom:6px}.history-label span{font-size:23px;color:var(--dim)}.history-row>strong{font-size:28px;font-weight:500;text-align:right}.caption{font-size:22px;color:var(--dim);margin-top:24px}
    footer{position:absolute;bottom:44px;left:100px;right:100px;border-top:1px solid var(--line);padding-top:18px;display:flex;justify-content:space-between;font-size:18px;color:var(--dim)}
    .schema-summary{display:flex;gap:90px;margin-bottom:20px}.schema-summary>div{display:flex;align-items:baseline;gap:20px}.schema-summary strong{font-size:64px;font-weight:500;color:var(--teal)}.schema-summary span{font-size:26px;color:var(--dim)}
    table{width:100%;border-collapse:collapse;font-size:26px}th{font-size:21px;font-weight:500;letter-spacing:.02em;color:var(--dim);text-align:left;padding:18px 20px;border-bottom:2px solid var(--line)}td{padding:19px 20px;border-bottom:1px solid var(--line)}th:nth-child(n+3),td:nth-child(n+3){text-align:right}td:last-child{color:var(--teal)}
    .attribute-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:40px}.attribute-grid>div{background:var(--paper);border-radius:26px;padding:30px 36px}.attribute-grid .label{font-size:18px;letter-spacing:2px;color:var(--dim)}.attribute-grid strong{display:block;font-size:84px;font-weight:500;color:var(--teal);margin:15px 0}.attribute-grid h3{font-size:28px;margin:0 0 20px}.attribute-grid p{font-size:25px;line-height:1.65;color:var(--dim);margin:0}.role-strip{display:grid;grid-template-columns:repeat(4,1fr);margin-top:32px;gap:28px}.role-strip>div{border-top:2px solid var(--line);padding-top:18px}.role-strip b{font-size:38px;font-weight:500;color:var(--teal);margin-right:14px}.role-strip span{font-size:22px;color:var(--dim)}
    .matrix-page{padding:62px 64px}.matrix-page h1{font-size:48px;line-height:1.15;font-weight:600;margin:0 0 18px;letter-spacing:-1.1px}.matrix-page .sub{font-size:24px;color:var(--dim);margin:0}.matrix-page .charts{position:absolute;left:45px;right:45px;top:190px;display:grid;grid-template-columns:1fr 1fr;gap:10px 18px}.matrix-page img{display:block;width:906px;height:350px;object-fit:contain}.matrix-page .matrix-footer{left:64px;right:64px;bottom:50px;font-size:20px;line-height:1.6;display:block;padding-top:16px}
    '''
    head = '<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>MBD-mini · Quy mô và biến dữ liệu</title><style>'+css+'</style></head>'
    document = head+'<body>'+''.join(pages)+'</body></html>'
    (OUT / "canva_supplement.html").write_text(document, encoding="utf-8")
    for i, page in enumerate(pages, 1):
        (OUT / f"preview_{i}.html").write_text(head+'<body>'+page+'</body></html>', encoding="utf-8")
    result = {"new_pages":4, "pages_including_matrix":5, "logical_columns":logical, "distinct_column_names":unique,
              "raw_events":total_events, "history_2022_events":year_total, "history_2022_customers_with_log":customers_2022,
              "event_attribute_fields":16, "embedding_dimensions":768}
    (OUT / "facts.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
