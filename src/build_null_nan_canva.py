"""Append three evidence-based null/NaN pages to the Canva report copy."""
from pathlib import Path
import csv
import html
import json
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/behavior_coverage/null_nan_canva"


def vi(n): return f"{n:,}".replace(",", ".")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    quality = ROOT / "docs/mbd_mini_eda/tables/descriptive.quality"
    missing = list(csv.DictReader((quality / "missing_values.csv").open(encoding="utf-8-sig")))
    audit = json.loads((ROOT / "docs/mbd_temporal_coverage/summary.json").read_text(encoding="utf-8"))
    missing_columns = [r for r in missing if int(r["missing_n"]) > 0]
    assert len(missing) == 36 and len(missing_columns) == 10
    for r in missing:
        if r["source"] in {"transactions", "geo", "dialog"} and r["column"] != "fold":
            mod = "trx" if r["source"] == "transactions" else r["source"]
            assert int(r["missing_n"]) == audit["modalities"][mod]["null_cells"][r["column"]]
        elif r["source"] == "targets":
            assert int(r["missing_n"]) == audit["target_null_cells"][r["column"]]
    trx = audit["modalities"]["trx"]
    dialog = audit["modalities"]["dialog"]
    assert trx["nonfinite_or_null_amount_rows"] == 0
    assert dialog["embedding_with_nonfinite_or_null_element_rows"] == 0
    assert dialog["wrong_or_null_embedding_length_rows"] == 0
    new_pages = []

    def add(title, subtitle, body, source, notes):
        new_pages.append(f'<section data-document-role="page" data-label="{html.escape(title)}" data-speaker-notes="{html.escape(notes, quote=True)}"><div class="brand">MBD / PROPENSITY LAB</div><h1>{title}</h1><p class="subtitle">{subtitle}</p><div class="content">{body}</div><footer><span>{source}</span><span>QUALITY</span></footer></section>')

    names = {"clients":"Khách hàng", "targets":"Nhãn", "transactions":"Giao dịch", "geo":"Vị trí", "dialog":"Hội thoại"}
    row_population = {r["source"]: int(r["rows"]) for r in missing}
    row_missing = {"clients":0, "targets":120786, "transactions":trx["invalid_rows"], "geo":0, "dialog":0}
    table_rows = ''
    for source in names:
        n = row_missing[source]
        total = row_population[source]
        cols = sum(int(r["missing_n"]) > 0 for r in missing if r["source"] == source)
        rate = f'{n/total*100:.3f}%'.replace('.',',')
        table_rows += f'<tr><td>{names[source]}</td><td>{vi(total)}</td><td>{cols}</td><td>{vi(n)}</td><td>{rate}</td></tr>'
    add("Giá trị thiếu theo từng bảng", "Phạm vi: toàn dữ liệu gốc · dòng có ít nhất một thuộc tính null",
        '<div class="q-summary"><div><strong>10 / 36</strong><span>cột logic có null</span></div><div><strong>0</strong><span>nhãn target bị thiếu</span></div></div><table class="q-table"><thead><tr><th>Bảng</th><th>Tổng dòng</th><th>Cột có null</th><th>Dòng có null</th><th>Tỷ lệ dòng</th></tr></thead><tbody>'+table_rows+'</tbody></table><p class="takeaway">Không cộng số ô null giữa các cột để suy ra số dòng bị thiếu.</p>',
        "Nguồn: missing_values.csv + temporal_coverage/summary.json · full scan",
        "Phạm vi toàn dữ liệu gốc, không chỉ 01–12/2022. "
        "10/36 là số cột logic theo bảng có ít nhất một null; 26 cột logic còn lại không thiếu. "
        "Số dòng có null: targets 120.786/1.202.688; transactions 342.181/38.764.930. "
        "Số 342.181 lấy từ audit strict: không có lỗi ID/timestamp hoặc amount; mọi dòng bị loại đều thiếu ít nhất một thuộc tính phân loại. "
        "Khách hàng, vị trí và hội thoại không có null trong các cột đã kiểm tra. ID/thời gian sự kiện và bốn nhãn đều đầy đủ. "
        "Nguồn: docs/mbd_mini_eda/tables/descriptive.quality/missing_values.csv; docs/mbd_temporal_coverage/summary.json. "
        "Null trong cột khác với khách không có log hoặc không có modality.")

    rows = ''
    fields = [("dst_type11 / dst_type12", "dst_type11"), ("src_type11 / src_type12", "src_type11"),
              ("src_type31 / src_type32", "src_type31"), ("src_type21 / src_type22", "src_type21"), ("currency", "currency")]
    for label, column in fields:
        r = next(r for r in missing if r["source"] == "transactions" and r["column"] == column)
        n = int(r["missing_n"])
        rate = n / int(r["rows"]) * 100
        digits = 5 if rate < .001 else 3
        rate_label = f'{rate:.{digits}f}%'.replace('.',',')
        rows += f'<div class="q-nullrow"><div><b>{label}</b><small>{vi(n)} ô null / cột</small></div><div class="track"><div class="fill teal" style="width:{rate/.7*100}%"></div></div><span>{rate_label}</span></div>'
    add("Các cột đang có null", "Giao dịch: mẫu số 38.764.930 dòng · cặp mã có số null bằng nhau, tính riêng từng cột",
        '<div class="q-split"><div class="q-chart"><h3>9 cột giao dịch</h3>'+rows+'<p class="q-scale">Các thanh cùng thang 0–0,7% · currency rất nhỏ, đọc số ghi bên cạnh</p></div><div class="q-target"><span>THUỘC TÍNH PHỤ TRỢ NHÃN</span><h3>diff_trans_date</h3><strong>10,043%</strong><p>120.786 / 1.202.688 dòng</p><div class="q-targetbar"><i></i></div><p class="q-target-note">Thang 0–100% riêng cho bảng targets.</p><hr><p>Bốn target đều đầy đủ.<br>trans_count không thiếu.</p></div></div>',
        "Nguồn: missing_values.csv · số null theo cột, không cộng chồng",
        "dst_type11 và dst_type12 mỗi cột 243.824 null (0,628981%); "
        "src_type11 và src_type12 mỗi cột 217.361 (0,560716%); "
        "src_type31 và src_type32 mỗi cột 78.996 (0,203782%); "
        "src_type21 và src_type22 mỗi cột 42.293 (0,109101%); currency 118 (0,0003044%). "
        "Việc hai cột có cùng số null không tự chứng minh null cùng vị trí dòng; báo cáo chỉ ghép nhãn cho gọn và ghi rõ mỗi cột. "
        "Bên trái thang 0–0,7% của giao dịch, bên phải 0–100% của targets; không so chiều dài thanh giữa hai bảng. "
        "diff_trans_date là int32 trong Parquet nên giá trị thiếu là null; pandas có thể biểu diễn thành NA hoặc NaN sau chuyển kiểu. "
        "Nguồn: docs/mbd_mini_eda/tables/descriptive.quality/missing_values.csv.")

    add("NaN / ±Inf và ảnh hưởng khi xử lý", "Kiểm tra giá trị số trên toàn lịch sử · không suy rộng sang feature phát sinh sau này",
        '<div class="q-checks"><div><span>AMOUNT</span><strong>0</strong><h3>null / NaN / ±Inf</h3><p>38.764.930 số tiền được kiểm tra</p></div><div><span>EMBEDDING</span><strong>0</strong><h3>vector có phần tử lỗi</h3><p>217.205 vector · đủ 768 chiều<br>Không có phần tử null / NaN / ±Inf</p></div><div><span>BỐN NHÃN TARGET</span><strong>0</strong><h3>ô nhãn bị thiếu</h3><p>Đủ 1.202.688 dòng khách–tháng<br>Không loại chỉ vì cột phụ trợ null</p></div></div><div class="q-decision"><b>Nếu dropna toàn bảng targets:</b><span>1.202.688 → 1.081.902 dòng</span><p>Mất 120.786 dòng nhãn hợp lệ chỉ vì diff_trans_date thiếu.</p></div><p class="takeaway">Đề xuất: giữ nhãn hợp lệ; xử lý missing theo cột. Không có log ≠ ô null.</p>',
        "Nguồn: temporal_coverage/summary.json + amount_summary.csv · full scan",
        "Null = không có giá trị tại một trường. NaN = giá trị dấu phẩy động không phải số; ±Inf = số vô hạn, không phải null. "
        "Audit np.isfinite trên toàn bộ amount cho thấy 0 giá trị null/không hữu hạn; vì vậy không phát hiện NaN hoặc ±Inf. "
        "Audit embedding kiểm tra chiều dài và từng phần tử: 0 vector sai chiều hoặc chứa phần tử null/không hữu hạn. "
        "Đây là kiểm tra các trường số trong dữ liệu gốc; không khẳng định feature sinh sau chia/log/chuẩn hóa sẽ không phát sinh NaN/Inf. "
        "dropna mọi cột targets chỉ còn 1.081.902 dòng trong khi bốn nhãn đầy đủ ở 1.202.688 dòng; số ID vẫn 100.224 xuất hiện ở ít nhất một dòng. "
        "Đề xuất chưa áp dụng biến đổi: giữ nhãn hợp lệ, cân nhắc unknown cho categorical; không dùng trans_count/diff_trans_date trước khi xác minh cutoff. "
        "Ma trận đã vẽ giữ log có ID/timestamp hợp lệ, gồm cả log thiếu categorical; không phải tập strict complete-case. "
        "Nguồn: docs/mbd_temporal_coverage/summary.json; docs/mbd_mini_eda/tables/descriptive.transactions/amount_summary.csv.")

    previous = (ROOT / "outputs/behavior_coverage/data_scale_canva/canva_supplement.html").read_text(encoding="utf-8")
    head = previous.split('</head>')[0]
    css = '''
    .q-summary{display:flex;gap:100px;align-items:baseline;margin-bottom:22px}.q-summary>div{display:flex;gap:24px;align-items:baseline}.q-summary strong{font-size:64px;color:var(--teal);font-weight:500}.q-summary span{font-size:26px;color:var(--dim)}
    .q-table th{font-size:22px}.q-table td{padding-top:20px;padding-bottom:20px}.q-table th:nth-child(n+2),.q-table td:nth-child(n+2){text-align:right}.q-table td:last-child{font-weight:600;color:var(--teal)}
    .q-split{display:grid;grid-template-columns:1.6fr 1fr;gap:40px}.q-chart{background:var(--paper);border-radius:26px;padding:28px 32px}.q-chart h3{font-size:28px;margin:0 0 28px}.q-nullrow{display:grid;grid-template-columns:370px 1fr 125px;align-items:center;gap:20px;margin-bottom:28px}.q-nullrow b{display:block;font-size:24px;font-weight:500}.q-nullrow small{display:block;font-size:20px;margin-top:7px;color:var(--dim)}.q-nullrow>span{text-align:right;font-size:25px}.q-nullrow .track{height:25px}.q-scale{font-size:18px;color:var(--dim);margin:26px 0 0;line-height:1.5}
    .q-target{background:#e6eee6;border-radius:26px;padding:36px}.q-target>span{font-size:18px;letter-spacing:1px;color:var(--dim)}.q-target h3{font-size:30px;margin:24px 0}.q-target strong{font-size:72px;font-weight:500;color:var(--teal)}.q-target p{font-size:24px;color:var(--dim);line-height:1.6}.q-targetbar{height:28px;border-radius:20px;background:#d3dfd3;overflow:hidden}.q-targetbar i{display:block;width:10.043%;height:100%;background:var(--teal);border-radius:20px}.q-target .q-target-note{font-size:18px;margin-top:10px}.q-target hr{border:0;border-top:1px solid #c3d2c4;margin-top:24px}
    .q-checks{display:grid;grid-template-columns:repeat(3,1fr);gap:35px}.q-checks>div{background:var(--paper);border-radius:26px;padding:28px 30px}.q-checks span{font-size:18px;letter-spacing:2px;color:var(--dim)}.q-checks strong{display:block;font-size:80px;font-weight:500;color:var(--teal);margin:12px 0}.q-checks h3{font-size:28px;margin:0 0 18px}.q-checks p{font-size:22px;line-height:1.6;margin:0;color:var(--dim)}
    .q-decision{margin-top:30px;border-top:2px solid var(--line);padding-top:22px}.q-decision b{font-size:26px;margin-right:36px}.q-decision span{font-size:35px;color:var(--teal)}.q-decision p{font-size:24px;line-height:1.5;color:var(--dim);margin:12px 0 0}
    '''
    head += '<style>'+css+'</style></head>'
    old_pages = re.findall(r'<section\b.*?</section>', previous, re.S)
    assert len(old_pages) == 5
    pages = old_pages[:4] + new_pages + old_pages[4:]
    for i in range(7):
        pages[i] = re.sub(r'<span>(?:\d{2} / 05|QUALITY)</span>', f'<span>{i+1:02d} / 08</span>', pages[i])
    doc = head+'<body>'+''.join(pages)+'</body></html>'
    (OUT / 'canva_report_null_nan.html').write_text(doc, encoding='utf-8')
    for i, page in enumerate(new_pages, 1):
        (OUT / f'quality_preview_{i}.html').write_text(head+'<body>'+page+'</body></html>', encoding='utf-8')
    facts = {"pages":8,"new_quality_pages":3,"scope":"full raw history, not the 2022 matrix subset",
             "columns_with_null":10,"logical_columns":36,"row_population":row_population,
             "rows_with_null":row_missing,"amount_nonfinite_or_null":0,"embedding_bad_vectors":0,
             "labels_missing":0,"targets_rows_lost_by_dropna_all_columns":120786}
    (OUT/'facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(facts,ensure_ascii=False))


if __name__ == '__main__':
    main()
