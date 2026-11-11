# Báo cáo EDA 7 bước

## Đính chính ngoại lai ngày 11/11/2026

Kết quả chính dùng **IQR trên log1p**, biến đổi từng giá trị trước khi tính quartile,
đếm cả hai đuôi, tách riêng currency. `ln(amount)` dùng phân tích độ nhạy, IQR thang
gốc chỉ đối chiếu. Cách này khôi phục kết quả đã ghi trong
[rà soát ngày 09/11](../../docs/propensity_feasibility/tail_review/assessment.md).
Với currency 11: raw 5.521.385 dòng (14,2582%), ln 614.192 dòng (1,5861%),
log1p **584.139 dòng (1,5085%)**, gồm 486.868 thấp và 97.271 cao.

Biến đã log không log lần hai, biến có âm dùng signed-log1p. Không kết luận
ngoại lai cho binary/biến hằng, IQR log bằng 0, nhóm ít hơn 30 giá trị hoặc currency
thiếu. Các ngưỡng raw toàn kỳ chỉ phục vụ EDA; ngưỡng cho mô hình phải fit trên train.
Giữ raw, không tự động xóa/cap từ các cờ mô tả này.

## Bản báo cáo và trình bày ngày 11/11/2026

- [Báo cáo PDF tóm tắt 3 trang](executive_report.pdf): phát hiện chính, chất lượng dữ liệu,
  ngoại lai và hướng xử lý đặc trưng.
- [Bài trình bày Canva 9 trang](https://www.canva.com/d/421OH9fKUzOsWPK): tiếng Việt,
  tỷ lệ 16:9, có ghi chú thuyết trình và nguồn từng trang.
- [Bản trình bày HTML](presentation/index.html): mở offline, dùng phím trái/phải để chuyển
  trang, `F` để toàn màn hình, `S` để mở presenter view.
- [Nội dung tóm tắt](executive_summary.md), [dàn trang và ghi chú](presentation/outline.json).

Số liệu ngoại lai đã tính lại bằng log-IQR. Bản tóm tắt không bổ sung kết quả mô hình.
Canva được tạo bằng plugin từ [HTML nguồn](presentation/canva_import.html). Chữ và bảng
được giữ trong thiết kế; biểu đồ nhập từ SVG. Để thay dữ liệu biểu đồ, sửa CSV/source và
tạo lại bản HTML thay vì giả định đây là chart liên kết trực tiếp với CSV.

## Báo cáo kỹ thuật đầy đủ

Mở [report.html](report.html) để đọc báo cáo có bảng đầy đủ và biểu đồ nhúng sẵn,
hoặc [report.md](report.md) để đọc bản Markdown. Các bảng CSV ở `tables/`, ảnh PNG ở
`figures/`. `manifest.json` ghi phạm vi, phiên bản thư viện, cỡ mẫu, fingerprint
metadata đầu vào và kết quả đối soát. `input_inventory.json` là danh sách file đầu vào.

Notebook thực thi: [../../notebooks/eda.ipynb](../../notebooks/eda.ipynb).

- Raw: toàn lịch sử của cả 8 bảng; không dùng mẫu nhỏ thay cho thống kê toàn tập.
- Đặc trưng: toàn bộ train 02–08/2022, theo output ready notebook 03.
- Chỉ overview/độ phủ/nhãn được trình bày cho đủ 12 tháng.
- Pearson với target và tỷ lệ outlier: toàn bộ train; Spearman, ma trận nhiều biến
  và đồ thị đặc trưng: mẫu hash xác định tối đa 30.000 dòng, seed 42.
- Histogram amount: mẫu hash 50.000 giao dịch currency 11; thống kê amount và
  outlier theo currency dùng toàn bộ giao dịch.
- Giữ nguyên raw/output tiền xử lý; không tự động xóa giao dịch, điền null hoặc cap outlier.

Chạy lại từ thư mục gốc, dùng Python `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[eda]"
.\.venv\Scripts\python.exe -u -X utf8 scripts/execute_eda_notebook.py
```

Yêu cầu Python >=3.11. Chỉ cần cài dependency một lần.

Lệnh thực thi checkpoint notebook sau mỗi cell, giữ output trong notebook và lưu
bản sao/log ở `outputs/eda_execution/`. `scripts/build_eda_notebook.py` là công cụ
tác giả để dựng lại cấu trúc notebook; chạy nó sẽ xóa output notebook hiện tại.
Logic SQL và xuất báo cáo tại `src/propensity_eda.py`.

Kiểm tra gói báo cáo sau khi chạy:

```powershell
.\.venv\Scripts\python.exe scripts/validate_eda.py
```

Để dựng lại bản tóm tắt và HTML trình bày, cần `reportlab` và skill `html-ppt` trong
`~/.codex/skills/html-ppt/`, sau đó chạy `python scripts/build_eda_publication.py`.
Lệnh tạo lại file cục bộ; không tự cập nhật thiết kế Canva đã nhập.
