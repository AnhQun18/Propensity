"""Build the seven-step EDA notebook. Execution is separate, using the local venv."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []
def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))
def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))

md("""# EDA MBD-mini — 7 bước đầy đủ

Phân tích raw đa bảng và đặc trưng khách–tháng, mục tiêu **mua ít nhất một sản phẩm (`y_any`)**.
Notebook tính lại từ dữ liệu cục bộ; không dùng bộ mẫu 1.000 khách hoặc số liệu cache cũ thay cho toàn bộ dữ liệu.
Mỗi bảng ghi phạm vi; thống kê SQL là toàn bộ dữ liệu trong phạm vi, biểu đồ lấy mẫu có seed và số dòng rõ ràng.

**Chạy:** chọn Python của `.venv`, Run All. Cần `duckdb`, `pandas`, `numpy`, `matplotlib`,
`scipy`, `nbformat`, `nbclient`, `ipykernel` và output ready của notebook 03.
Kết quả: `deliverables/eda/report.html` (tự chứa ảnh), `report.md`, `tables/*.csv`, `figures/*.png`, `manifest.json`.
Raw và output tiền xử lý được đọc nguyên trạng. Thời gian được diễn giải theo quy ước UTC của dự án.

Các ô bên dưới thể hiện trực tiếp phép phân tích, bảng và biểu đồ. Hàm SQL/xuất báo cáo dùng chung ở `src/propensity_eda.py`.
""")
code('''from pathlib import Path
import sys
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "configs/propensity_processing.json").exists())
sys.path.insert(0, str(ROOT / "src"))
from propensity_eda import EDA
from IPython.display import display, Markdown
eda = EDA(ROOT)
display(Markdown(eda.scope_note))
''')
md("""## 1. Cái nhìn tổng quan

Đơn vị quan sát: khách, khách–tháng, sự kiện và chuỗi khách. `detail` và `ptls` là hai biểu diễn,
không cộng số sự kiện. Kiểu lưu số không đồng nghĩa biến định lượng: currency, geohash, event_type là mã phân loại.
Head/tail theo khóa/thời gian, ID được ẩn khi hiển thị; vector/mảng chỉ hiện số phần tử để bảng dễ đọc.
""")
code('''overview, schema = eda.overview()
display(overview)
display(schema)
for name, frame in eda.previews.items():
    display(Markdown(f"**{name}: head(3) và tail(3), đã sắp xếp**"))
    display(frame)
''')
md("""## 2. Thống kê mô tả

Biến số: count, mean, median, std, min/max, Q1/Q3, skewness. Biến phân loại: số giá trị khác nhau,
mode, số lần/tỷ lệ xuất hiện mode và top nhóm. Thống kê amount tách theo currency; không quy đổi hoặc gộp tổng khác đơn vị.
Đặc trưng chỉ mô tả trên train 02–08/2022; các nhãn gốc được giữ để đối soát `y_any`.
""")
code('''raw_numeric, categories, currency = eda.raw_descriptive()
display(raw_numeric)
display(categories)
display(currency)
feature_stats = eda.feature_descriptive()
display(feature_stats)
''')
md("""## 3. Dữ liệu thiếu và trùng lặp

Tỷ lệ thiếu được tính cho từng cột, phân biệt SQL NULL, NaN và vô cực.
Kiểm tra full-row duplicate cho detail; kiểm tra khóa khách và khách–tháng cho bảng có khóa duy nhất.
Trùng khách–timestamp chỉ là ứng viên kiểm tra, không đủ bằng chứng để xóa giao dịch.
Chỉ tạo bản nhãn loại duplicate **giống toàn bộ dòng** để minh họa xử lý an toàn; dừng nếu khóa nhãn mâu thuẫn.
Không ghi đè raw, không điền giá trị hay xóa outlier trong EDA.
""")
code('''missing, duplicates, conflicts = eda.quality()
display(missing.sort_values("missing_pct", ascending=False))
display(duplicates)
display(conflicts)
display(Markdown(eda.missing_note))
''')
md("""## 4. Phân tích đơn biến

Histogram gốc giữ trục tuyến tính; bản log1p bổ sung để quan sát đuôi dài và không thay thế dữ liệu.
Bar chart hiển thị top mã kèm Other/Missing. Biểu đồ target/tháng và độ phủ nguồn dùng toàn bộ cohort.
""")
code('''eda.univariate()
eda.show_figures("univariate")
display(eda.tables["target_monthly"])
display(eda.tables["source_coverage"])
''')
md("""## 5. Phân tích hai biến và đa biến

Scatter giữa số lượng giao dịch và giá trị giao dịch; boxplot số–nhãn; tỷ lệ mua theo nhóm đặc trưng;
Pearson và Spearman với target, heatmap tương quan và cặp có |r| ≥ 0,90.
Pearson với target dùng toàn bộ train. Spearman/ma trận nhiều biến dùng mẫu hash cố định tối đa 30.000 dòng,
ghi số quan sát cặp, loại ID/cutoff/nhãn/biến hằng. Không suy diễn quan hệ nhân quả hoặc coi tương quan là feature importance của mô hình.
""")
code('''associations, pairs = eda.multivariate()
display(associations.head(30))
display(pairs.head(40))
display(eda.tables["feature_band_target_rates"])
eda.show_figures("multivariate")
''')
md("""## 6. Phát hiện ngoại lai

**Kết quả chính: IQR trên `log1p(x)`**, tính cả hai đuôi. Biến đổi từng giá trị DOUBLE trước
khi tính Q1/Q3, rồi xét ngoài [Q1−1,5×IQR; Q3+1,5×IQR] trên thang log.
Amount tính riêng từng currency trên toàn log; đặc trưng số tính trên toàn train.
Xuất thêm `ln(amount)` để phân tích độ nhạy và thang gốc để đối chiếu. Z-score phụ trợ cũng tính trên thang biến đổi.
Biến đã log không log lần hai; biến có âm dùng signed-log1p. Binary/biến hằng, IQR log=0,
nhóm dưới 30 giá trị hoặc currency thiếu không được kết luận ngoại lai.
Ngưỡng toàn kỳ của raw chỉ là EDA, không áp trực tiếp vào mô hình quá khứ. Giữ nguyên raw, không tự động xóa/cap.
""")
code('''feature_outliers, amount_outliers = eda.outliers()
display(feature_outliers.sort_values("iqr_pct", ascending=False).head(40))
display(amount_outliers)
display(eda.tables["amount_iqr_scale_comparison_currency11"])
eda.show_figures("outliers")
''')
md("""## 7. Kết luận và chuẩn bị Feature Engineering

Kết luận được sinh từ kết quả vừa tính: mất cân bằng nhãn, nguồn thiếu, biến hằng, độ lệch,
nhóm biến liên hệ với nhãn và các cặp tương quan cao. Đề xuất encoding/scaling/cờ thiếu/log/count/rate,
giữ cutoff, tách currency và tránh leakage. Các biến được xếp hạng là ứng viên nghiên cứu cần xác nhận bằng validation theo thời gian.
""")
code('''conclusions, actions = eda.conclude()
display(Markdown(conclusions))
display(actions)
''')
md("""## Xuất báo cáo và xác minh

Báo cáo HTML mở trực tiếp bằng trình duyệt, không cần mạng. Bảng CSV giữ đầy đủ cột và biến không bị cắt như phần xem trước.
Manifest ghi fingerprint metadata nguồn, phiên bản thư viện, phạm vi, cỡ mẫu, kiểm tra đối soát và danh mục kết quả.
""")
code('''report = eda.export()
display(Markdown(f"Đã xuất **{report}**"))
display(eda.validation)
eda.close()
''')
nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python (propensity .venv)", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbf.write(nb, ROOT / "notebooks/eda.ipynb")
print("Built notebooks/eda.ipynb")
