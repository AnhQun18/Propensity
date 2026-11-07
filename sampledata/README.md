# Mẫu MBD-mini theo phân phối

`mbd_mini/` chứa **1.000 khách không lặp**, seed **42**, lấy từ 100.224 khách.
Phân tầng theo `fold × source_mask × activity_band`, phân bổ tỷ lệ bằng
Hamilton, lấy ngẫu nhiên không hoàn lại trong từng tầng. Các band dùng tổng
log toàn lịch sử: 0 log và các khoảng P25/P50/P75/P95/P99 của khách có log.

| File | Nội dung |
| --- | --- |
| `selection.parquet` | ID ẩn danh, fold, tầng và trọng số `population_n/sample_n` |
| `clients.parquet` | Khách được chọn và fold |
| `targets.parquet` | Toàn bộ 12 tháng nhãn của khách được chọn |
| `trx.parquet`, `geo.parquet`, `dialog.parquet` | Mọi log detail của khách được chọn, giữ cả lịch sử trước 01/2022 |
| `ptls_*.parquet` | Các chuỗi gốc tương ứng để đối chiếu; không cộng với detail |
| `customer_month.parquet` | Lưới khách × toàn bộ 25 tháng; số log raw và số log có chính sách null nguồn sớm |
| `coverage_selection.parquet` | Ánh xạ 40 khách minh họa theo nhóm có chủ đích; độc lập với mẫu phân phối 1.000 khách |
| `metadata.json` | Phương pháp, seed, fingerprint raw và số dòng từng bảng |

Trước 01/2022, ô geo/dialog không có log được đặt null với cờ
`unobserved_early_source`; ô có log giữ nguyên. `*_raw_events` bảo toàn số đếm.
Không sửa null thuộc tính, xóa duplicate ứng viên hoặc loại outlier.

Đây là mẫu phục vụ EDA theo toàn lịch sử. Các tầng hiếm có thể nhận 0 mẫu;
weight không khôi phục được tầng vắng. Mẫu không bảo toàn chính xác cực trị
hoặc tỷ lệ nhãn hiếm. Không dùng tầng toàn lịch sử để lấy mẫu train/test tại
cutoff quá khứ. Chẩn đoán mẫu và mọi thống kê chính trên toàn bộ raw nằm trong
[báo cáo EDA](../docs/mbd_mini_eda/report.md).

Tái tạo từ raw đã giải nén:

```powershell
.\.venv\Scripts\python.exe -u -X utf8 src/review_mbd_eda.py --sample-size 1000 --seed 42
```

Đọc một bảng (các mã nguyên có null nên giữ dtype nullable):

```python
import pandas as pd
trx = pd.read_parquet("sampledata/mbd_mini/trx.parquet", dtype_backend="pyarrow")
```

Nguồn dữ liệu: [ai-lab/MBD-mini](https://huggingface.co/datasets/ai-lab/MBD-mini)
