# Notebook của dự án

| Notebook | Mục đích | Dữ liệu |
| --- | --- | --- |
| [01_data_preview.ipynb](01_data_preview.ipynb) | Xem schema và vài dòng mẫu | `sampledata/mbd_mini/` |
| [02_data_preparation.ipynb](02_data_preparation.ipynb) | Chuẩn hóa nhãn/log, feature khách–tháng, audit và protocol | Raw `data/MBD-mini/` hoặc đường dẫn cấu hình |
| [03_data_processing.ipynb](03_data_processing.ipynb) | Quy trình xử lí hiện tại, có bảng các bước, kiểm tra ma trận và tách supervised/unlabelled | Raw theo `configs/propensity_processing.json` |
| [04_merge_processed.ipynb](04_merge_processed.ipynb) | Ghép các bảng khách–tháng đã nối nguồn thành một Parquet, đối soát và lưu vào `processed/` | Output ready/schema v2 của notebook 03 |

Sau notebook 03, dùng notebook 04 nếu cần một bảng hợp nhất: mặc định xuất
`processed/propensity.parquet`, giữ đủ các nguồn, khóa và nhãn. Kèm manifest,
schema và protocol. Notebook 04 mới viết, chưa chạy trên dữ liệu thật; đã kiểm
thử workflow bằng dữ liệu tổng hợp. Toàn bộ `/processed/` được Git ignore.

**Dùng notebook 03 cho đợt xử lí mới.** Notebook này ghi mục tiêu/cách làm/đầu ra
từng bước, bao gồm sửa cờ ngoại tệ khi chỉ có currency unknown. Output riêng tại
`outputs/propensity_processing_v3/`, không ghi đè output của notebook 02.
Giữ file features đầy đủ cho scoring; dòng mất nhãn được tách khỏi supervised.
Ngày 10/11/2026, cả hai đã chạy raw thành công. Mỗi bộ có 1.202.688 dòng và 12.854
nhãn dương; bản thực thi/log nằm trong `outputs/notebook_execution_20261110_01/`.

Notebook 02 chứa đầy đủ định nghĩa hàm và SQL, chia riêng các bước chạy. Không cần
import package nội bộ. Cài dependency và chọn kernel Python >=3.11 theo hướng dẫn
đầu notebook. Chỉ các cell từ mục 5 trở đi đọc raw và ghi dữ liệu khi được chạy.

Notebook nguồn giữ output trống. Bản sửa schema v2 và validation dùng chung đã
được kiểm thử tổng hợp; output thật đã kiểm tra lại đủ 12 tháng. Chưa có kết quả mô hình.
Notebook không tự chạy khi mở file. Sau lỗi, sửa nguyên nhân, đổi output và Restart Kernel; không
chạy lại nửa chừng trên output đã tồn tại.

Helper tương ứng ở `src/propensity_data/` để CLI và kiểm thử dùng chung định nghĩa.
Metadata `source_file` của các cell cho biết nguồn. Khi sửa helper, công cụ tác giả
có thể cập nhật notebook trên máy được phép thao tác:

```powershell
python scripts/sync_preparation_notebook.py
```

Công cụ đồng bộ notebook 02, 03 và 04 bằng văn bản, không execute cell hoặc đọc dataset. Các cell điều
phối được giữ nguyên. Không commit output, ma trận hoặc dữ liệu khách hiển thị từ
lần chạy cục bộ. Xem [quy tắc và cấu hình](../docs/propensity_data_preparation.md).
