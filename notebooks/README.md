# Notebook của dự án

| Notebook | Mục đích | Dữ liệu |
| --- | --- | --- |
| [01_data_preview.ipynb](01_data_preview.ipynb) | Xem schema và vài dòng mẫu | `sampledata/mbd_mini/` |
| [02_data_preparation.ipynb](02_data_preparation.ipynb) | Chuẩn hóa nhãn/log, feature khách–tháng, audit và protocol | Raw `data/MBD-mini/` hoặc đường dẫn cấu hình |
| [03_data_processing.ipynb](03_data_processing.ipynb) | Quy trình xử lí hiện tại, có bảng các bước, kiểm tra ma trận và tách supervised/unlabelled | Raw theo `configs/propensity_processing.json` |

**Dùng notebook 03 cho đợt xử lí mới.** Notebook này ghi mục tiêu/cách làm/đầu ra
từng bước, bao gồm sửa cờ ngoại tệ khi chỉ có currency unknown. Output riêng tại
`outputs/propensity_processing_v3/`, không ghi đè output của notebook 02.
Giữ file features đầy đủ cho scoring; dòng mất nhãn được tách khỏi supervised.
Không suy ra đã chạy pipeline từ nội dung notebook: cả 02 và 03 đều chưa thực thi.

Notebook 02 chứa đầy đủ định nghĩa hàm và SQL, chia riêng các bước chạy. Không cần
import package nội bộ. Cài dependency và chọn kernel Python >=3.11 theo hướng dẫn
đầu notebook. Chỉ các cell từ mục 5 trở đi đọc raw và ghi dữ liệu khi được chạy.

Notebook mới **chưa được thực thi**, không có output hoặc kết quả mô hình. Không tự
chạy khi mở file. Sau lỗi, sửa nguyên nhân, đổi output và Restart Kernel; không
chạy lại nửa chừng trên output đã tồn tại.

Helper tương ứng ở `src/propensity_data/` để CLI và kiểm thử dùng chung định nghĩa.
Metadata `source_file` của các cell cho biết nguồn. Khi sửa helper, công cụ tác giả
có thể cập nhật notebook trên máy được phép thao tác:

```powershell
python scripts/sync_preparation_notebook.py
```

Công cụ đồng bộ cả notebook 02 và 03 bằng văn bản, không execute cell hoặc đọc dataset. Các cell điều
phối được giữ nguyên. Không commit output, ma trận hoặc dữ liệu khách hiển thị từ
lần chạy cục bộ. Xem [quy tắc và cấu hình](../docs/propensity_data_preparation.md).
