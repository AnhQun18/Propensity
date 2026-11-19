# Notebook của dự án

Chọn kernel Python của `.venv` và mở notebook từ repo này. Các notebook EDA được đánh số theo thứ tự đọc, chia theo một chủ đề; chúng chứa bản sao mã phân tích từ `src/` và ghi SHA-256 của file nguồn trong metadata `eda_sources`. Script và thư viện EDA trong `src/` được giữ lại theo yêu cầu. Tạo lại các bản sao bằng:

```powershell
.venv/Scripts/python.exe src/scripts/build_eda_workflow_notebooks.py
```

| Số | Chủ đề |
| --- | --- |
| 01–04 | Xem dữ liệu mẫu, kiểm kê dữ liệu, chuẩn bị và ghép dữ liệu |
| 05–07 | Nhãn mua và thống kê targets, độ phủ thời gian, missingness toàn lịch sử |
| 08–10 | Chất lượng, biểu đồ, thời điểm và amount giao dịch TRX |
| 11–13 | Chất lượng, hoạt động, liên hệ nhãn và biểu đồ GEO |
| 14 | Thời gian, hình học và phân cụm DIALOG |
| 15–17 | Quan hệ đa nguồn, vòng đời khách hàng và biểu đồ EDA theo Moscow |

`01_data_preview.ipynb` đọc `sample_data/`; mọi kết luận EDA chính phải kiểm kê toàn bộ khách hàng và ít nhất một năm nhãn. Một số mã nguồn được chép vào notebook là phương pháp cũ (UTC, cửa sổ cố định hoặc mẫu khách); việc đưa mã vào notebook **không** biến kết quả cũ thành kết quả tuân thủ Moscow. Xem [đánh giá phạm vi](../docs/eda_two_lists_audit_20261118.md) trước khi dùng các đầu ra này.

Các notebook có thể quét raw và ghi `outputs/`; không chạy toàn bộ 17 notebook như một pipeline. Chọn notebook theo câu hỏi nghiên cứu, đọc phần mô tả và kiểm tra phạm vi dữ liệu trước khi chạy. Mã thư viện, kiểm tra, xuất báo cáo và notebook EDA cũ không được nhân bản tại đây; chúng vẫn có trong `src/` khi cần đối chiếu. Notebook 03 và 04 dùng cho chuẩn bị dữ liệu; kiểm tra mã nguồn được nhúng bằng:

```powershell
.venv/Scripts/python.exe src/scripts/sync_preparation_notebook.py --check
```
