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
| [18](18_join_quality_audit.ipynb) | Audit join từ raw, đối soát cache/panel, kiểm tra nhãn và độ phủ nguồn |
| [19](19_univariate_distributions.ipynb) | Đơn biến: nhãn 12 tháng, phân vị amount theo currency và quy mô nhóm |
| [20](20_multivariate_insights.ipynb) | Insight đa biến: tỷ lệ mua, đóng góp, lift theo tháng/horizon, tương tác TRX–DIALOG và lịch sử mua |

**Bộ notebook insight 18–20:** chạy theo thứ tự 18 → 19 → 20 bằng kernel `.venv`. Notebook 18 quét toàn bộ raw và làm mới audit; notebook 19–20 dùng các kết quả đó. Notebook 20 còn dùng hai bảng tương tác có sẵn trong `outputs/target_focused_moscow_20261119/`, ghi rõ nguồn ở cell tương ứng. Các notebook lưu bảng và biểu đồ inline, xuất PNG/SVG tại `outputs/eda_insights_20261119/charts/`. Tạo lại và chạy riêng bộ này bằng:

```powershell
.venv/Scripts/python.exe -B -X utf8 src/scripts/build_insight_notebooks.py --execute
```

Lệnh trên ghi lại ba notebook 18–20 từ mã nguồn sinh notebook. Không áp dụng `build_eda_workflow_notebooks.py` để tái tạo bộ này. Xem [sổ insight và checklist](../docs/eda_insights_checklist_20261119.md) để phân biệt kiểm tra đã chạy với phần nghiên cứu còn mở.

`01_data_preview.ipynb` đọc `sample_data/`; mọi kết luận EDA chính phải kiểm kê toàn bộ khách hàng và ít nhất một năm nhãn. Một số mã nguồn được chép vào notebook là phương pháp cũ (UTC, cửa sổ cố định hoặc mẫu khách); việc đưa mã vào notebook **không** biến kết quả cũ thành kết quả tuân thủ Moscow. Xem [đánh giá phạm vi](../docs/eda_two_lists_audit_20261118.md) trước khi dùng các đầu ra này.

Các notebook có thể quét raw và ghi `outputs/`; không chạy toàn bộ 17 notebook như một pipeline. Chọn notebook theo câu hỏi nghiên cứu, đọc phần mô tả và kiểm tra phạm vi dữ liệu trước khi chạy. Mã thư viện, kiểm tra, xuất báo cáo và notebook EDA cũ không được nhân bản tại đây; chúng vẫn có trong `src/` khi cần đối chiếu. Notebook 03 và 04 dùng cho chuẩn bị dữ liệu; kiểm tra mã nguồn được nhúng bằng:

```powershell
.venv/Scripts/python.exe src/scripts/sync_preparation_notebook.py --check
```
