# Chẩn đoán baseline sau phản biện — 10/11/2026

Đọc diễn giải và kế hoạch ưu tiên trong [báo cáo baseline](../../BASELINE_REPORT.md).
Các file ở đây chỉ chứa số liệu tổng hợp, không có ID hoặc dự đoán theo khách.

| File | Ý nghĩa |
| --- | --- |
| `budget_metrics.csv` | Precision/Recall/Lift từng tháng, ba phương pháp, ngân sách 0,5%/1%/2%/5%/10% |
| `budget_macro.csv` | Trung bình đều các tháng của các metric trên |
| `segments_top1.csv` | Chọn top 1% toàn cohort rồi phân nhóm theo mua tháng trước hoặc lịch sử nhãn từ 02/2022; mỗi cách chia có mẫu số riêng |
| `overlap_top1.csv` | Độ chồng danh sách, Jaccard và lượt trúng chung/riêng của mô hình được chọn với rule |
| `cumulative_gain.csv` | Tỷ lệ ca mua được tìm thấy theo tỷ lệ khách được chọn, tới 100% |
| `label_prevalence.csv` | Bản sao thống kê nhãn tổng hợp đã audit, gồm tháng test đã xuất hiện trong EDA |
| `diagnostics.png` | Gain curve ở vùng 0–10% và tỷ lệ nhãn toàn chuỗi |
| `final_training_logloss.csv`, `final_training_logloss.png` | Logloss train theo số cây trích từ base cuối đã lưu; không phải cặp train/validation của các fold |
| `manifest.json` | Phạm vi, định nghĩa phân nhóm, seed và SHA256 input |

Nguồn: run `outputs/propensity_baseline_geo_dialog_v1/`, feature calendar v2
và bảng nhãn tổng hợp đã lưu trong EDA. Dự đoán được kiểm tra chỉ thuộc validation
09–11/2022. Nhãn dùng tạo phân nhóm thuộc 02–10 và luôn trước cutoff tương ứng.
Không train lại, không đọc feature hoặc dự đoán test. Model cuối được đọc để
kiểm tra hash và lấy chuỗi Logloss trên train 02–11/2022; không tính lại dự đoán.

Các mức 0,5%/1%/5% được đối chiếu khớp metric gốc. Các phân nhóm bảo toàn tổng
dòng/ca mua/lượt chọn/lượt trúng của top 1%; gain tăng không giảm và kết thúc
ở 100%; lượt trúng chung + riêng khớp 238 của mô hình và 169 của rule.
Validation đã dùng để chọn mô hình, nên chẩn đoán này vẫn mang tính khám phá.

Tái lập từ root, khi đã có artifact cục bộ và môi trường baseline:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[baseline,plots]"
.\.venv\Scripts\python.exe -B -X utf8 scripts/review_baseline.py --output outputs/baseline_diagnostics_recheck
```

Thư mục output phải chưa tồn tại. Clone Git chỉ có script và các bảng tổng hợp
đã công bố; cần tạo lại hoặc cung cấp artifact trong `outputs/` để chạy kiểm tra.
