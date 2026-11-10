# Baseline propensity — bàn giao 11/11/2026

- [Canva 8 slide](https://www.canva.com/d/YUag-LZ1qm4-TYU): tiếng Việt, khổ 16:9,
  có ghi chú thuyết trình. Đã kiểm tra hình xem trước và số liệu sau khi nhập.
- [PDF tóm tắt 3 trang](Baseline_Propensity_Report.pdf): kết quả, thiết kế đánh giá,
  khoảng bootstrap và định hướng tiếp theo.
- [Báo cáo đầy đủ](REPORT.md) và [bản HTML](report.html): diễn giải, giới hạn và nguồn.
- [Trình bày HTML offline](presentation/index.html): phím trái/phải chuyển slide,
  `F` toàn màn hình, `S` mở presenter view. [Nguồn nhập Canva](presentation/canva_import.html)
  và [dàn trang/ghi chú](presentation/outline.json) giữ nội dung chính xác để tái lập.

## Kết quả

12 lần fit CatBoost, cùng ba fold validation 09–11/2022, cùng 1.003 khách được
chọn mỗi tháng. Recall@1% trung bình: **kết hợp 9,74%**, hành vi bỏ metadata độ
phủ theo lịch 8,89%, hành vi gốc 8,70%, lịch sử mua 6,44%, rule tháng trước 6,18%.
Kết hợp − lịch sử mua **+3,30 điểm phần trăm**, bootstrap 95% **[+2,12; +4,28]**.
Đây là kết quả validation phục vụ phát triển. Test 01/2023 chưa mở.

## Bằng chứng công bố

| File | Phạm vi |
| --- | --- |
| `summary.csv` | Metric trung bình theo phương pháp |
| `monthly_budget_metrics.csv` | Precision/Recall/Lift từng tháng tại top 0,5%/1%/5% |
| `monthly_metrics.json` | AP, ROC-AUC, xác suất gốc và metric ngân sách từng tháng |
| `paired_comparisons.json` | 500 lần paired bootstrap theo khách, bốn đối chứng |
| `feature_contract.json` | Tên predictor từng bộ biến, tham số và protocol |
| `verification.json` | Hash số liệu công bố và tái lập baseline gốc khớp ba tháng |
| `publication_checks.json` | Kiểm tra PDF/HTML/Canva và kiểm thử mã nguồn |

Chỉ có số liệu tổng hợp ở thư mục này. Dự đoán theo khách, model từng fold và
nguồn dữ liệu ở `outputs/`/`data/`, không đưa lên Git. Các thanh biểu đồ và bảng
trong Canva nhập từ HTML tĩnh, không liên kết trực tiếp với CSV. Khi thay số liệu,
cập nhật nguồn và tạo lại bản trình bày thay vì coi chart là một liên kết dữ liệu.

## Tái lập

Chạy `scripts/run_purchase_ablation.py --config configs/propensity_ablation.json`
bằng Python >=3.11 và extra `baseline,plots`, với output mới chưa tồn tại. Cần
calendar feature cục bộ; clone Git không chứa dữ liệu/model của run thật.
`scripts/publish_baseline_ablation.py` xuất báo cáo và kiểm tra parity/hash.
`scripts/build_baseline_pdf.py` cần `reportlab`, `pypdf` và Segoe UI trên Windows.
`scripts/build_baseline_presentation.py` dùng skill html-ppt cục bộ. Script
`render_baseline_presentation.mjs` dùng Node + Playwright Core và Edge để render,
kiểm tra bố cục và phím điều hướng.

Layout dựa trên template weekly-report và table của html-ppt, MIT, tác giả
lewis. Bản quyền và giấy phép đi kèm trong `presentation/LICENSE.txt`.
