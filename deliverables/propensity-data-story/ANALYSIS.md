# Phân tích bổ sung sau rà soát deck

Tất cả thí nghiệm dưới đây chỉ dùng validation 09–11/2022, cùng 100.224 ID/tháng. Test 01/2023 và calibration không mở. Cấu hình cố định depth 4, 200 cây, learning rate 0,04, L2 5, seed 20261110.

## Bỏ từng nguồn trên cùng nền lịch sử mua

9 fit mới, full B+H 156 biến tái sử dụng model đã lưu sau xác minh hash. ΔAP = AP full − AP reduced; đóng góp có điều kiện theo các nguồn còn lại, không cộng thành tổng.

| Phương án | AP macro | Recall@1% macro | ΔAP | Giảm recall (đpt) |
|---|---:|---:|---:|---:|
| combined | 0,0455 | 9,74% | 0,000000 | 0,00 |
| without_dialog | 0,0372 | 7,56% | 0,008304 | 2,18 |
| without_geo | 0,0451 | 9,55% | 0,000344 | 0,19 |
| without_trx | 0,0372 | 8,59% | 0,008327 | 1,15 |

Geo có chênh nhỏ và đổi dấu ở tháng 10; chưa có kiểm định độ ổn định trên nhiều seed hoặc tháng độc lập. Trx và dialog có ΔAP dương ở cả ba tháng. Không suy nhân quả từ ablation này.

## Cùng bộ biến, khác độ dài cửa sổ

12 fit mới, cùng 16 thống kê hành vi + 15 biến H. Chỉ thay giá trị cửa sổ; cutoff, cohort, split và cấu hình giữ nguyên. Không so trực tiếp kết quả bộ 31 biến này thành tác dụng cửa sổ cho full 156 biến.

| Cửa sổ | AP macro | Recall@1% macro |
|---|---:|---:|
| window_30d | 0,0377 | 7,84% |
| window_84d | 0,0384 | 7,73% |
| window_all | 0,0406 | 9,31% |
| window_recent | 0,0380 | 8,34% |

Nhãn tháng không xác định được nhãn mua trong 30 ngày sau một ngày T bất kỳ. Có thể tạo feature tại T; đổi horizon cần nhãn tương thích và backtest riêng.

## Lỗi của B+H mới nhất

| Nhóm | Dương | Được chọn | TP | FP | Recall nhóm | Precision nhóm |
|---|---:|---:|---:|---:|---:|---:|
| bought_lag1 | 180 | 1116 | 92 | 1024 | 51,11% | 8,24% |
| has_dialog_history | 1856 | 2895 | 255 | 2640 | 13,74% | 8,81% |
| has_geo_history | 2110 | 2295 | 209 | 2086 | 9,91% | 9,11% |
| no_dialog_history | 892 | 114 | 11 | 103 | 1,23% | 9,65% |
| no_geo_history | 638 | 714 | 57 | 657 | 8,93% | 7,98% |
| no_purchase_history | 2016 | 1291 | 141 | 1150 | 6,99% | 10,92% |
| not_bought_lag1 | 2568 | 1893 | 174 | 1719 | 6,78% | 9,19% |
| prior_purchase_history | 732 | 1718 | 125 | 1593 | 17,08% | 7,28% |

Các nhóm có chồng lấn. Top-K được chọn trên toàn cohort rồi mới phân nhóm; gộp TP/dương qua ba tháng, khác macro đều tháng. Thiếu dialog là trạng thái log trước cutoff, không tự chứng minh lỗi thu thập.

## Giới hạn dữ liệu cần giữ đúng

- Nhãn nhị phân chỉ đo tháng dương, không đếm số đơn hoặc số lần mua trong tháng.
- Không có text, duration hoặc độ dài hội thoại; embedding 768D chưa được dùng trong các thí nghiệm này.
- Mã geo ẩn danh không cho phép suy quãng đường thật. Đổi mã nhanh là proxy để kiểm tra, chưa phải chứng cứ nhiễu GPS.
- Khoảng trx trước 02/2022 chưa có target là lịch sử; SLA công bố nhãn thực tế chưa xác minh.

Chi tiết số liệu: CSV trong analysis/. Đối chiếu đề cương: coverage_ledger.md.
