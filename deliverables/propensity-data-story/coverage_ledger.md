# Đối chiếu đề cương với slide và bằng chứng

Bản chính giữ 19 slide; bản chi tiết có 23 slide theo A3 + B3 + C11 + D3 + E3. Số phụ lục dưới đây tính từ A1 của bản chính.

| Mục | Yêu cầu | Chính | Chi tiết | Phụ lục | Kết quả rà soát |
|---|---|---|---|---|---|
| A1 | Phát biểu bài toán và định nghĩa nhãn | 1 | 1 | — | Đã có |
| A2 | Lệch nhãn, quy mô, target đến muộn | 2 | 2 | 1 | Đã đo; SLA 6 tháng không xác minh |
| A3 | 3–4 câu hỏi dẫn dắt | 2 | 3 | 2 | Đã có |
| B1 | 4 nguồn: dòng, ID, cột, vai trò | 3 | 4 | — | Đã đo |
| B2 | Gantt và vùng trx chưa có target | 4 | 5 | — | Đã đo |
| B3 | UpSet tổ hợp đủ trx+geo+dialog | 5 | 6 | — | Đã đo |
| C.Target1 | Tỷ lệ dương theo tháng và cohort; drift | 6 | 7 | — | Bổ sung cohort cố định qua 10 tháng |
| C.Target2 | Số lần mua / heavy tail | — | 8 | 3 | Đo số tháng dương; không có số đơn trong nhãn |
| C.Target3 | P(mua T|mua T−1) và đối chứng | 7 | 9 | — | Đã tính đúng consecutive-month |
| C.Trx1 | Số dòng và ID active theo tháng | 8 | 10 | — | Sửa hai trục về đơn vị thực |
| C.Trx2 | Sparsity và khách im lặng lâu | — | 11 | 4 | Bổ sung đường 30/84d theo cutoff |
| C.Trx3 | R/F/M theo nhóm nhãn | 9 | 12 | 7, 8 | Bổ sung decile lift cả R/F/M, giữ ties |
| C.Trx4 | Treemap + lift danh mục / loại giao dịch | 10 | 13 | — | Bổ sung lift từng event_type |
| C.Trx5 | Kiểm tra leakage trước khi có target | — | — | 18 | Bổ sung số sự kiện trước nhãn; có giới hạn availability |
| C.Dialog1 | Độ phủ thời gian và khách | 11 | 14 | — | Có cả dòng, active ID, cấu trúc |
| C.Dialog2 | Metadata frequency / length / hour và lift | 12 | 15 | — | Bổ sung recency, giờ, weekend; không có độ dài text/duration |
| C.Dialog3 | Nội dung/embedding nếu đã thử | 12 | 15 | — | Điều kiện chưa phát sinh: chưa thử embedding, ghi rõ giới hạn |
| C.Geo1 | Độ phủ và mức nhiễu | — | 16 | 5 | Bổ sung quality proxy mẫu hash ID |
| C.Geo2 | Di chuyển / vùng và liên hệ nhãn | 13 | 17 | — | Đo đa dạng/tập trung mã vùng; không có tọa độ thật |
| C.Geo3 | Kết luận bổ trợ hay không đáng kể | 13 | 17 | — | Bổ sung AP loss khi bỏ geo |
| D1 | Importance / lift theo nguồn / ΔAP ablation | 14 | 18 | — | 9 fit mới; importance và từng tháng trong phụ lục |
| D2 | Baseline vs rule sai ở nhóm nào | 15 | 19 | — | Bổ sung B+H 156 biến, đúng model mới nhất |
| D3 | Snapshot tháng lịch / T bất kỳ / cửa sổ | 16 | 20 | — | 12 fit matched-window; T bất kỳ giới hạn bởi nhãn |
| E1 | Heatmap mức phù hợp sáu chiều | 17 | 21 | — | Đã có; đánh giá định tính theo bằng chứng |
| E2 | Rủi ro → xử lý | 18 | 22 | — | Đã có; không gán target thiếu thành âm |
| E3 | Mức phù hợp và 2–3 việc tiếp theo | 19 | 23 | — | Cập nhật theo đối chứng mới |
