# Kiểm tra khóa ghép, lịch sử và nhãn

Ngày kiểm tra: 05/11/2026. Đọc toàn bộ client_split, targets và các cột ID/thời gian cần thiết của detail. Không sửa dữ liệu gốc hoặc xuất ID. Kết quả đầy đủ: `join_questions_audit.json`; tái lập bằng `.\.venv\Scripts\python.exe -X utf8 src/audit_mbd_join_questions.py`.

## 1. Đơn vị quan sát và khóa ghép

100.224 khách có đúng 12 tháng nhãn mỗi khách, tổng 1.202.688 dòng, không trùng `(client_id, mon)`. Fold trong targets khớp client_split.

Ghép targets với bảng khách/feature tĩnh duy nhất theo client_id là many-to-one hợp lệ và giữ nguyên 1.202.688 dòng. Feature có cửa sổ thời gian phải duy nhất theo `(client_id, mon)` và ghép one-to-one trên cả hai cột. Ghép hai bảng cùng nhiều dòng mỗi client chỉ theo client_id sẽ nhân dòng.

Không tìm thấy đoạn mã tạo các thống kê cũ trong workspace. Phép tính mới dưới đây tái lập đúng trung vị −113 và khoảng −307 đến 759, nhưng không chứng minh đoạn ghép trước đây đã dùng đúng cardinality.

## 2. Thời gian giao dịch

Tổng 38.764.930 dòng giao dịch, 98.721 khách có giao dịch, 1.503 khách không có log giao dịch. Lấy giao dịch cuối trên toàn bộ lịch sử, rồi tính `floor((mon - last_trx_time) / 1 ngày)` cho từng khách–tháng có log, thu được trung vị −113 ngày. Đây là thống kê mô tả toàn lịch sử, không phải recency hợp lệ để huấn luyện.

“Sau mon” ở bảng dưới là `event_time > mon 00:00:00`; bao gồm sự kiện diễn ra muộn hơn trong chính ngày mon. Mẫu số khách mỗi tháng là 100.224; mẫu số giao dịch mỗi tháng là toàn bộ 38.764.930 dòng. Không cộng các giao dịch sau mon giữa các tháng vì một sự kiện có thể được đếm ở nhiều mốc.

| mon | Khách có giao dịch sau mon | % khách | Giao dịch sau mon | % giao dịch |
|---|---:|---:|---:|---:|
| 2022-02-28 | 96.616 | 96,40 | 16.582.219 | 42,78 |
| 2022-03-31 | 95.625 | 95,41 | 14.954.200 | 38,58 |
| 2022-04-30 | 94.540 | 94,33 | 13.413.981 | 34,60 |
| 2022-05-31 | 93.463 | 93,25 | 11.848.620 | 30,57 |
| 2022-06-30 | 92.252 | 92,05 | 10.229.164 | 26,39 |
| 2022-07-31 | 90.467 | 90,26 | 8.576.934 | 22,13 |
| 2022-08-31 | 88.774 | 88,58 | 6.954.549 | 17,94 |
| 2022-09-30 | 86.705 | 86,51 | 5.298.194 | 13,67 |
| 2022-10-31 | 83.716 | 83,53 | 3.529.474 | 9,10 |
| 2022-11-30 | 78.753 | 78,58 | 1.833.909 | 4,73 |
| 2022-12-31 | 27.538 | 27,48 | 49.510 | 0,13 |
| 2023-01-31 | 0 | 0,00 | 0 | 0,00 |

Để dựng feature, theo quy ước triển khai hiện có ở `mon_research.md`, cutoff là đầu tháng chứa mon. Lọc `event_time < cutoff` trước khi tổng hợp, rồi tính recency từ cutoff. Bảng trên chỉ trả lời câu hỏi sau mon, không đo toàn bộ sự kiện không được phép dùng làm input: cả sự kiện trong tháng nhãn trước mon cũng phải bị loại.

## 3. Vắng nguồn

27.651 khách không xuất hiện trong detail/geo (27,59%); 53.904 khách không xuất hiện trong detail/dialog (53,78%). Số khách xuất hiện lần lượt 72.573 và 46.320 khớp số dòng chuỗi ptls tương ứng trong audit trước. Điều này phù hợp với sự vắng nguồn trong dữ liệu được phát hành, nhưng chưa phân biệt được không hoạt động, không thuộc phạm vi thu thập hay thất thoát dữ liệu. Không có cờ theo dõi thu thập liên tục để kết luận nguyên nhân.

Khi triển khai feature, tính `has_geo`, `has_dialog` và count trên cùng cửa sổ trước cutoff. Sau left join, count bằng 0 khi không có sự kiện; các thống kê không xác định như trung bình, embedding và recency giữ NaN. Cờ chỉ biểu thị có log quan sát được, không xác nhận hoạt động thực tế. Không lấy cờ toàn lịch sử ở báo cáo này làm feature cho tháng quá khứ.

## 4. Độ phủ theo fold

Tỷ lệ dưới đây là không có log trong toàn bộ nguồn, chưa lọc thuộc tính hoặc cửa sổ trước cutoff.

| Fold | Khách | Không có trx (%) | Không có geo (%) | Không có dialog (%) |
|---|---:|---:|---:|---:|
| 0 | 20.328 | 1,46 | 26,88 | 53,75 |
| 1 | 19.902 | 1,53 | 27,60 | 54,02 |
| 2 | 19.917 | 1,33 | 27,72 | 53,72 |
| 3 | 20.017 | 1,60 | 28,16 | 54,23 |
| 4 | 20.060 | 1,59 | 27,59 | 53,20 |

Chênh lệch lớn nhất giữa các fold là 1,28 điểm phần trăm cho geo và 1,03 cho dialog; chưa thấy fold nào lệch lớn ở phép đo này. Đây chưa phải xác nhận tính hợp lệ của validation. Khi chốt cửa sổ feature cần kiểm tra lại theo fold × cutoff, phân bố nhãn và phân bố feature. Giữ mọi tháng của cùng khách trong cùng fold; đánh giá dự đoán tương lai cần thêm phép chia theo thời gian.

## 5. Nhãn

Mẫu số là 1.202.688 khách–tháng, không phải 100.224 khách duy nhất.

| Nhãn | Dương | Tỷ lệ dương (%) |
|---|---:|---:|
| target_1 | 5.362 | 0,4458 |
| target_2 | 577 | 0,0480 |
| target_3 | 4.530 | 0,3767 |
| target_4 | 3.060 | 0,2544 |

Có 620 dòng có đúng hai nhãn dương, 26 dòng có ba và 1 dòng có cả bốn: tổng 647 dòng đa nhãn. Giao cặp theo thứ tự 1–2, 1–3, 1–4, 2–3, 2–4, 3–4 là 49, 263, 205, 32, 24, 131; không cộng các giao cặp để tính số dòng đa nhãn.

Đây là bài toán multi-label; bốn mô hình nhị phân riêng là một cách triển khai hợp lệ (binary relevance). Đề xuất baseline bốn mô hình và báo cáo Average Precision từng nhãn cùng macro AP; thêm precision/recall/lift ở top-k phù hợp số khách có thể tiếp cận. ROC-AUC có thể báo cáo bổ sung. Không dùng accuracy làm metric chính. Định nghĩa AP: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html.

## 6. Đuôi phân bố

Với chênh lệch toàn lịch sử ở mục 2:

- −307 ngày: 27.538 dòng, 27.538 khách, xuất hiện tại mon=2022-02-28; không phải vài outlier.
- 759 ngày: 1 dòng của 1 khách, tại mon=2023-01-31.
- Lớn hơn 180 ngày: 34.251 dòng khách–tháng thuộc 8.414 khách duy nhất.
- Lớn hơn 365 ngày: 5.255 dòng khách–tháng thuộc 1.326 khách duy nhất.

Không tự lọc bỏ các khách này. Khi triển khai mô hình, tính lại recency hợp lệ theo từng cutoff, thêm nhóm không có lịch sử và có thể chia nhóm 0–30, 31–90, 91–180, trên 180 ngày. Các nhóm trong báo cáo dùng giao dịch cuối toàn lịch sử nên chỉ có vai trò chẩn đoán, không dùng để chọn cohort cho các tháng quá khứ.
