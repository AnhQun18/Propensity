# So sánh MBD và MBD-mini

Ngày nghiên cứu: **04/11/2026**.

**Kết luận:** MBD-mini là tập con khách hàng của MBD, giữ cùng bài toán dự đoán bốn sản phẩm. Tuy nhiên, khác biệt thực tế không chỉ là dung lượng: **tên cột nhãn khác, fold được chia lại, mini thiếu `is_balanced`, và số mẫu dương ít hơn nhiều**. Bảng nhãn của hai bản có cùng 12 mốc tháng.

## 1. Cách kiểm chứng

- Đọc dataset card, danh sách file Hugging Face và paper phiên bản v2 (02/06/2025).
- Tải và đọc **toàn bộ** `client_split.tar.gz` và `targets.tar.gz` của MBD, tổng 101.504.526 byte, khoảng 101,5 MB.
- So khớp với `client_split` và `targets` của MBD-mini đang có trong dự án bằng `client_id` và `mon`.
- Kiểm tra schema của toàn bộ 3.820 file Parquet mini. Số sự kiện và khoảng timestamp mini lấy từ audit đã lưu ngày 03/11/2026 trong `docs/propensity_feasibility/mbd_summary.json`.
- Không tải hai archive lịch sử sự kiện đầy đủ, nên **chưa đối chiếu từng sự kiện, mã categorical hay độ dài lịch sử từng khách giữa hai bản**.

Revision Hugging Face đã kiểm tra:

| Dataset | Revision |
|---|---|
| MBD | `7c1642ca66eb460b0a2097e63afbed2eece739fa` |
| MBD-mini | `4b0d5b4c0b014e02af6c2e6b7b27851e6fb0d5c3` |

Revision mini là metadata online tại ngày kiểm tra; không phải khẳng định đã kiểm tra checksum toàn bộ bản mini trên máy.

## 2. Quy mô và thời gian

| Tiêu chí | MBD | MBD-mini | Cơ sở |
|---|---:|---:|---|
| Khách hàng trong `client_split` | 1.000.000 | 100.224 | Đếm trực tiếp |
| Khách hàng có trong bảng `targets` | 1.000.000 | 100.224 | Đếm trực tiếp |
| Dòng nhãn khách hàng–tháng | 12.000.000 | 1.202.688 | Đếm trực tiếp |
| Số mốc `mon` | 12 | 12 | Đọc trực tiếp |
| `mon` đầu tiên | 2022-02-28 | 2022-02-28 | Đọc trực tiếp |
| `mon` cuối cùng | 2023-01-31 | 2023-01-31 | Đọc trực tiếp |
| Số fold | 5 | 5 | Đọc trực tiếp |
| Tổng dung lượng bốn archive tải xuống | 68,972 GB | 3,378 GB | Metadata Hugging Face; GB thập phân |

Mini chứa **10,0224%** số khách hàng trong bảng chia tập của MBD. Toàn bộ ID mini đều có trong MBD; không có ID ngoài tập. Tác giả mô tả mini là khoảng 10% khách hàng trong [dataset card MBD-mini](https://huggingface.co/datasets/ai-lab/MBD-mini).

Không nhầm số khách hàng có nhãn với quy mô toàn bộ nguồn sự kiện. [Paper v2, mục 3.1](https://arxiv.org/html/2409.17587v2#S3.SS1) mô tả 2.186.230 khách hàng trong nguồn tổng thể, trong đó khoảng 1 triệu có nhãn. Các số sự kiện công bố tương ứng với phạm vi lớn hơn đó:

| Nguồn sự kiện | MBD, số công bố trong paper | Mini trên máy, số audit |
|---|---:|---:|
| Giao dịch | 947.899.612 | 38.764.930 |
| Vị trí | 1.117.213.760 | 66.295.724 |
| Hội thoại | 5.080.781 | 217.205 |

Các số full trên là **số của paper**, chưa được kiểm đếm lại trên archive sự kiện của revision hiện tại. Vì phạm vi khách hàng khác nhau, không kỳ vọng mỗi bảng mini bằng đúng 10% số dòng full.

**Không có bằng chứng mini bị rút ngắn số tháng nhãn:** cả hai bảng nhãn thực tế có cùng 12 mốc tháng. Dataset card nói mini có khoảng thời gian ngắn hơn nhưng không nêu cutoff cụ thể. Với lịch sử sự kiện, mini trên máy có giao dịch từ 31/12/2020 đến 31/12/2022, vị trí từ 11/12/2021 đến 31/12/2022 và hội thoại từ 10/02/2021 đến 31/12/2022. Đây là timestamp lưu trong file, chưa xác nhận timezone. Chưa đủ bằng chứng để kết luận lịch sử từng khách của mini được giữ nguyên hoặc đã cắt ngắn so với full.

## 3. Khác biệt schema: tên nhãn và `is_balanced`

**Dataset card full không phản ánh đúng tên cột nhãn của file tải xuống.** Card ghi `target_1`–`target_4`, nhưng Parquet trong `targets.tar.gz` có các tên sau:

| Tên nhãn thực tế trong MBD | Tên tương ứng trong mini | Số giá trị khác nhau trên 1.202.688 khách hàng–tháng chung |
|---|---|---:|
| `bcard_target` | `target_1` | 0 |
| `cred_target` | `target_2` | 0 |
| `zp_target` | `target_3` | 0 |
| `acquiring_target` | `target_4` | 0 |

Mapping này được xác minh bằng giá trị của **từng khách hàng–tháng**, không suy đoán theo thứ tự cột. Các cặp nhãn khác đều có sai khác, nên mapping là duy nhất trong bốn cột đã kiểm tra. Ngoài ra, `trans_count` và `diff_trans_date` cũng khớp hoàn toàn trên tập chung, kể cả vị trí null. Điều này chưa xác nhận hai cột phụ trợ an toàn để dùng làm feature tại cutoff.

Tên cột gốc là thông tin bổ sung hữu ích, nhưng chưa đủ để xác nhận tên thương mại, điều kiện phát hành hoặc định nghĩa nghiệp vụ đầy đủ của sản phẩm.

MBD có `is_balanced` trong bảng khách hàng và nhãn; toàn bộ schema Parquet mini được kiểm tra không có trường này. [Card MBD](https://huggingface.co/datasets/ai-lab/MBD) hướng dẫn lọc `is_balanced == 1` để lấy mẫu cân bằng hơn.

Kết quả đọc file hiện tại:

- Full: **70.532 khách hàng** có `is_balanced = 1`; **929.468** khách hàng có giá trị 0.
- Trong 100.224 khách hàng mini, có **7.135** người thuộc nhóm `is_balanced = 1` của full và **93.089** thuộc nhóm 0.
- Vì vậy **MBD-mini không đồng nhất với tập lọc `is_balanced = 1`**.

Paper v2 mô tả một bản **MBD-small** gồm **69.474 khách hàng**, lấy theo nhãn dương tháng cuối và lấy mẫu nhóm toàn nhãn 0. Đây là một khái niệm khác với MBD-mini. Số 69.474 trong paper cũng không trùng 70.532 quan sát từ cờ của bản công khai hiện tại; chưa xác định nguyên nhân chênh lệch. Xem [paper v2, mục 3.1](https://arxiv.org/html/2409.17587v2#S3.SS1).

## 4. Khác biệt fold: không thể dùng chung split một cách máy móc

Đối chiếu theo ID cho thấy **80.268/100.224 khách hàng**, tương ứng **80,0886%**, có số fold khác nhau. Bảng chéo dưới đây cho thấy sự khác biệt không chỉ là hoán đổi tên fold:

| Fold mini / fold MBD | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| 0 | 4.048 | 4.112 | 4.072 | 4.038 | 4.058 |
| 1 | 4.081 | 3.882 | 3.914 | 4.061 | 3.964 |
| 2 | 4.030 | 4.029 | 3.955 | 3.939 | 3.964 |
| 3 | 3.951 | 4.030 | 4.015 | 3.999 | 4.022 |
| 4 | 3.943 | 3.991 | 4.123 | 3.931 | 4.072 |

Hệ quả cho thực nghiệm:

- Chạy cùng quy tắc `fold == 0` trên hai bản sẽ không tạo cùng nhóm khách hàng test.
- Mini nằm hoàn toàn trong tập khách hàng có nhãn của full. Nếu train trên full rồi test trên mini mà chưa loại ID trùng, test sẽ chứa khách hàng mô hình đã thấy.
- Để đo lợi ích của tăng dữ liệu, cần cố định một tập test theo ID và thời gian, sau đó thay đổi lượng dữ liệu train. Không so hai điểm số với hai cách chia khác nhau rồi quy toàn bộ chênh lệch cho kích thước dữ liệu.

## 5. Phân bố nhãn: gần nhau nhưng mini ít mẫu dương

Tỷ lệ dưới đây tính trên **toàn bộ dòng khách hàng–tháng**, không phải tỷ lệ khách có ít nhất một lần mua trong năm.

| Nhãn mini | Mẫu dương MBD | Tỷ lệ MBD | Mẫu dương mini | Tỷ lệ mini |
|---|---:|---:|---:|---:|
| `target_1` | 53.774 | 0,4481% | 5.362 | 0,4458% |
| `target_2` | 5.986 | 0,0499% | 577 | 0,0480% |
| `target_3` | 44.980 | 0,3748% | 4.530 | 0,3767% |
| `target_4` | 30.389 | 0,2532% | 3.060 | 0,2544% |

Tỷ lệ nhãn dương tổng thể gần nhau; đó chưa phải bằng chứng mọi phân bố feature, thời gian và nhóm khách hàng đều tương đương. Mini vẫn mất cân bằng rất mạnh. Riêng `target_2` chỉ có 577 dòng dương trên cả 12 tháng; khi tách theo tháng/fold, số dương dùng đánh giá có thể rất nhỏ.

Suy luận thực nghiệm: full cung cấp nhiều mẫu dương hơn để huấn luyện và đánh giá, nhưng không bảo đảm tự động tăng chất lượng mô hình. Nên báo cáo chỉ số riêng từng nhãn, PR-AUC và Lift/Recall@K bên cạnh ROC-AUC, đồng thời xem độ dao động giữa các split phù hợp với mục tiêu triển khai.

## 6. Dung lượng và chi phí xử lý

| Archive | MBD | MBD-mini |
|---|---:|---:|
| `client_split.tar.gz` | 39,347 MB | 4,183 MB |
| `targets.tar.gz` | 62,157 MB | 7,932 MB |
| `detail.tar.gz` | 38,153 GB | 1,969 GB |
| `ptls.tar.gz` | 30,718 GB | 1,397 GB |
| Tổng bốn archive | 68,972 GB | 3,378 GB |

Nguồn: danh sách file [MBD](https://huggingface.co/datasets/ai-lab/MBD/tree/main) và [MBD-mini](https://huggingface.co/datasets/ai-lab/MBD-mini/tree/main), đã lưu số byte và revision trong `audit.json`.

Full lớn hơn khoảng **20,42 lần về dung lượng archive**. Đây không phải hệ số RAM hoặc thời gian train: các đại lượng đó còn tùy cột đọc, cách tổng hợp, độ dài chuỗi và mô hình. `detail` và `ptls` là hai cách tổ chức dữ liệu; không cộng chúng thành số sự kiện độc lập. Nếu pipeline chỉ dùng một dạng, có thể tải riêng archive tương ứng.

## 7. Áp dụng cho dự án propensity này

**Dùng mini để hoàn thiện pipeline trước:** chuẩn hóa cutoff/nhãn, tạo feature lịch sử, thử baseline, đánh giá theo thời gian và kiểm tra leakage. Mini đã có hơn 1,2 triệu dòng nhãn và lịch sử tổng thể đủ dài để nghiên cứu cửa sổ 84 ngày; vẫn phải kiểm tra độ phủ từng khách tại cutoff.

**Chuyển sang full khi cần nhiều mẫu dương hoặc đo hiệu quả theo quy mô.** Khi chuyển cần:

1. Đổi tên bốn cột nhãn theo mapping đã xác minh.
2. Chọn rõ tập full hoặc tập lọc `is_balanced`, không đồng nhất tập lọc với mini.
3. Giữ một định nghĩa train/validation/test thống nhất theo ID và thời gian.
4. Kiểm tra schema và mapping categorical của lịch sử sự kiện trước khi tái sử dụng encoder/model đã fit trên mini.
5. Đánh giá lại phân bố điểm và calibration nếu dùng đầu ra như xác suất; không suy ra xác suất ngoài mẫu từ tỷ lệ trong tập đã lấy mẫu cân bằng.

Hai bản giữ cùng loại thông tin chính và cùng mục tiêu bốn nhãn theo [card MBD](https://huggingface.co/datasets/ai-lab/MBD) và [card mini](https://huggingface.co/datasets/ai-lab/MBD-mini). Những gì được kiểm chứng chính xác ở đây là khách hàng, bảng nhãn, cột phụ trợ trong bảng nhãn và cách chia fold; tương đương từng sự kiện lịch sử vẫn chưa được xác minh.

## 8. Tệp kết quả và tái lập

- `docs/mbd_comparison/audit.json`: số đếm, tỷ lệ, mapping nhãn, bảng chéo fold, schema mini và revision nguồn.
- `src/research_mbd_comparison.py`: script đọc archive trực tiếp trong bộ nhớ, không giải nén đường dẫn từ archive; chỉ xuất số liệu tổng hợp.
- `data/research_mbd_sources/`: hai archive MBD đã tải, nằm trong vùng dữ liệu được `.gitignore` loại trừ.

Chạy lại từ thư mục dự án, với quyền truy cập mạng:

```powershell
.\.venv\Scripts\python.exe -X utf8 src/research_mbd_comparison.py
```

Chưa huấn luyện mô hình và chưa đo mức cải thiện chất lượng giữa mini và full.
