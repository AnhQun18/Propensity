# Khả năng sử dụng 3 bộ dữ liệu cho bài toán propensity

Ngày kiểm tra: 03/11/2026. Phạm vi: MBD-mini, eCommerce và Fliggy trong thư mục `data/` của dự án. Đây là kiểm tra dữ liệu và thiết kế bài toán; chưa huấn luyện mô hình và chưa có điểm dự đoán hay chỉ số chất lượng mô hình.

## Kết luận và đầu ra

Propensity trong báo cáo được hiểu là xác suất một khách hàng thực hiện hành vi xác định trong một khoảng tương lai, dựa trên dữ liệu đã có tại thời điểm chấm điểm:

`p(u,k,t,H) = P(có hành vi k trong [t, t+H) | lịch sử trước t)`.

| Bộ dữ liệu | Mức độ sử dụng | Bài toán phù hợp | Nhãn huấn luyện | Đầu ra mô hình sau khi huấn luyện |
|---|---|---|---|---|
| MBD-mini | Phù hợp nhất trong 3 bộ cho propensity sản phẩm ngân hàng, có nhãn sẵn | Dự đoán mua từng sản phẩm trong tháng đích | `target_1`–`target_4`, mỗi cột 0/1 | Mỗi khách hàng tại một cutoff có 4 điểm xác suất: `p_target_1`–`p_target_4` |
| eCommerce | Phù hợp cho propensity mua hàng trong thời gian ngắn | Người dùng có phát sinh `purchase` trong 7 ngày tới không? | Tự tạo: có ít nhất một log `purchase` trong khoảng nhãn = 1; còn lại = 0 trong cohort đã chọn | Một điểm `p_purchase_7d` cho mỗi người dùng tại cutoff |
| Fliggy | Phù hợp có điều kiện; có thể bắt đầu với nhãn tương tác | Người dùng có quay lại tương tác trong 7 ngày tới không? | Tự tạo: có ít nhất một log hành vi đã chọn trong khoảng nhãn = 1 | Một điểm `p_activity_7d`; chỉ tạo `p_purchase_7d` khi xác minh được mã hành vi mua |

Bốn điểm MBD là bốn xác suất nhị phân riêng: khách có thể mua nhiều sản phẩm, nên tổng bốn điểm không bắt buộc bằng 1. Tên nghiệp vụ cụ thể của bốn sản phẩm chưa được xác định; giữ nguyên `target_1`–`target_4`.

Nếu mục tiêu thực tế là dự đoán **hoạt động theo ngày**, cả ba bộ đều có ID và thời gian để dựng ma trận lịch sử khách hàng × ngày. Khi đó, đầu vào có thể là chuỗi nhị phân theo ngày; đầu ra có thể là một xác suất có hoạt động trong 7 ngày, hoặc 7 xác suất cho từng ngày nếu thiết kế mô hình và nhãn theo ngày. Hai thiết kế này khác nhau: không tự cộng các xác suất theo ngày để có xác suất hoạt động trong cả khoảng. Ma trận lịch sử 0/1 là dữ liệu đầu vào, chưa phải kết quả propensity.

## Bằng chứng từ dữ liệu thực tế

### MBD-mini

- `client_split`: **100.224 khách hàng**, mỗi ID một fold; không có khách xung đột fold.
- `targets`: **1.202.688 dòng**, đủ **12 tháng × 100.224 khách hàng**, từ `2022-02-28` đến `2023-01-31`. Không trùng khóa khách hàng–tháng, không thiếu nhãn; toàn bộ khách và fold khớp `client_split`.
- `detail/trx`: **38.764.930 sự kiện**, thời gian lưu trong file từ **2020-12-31 đến 2022-12-31**.
- `detail/geo`: **66.295.724 sự kiện**, từ **2021-12-11 đến 2022-12-31**.
- `detail/dialog`: **217.205 sự kiện**, từ **2021-02-10 đến 2022-12-31**. Timestamp của ba bảng không có null; chưa xác nhận timezone của timestamp không kèm múi giờ.
- Các bảng `ptls/trx`, `ptls/geo`, `ptls/dialog` lần lượt có **98.721**, **72.573**, **46.320** dòng chuỗi. Nguồn mô tả `ptls` là biểu diễn chuỗi của dữ liệu; không cộng số dòng này với `detail` thành số sự kiện mới. Chưa đối chiếu tương đương từng sự kiện hoặc từ điển mã hóa giữa hai dạng.

| Nhãn | Số dòng dương | Tỷ lệ trên khách hàng–tháng |
|---|---:|---:|
| target_1 | 5.362 | 0,4458% |
| target_2 | 577 | 0,0480% |
| target_3 | 4.530 | 0,3767% |
| target_4 | 3.060 | 0,2544% |

Điều kiện sử dụng:

1. Nhãn mất cân bằng mạnh. Riêng `target_2` có rất ít mẫu dương theo tháng và theo fold, nên kết quả có thể dao động. Không dùng accuracy làm tiêu chí chính.
2. Card dữ liệu mô tả dự đoán trong tháng sau ngày báo cáo; paper giải thích bảng nhãn đã dịch khoảng ngày một tháng. Trước khi tạo mẫu, phải đối chiếu cách dùng `mon` với pipeline benchmark để xác định chính xác tháng nhãn và cutoff; không lấy tùy tiện `mon` làm mốc cuối phần feature. Nguyên tắc là mọi sự kiện dùng làm feature phải có trước đầu khoảng nhãn.
3. Chưa xác minh `trans_count` và `diff_trans_date` trong bảng targets được tính đến thời điểm nào; ban đầu chỉ lấy ID, tháng, fold và nhãn từ bảng này, tự tạo feature từ lịch sử sự kiện. `diff_trans_date` có **120.786 giá trị thiếu**.
4. Timeline chung đủ để thử lịch sử 84 ngày, nhưng không bảo đảm từng khách có đủ lịch sử hoặc đủ cả ba nguồn. Điều kiện cohort phải được kiểm tra tại từng cutoff, chỉ từ phần quá khứ.

Mô tả nhãn được đối chiếu với [dataset card MBD-mini](https://huggingface.co/datasets/ai-lab/MBD-mini) và [paper MBD, mục 2.1 và 3.2.1](https://arxiv.org/html/2409.17587v1). Các số đếm trong báo cáo này lấy từ bản mini đang có trên máy, không lấy quy mô bộ MBD đầy đủ trong paper.

### eCommerce

- Hai file tháng 10 và 11/2019 có **109.950.743 sự kiện**, **5.316.649 người dùng**.
- Thời gian: **01/10–30/11/2019 UTC**, **61 ngày**, ngày nào cũng có log.
- `view`: **104.335.509**; `cart`: **3.955.446**; `purchase`: **1.659.788**. Có **697.470 người dùng** có ít nhất một log purchase trong hai tháng.
- Không thiếu `event_time`, `event_type`, `user_id` trong toàn bộ dữ liệu đã đọc. `category_code` thiếu **35.413.780 dòng**, khoảng **32,21%**; không loại toàn bộ những dòng này khi xây bài toán mua bất kỳ.

Đã kiểm tra một cửa sổ cụ thể, cohort gồm người có ít nhất một log trong lịch sử:

| Thành phần | Kết quả |
|---|---|
| Lịch sử | 03–23/11/2019 UTC, 21 ngày |
| Khoảng nhãn | 24–30/11/2019 UTC, 7 ngày |
| Người đủ điều kiện | 3.012.913 |
| Người có purchase trong khoảng nhãn | 71.587 |
| Tỷ lệ nhãn dương | 2,3760% |
| Người có bất kỳ log tương lai trong cohort | 712.448 |

Đây là tỷ lệ quan sát, không phải xác suất dự đoán của mô hình. Nhãn 0 nghĩa là không thấy log mua trong khoảng đã chọn; chưa chứng minh người dùng được quan sát đầy đủ hoặc không mua ở nơi khác. Một log purchase chưa được xác minh là đơn hàng duy nhất, thanh toán thành công cuối cùng hay doanh thu sau hoàn/hủy.

Có thể dùng feature recency, số lượt view/cart/purchase và số ngày hoạt động trong các cửa sổ trước cutoff. Bài toán theo danh mục/sản phẩm cần kiểm tra thêm độ phủ ID và định nghĩa tập ứng viên; không mặc định mọi sản phẩm người dùng chưa mua là một mẫu âm đã được tiếp xúc.

### Fliggy

- **42.433.919 sự kiện**, **2.524.995 người dùng**, **34 ngày** từ **16/08–18/09/2022** theo giả định lịch **UTC+8**. Đây là giả định timezone, chưa phải xác nhận từ nguồn.
- **59,46%** người dùng chỉ hoạt động đúng một ngày; trung vị số ngày hoạt động là **1**. Nhiều dòng log không đồng nghĩa lịch sử dài cho mỗi khách.
- Log có ba mã **1/2/3**, nhưng chưa xác minh được bảng quy đổi sang hành vi nghiệp vụ. Chưa đủ cơ sở đặt tên nhãn là mua hàng.
- Kiểm tra lịch sử **22/08–11/09** và nhãn **12–18/09**: **1.673.450** người có lịch sử, **345.405** người có log trong 7 ngày sau, tỷ lệ **20,6403%**. Tất cả mã được tính vào định nghĩa hoạt động trong phép kiểm tra này.
- Hồ sơ phương án có khóa lặp; theo cách đọc khóa hiện tại chỉ **22,45% dòng log** có plan ID trong bảng profile. Cần xác nhận schema và chuẩn hóa khóa trước khi bổ sung feature hành trình/giá. Không inner join hoặc join một khóa chưa duy nhất tùy tiện.

Có thể bắt đầu từ log để dự đoán hoạt động với lịch sử 7 ngày và khoảng nhãn 7 ngày. Thử 14/21 ngày lịch sử là khả thi cho một số cửa sổ, nhưng sẽ hạn chế số khoảng train/validation/test có nhãn không chồng nhau. Chi tiết kiểm tra nằm trong [báo cáo Fliggy](../fliggy_140721_profile/assessment.md).

## Nếu yêu cầu là lịch sử 84 ngày

| Bộ dữ liệu hiện có | Khả năng đáp ứng |
|---|---|
| MBD-mini | Có timeline đủ dài để tạo cửa sổ 84 ngày và tương lai; cần chọn cutoff phù hợp và kiểm tra lịch sử từng khách |
| eCommerce | Không đủ: toàn bộ bản tải hiện tại chỉ có 61 ngày, chưa tính khoảng nhãn tương lai |
| Fliggy | Không đủ: chỉ có 34 ngày |

Không điền các ngày nằm ngoài phạm vi dữ liệu bằng 0 để giả lập lịch sử 84 ngày. Việc đó sẽ biến thiếu dữ liệu thành không hoạt động.

## Output nên bàn giao khi xây mô hình

1. **Bảng dữ liệu huấn luyện**: một dòng cho mỗi khách hàng–cutoff, có feature chỉ từ quá khứ, nhãn tương lai, thời điểm bắt đầu/kết thúc nhãn và split. MBD có bốn nhãn; eCommerce/Fliggy theo thiết kế trên có một nhãn.
2. **Bảng chấm điểm** theo schema chung dưới đây. Dạng dài cho phép mỗi khách có nhiều mục tiêu; MBD tạo bốn dòng, hai bài toán còn lại tạo một dòng tại mỗi cutoff.
3. **Bảng đánh giá** theo tháng/cutoff và từng target: PR-AUC, ROC-AUC, Precision/Recall/Lift@top 5% hoặc 10%, so với baseline recency/frequency; kiểm tra calibration nếu sử dụng điểm như xác suất. Không có số đo này trước khi huấn luyện và kiểm thử.

| Cột output chấm điểm | Ý nghĩa |
|---|---|
| `dataset` | `mbd_mini`, `ecommerce`, `fliggy` |
| `customer_id` | ID nội bộ của chính bộ dữ liệu đó |
| `as_of_time` | Cutoff: thời điểm chấm điểm |
| `label_start`, `label_end_exclusive` | Khoảng tương lai được dự đoán; tránh mơ hồ giữa tháng lịch và 30 ngày |
| `target` | `target_1`…`target_4`, `purchase`, hoặc `activity` |
| `propensity_score` | Điểm xác suất trong [0,1]; cần kiểm tra calibration |
| `rank_percentile` | Thứ hạng trong cùng dataset, target, cutoff và cohort |
| `model_version` | Phiên bản mô hình để tái lập kết quả |

Ngưỡng chia cao/vừa/thấp hoặc danh sách top-K được xác định từ validation và ngân sách chiến dịch, chưa có cơ sở ấn định 0,5 hay các ngưỡng tùy ý. Đây là xác suất hành vi quan sát được, chưa phải mức tăng xác suất do một chiến dịch gây ra; chưa xác minh được dữ liệu treatment/control để xây bài toán uplift.

Khuyến nghị triển khai: dùng **MBD-mini** nếu mục tiêu là propensity sản phẩm với lịch sử dài; dùng **eCommerce** nếu muốn bắt đầu với nhãn mua dễ diễn giải và cửa sổ ngắn; dùng **Fliggy** để thử propensity hoạt động, rồi mở rộng sang mua khi có mapping hành vi. Có thể dùng chung thiết kế pipeline, nhưng phải tạo nhãn, huấn luyện và đánh giá riêng từng miền. Không có bằng chứng ID của ba bộ liên thông để ghép khách hàng hoặc gộp chúng thành cùng một bảng hành vi.

## Phạm vi kiểm chứng và cách chạy lại

- MBD: kiểm tra footer toàn bộ 3.820 file Parquet; đọc toàn bộ timestamp của ba bảng detail vì footer INT96 không chứa min/max; đọc đầy đủ targets và client_split để kiểm tra nhãn, khóa, fold. Chưa quét toàn bộ giá trị các thuộc tính sự kiện, trùng log hay lịch sử từng người.
- eCommerce: quét toàn bộ hai CSV nhưng chỉ phân tích bốn cột `event_time`, `event_type`, `user_id`, `category_code`; đối chiếu tổng dòng theo hành vi và ngày; tính một cửa sổ nhãn. Chưa kiểm tra trùng nguyên dòng, session, product, brand hoặc price.
- Fliggy: chạy lại hai script có sẵn để kiểm tra toàn bộ log, hồ sơ khách và khóa hồ sơ phương án; chưa xác minh mapping nghiệp vụ và chưa kiểm tra trùng nguyên log. Đã cập nhật đường dẫn script theo thư mục hiện tại `data/Fliggy`.
- Không thay đổi dữ liệu nguồn, không xuất ID khách hàng vào báo cáo mới, không huấn luyện mô hình.

```powershell
.\.venv\Scripts\python.exe -X utf8 src/audit_propensity_feasibility.py mbd
.\.venv\Scripts\python.exe -X utf8 src/audit_propensity_feasibility.py ecommerce
.\.venv\Scripts\python.exe -X utf8 src/profile_fliggy_140721.py
.\.venv\Scripts\python.exe -X utf8 src/check_fliggy_profile_links.py
```

Kết quả kèm theo: `mbd_summary.json`, `mbd_monthly.csv`, `ecommerce_summary.json`, `ecommerce_daily.csv`; kết quả Fliggy nằm trong `docs/fliggy_140721_profile/`.
