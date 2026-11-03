# Giới thiệu đề tài và ba bộ dữ liệu

## 1. Đề tài: dự đoán xu hướng hành vi khách hàng

Đề tài nghiên cứu bài toán **propensity**: từ lịch sử của một khách hàng, ước lượng khả năng người đó thực hiện một hành vi cụ thể trong tương lai. Hành vi có thể là mua sản phẩm ngân hàng, mua hàng trực tuyến hoặc quay lại tương tác với nền tảng du lịch.

Một câu hỏi propensity cần xác định rõ ba thành phần:

- **Đối tượng:** dự đoán cho khách hàng/người dùng nào?
- **Hành vi:** mua sản phẩm nào, mua hàng bất kỳ hay quay lại tương tác?
- **Thời gian:** hành vi có xảy ra trong 7 ngày tới, tháng tới hoặc một khoảng xác định khác không?

**Input** là lịch sử đã có trước thời điểm dự đoán, có thể gồm giao dịch, lượt xem, hội thoại, vị trí và hồ sơ khách hàng. **Output** là một hoặc nhiều điểm xác suất trong khoảng từ 0 đến 1, tương ứng với các hành vi cần dự đoán.

Ví dụ minh họa: với lịch sử xem và thêm hàng vào giỏ trước một ngày nhất định, mô hình trả lời câu hỏi “Người dùng này có khả năng mua hàng trong 7 ngày tiếp theo là bao nhiêu?”. Khoảng 7 ngày là một cách đặt bài toán, không phải quy định chung của cả ba bộ dữ liệu.

Repository đang chứa ba bộ dữ liệu thuộc ba lĩnh vực khác nhau:

| Bộ dữ liệu | Lĩnh vực | Dữ liệu chính | Đối tượng dự đoán |
|---|---|---|---|
| MBD-mini | Ngân hàng | Giao dịch, hoạt động vị trí, biểu diễn hội thoại và nhãn sản phẩm theo tháng | Khách hàng ngân hàng |
| eCommerce | Thương mại điện tử | Lượt xem, thêm giỏ hàng và mua hàng | Người dùng cửa hàng trực tuyến |
| Fliggy Transfer-Plans | Du lịch | Log hành vi với phương án di chuyển, hồ sơ người dùng và hồ sơ phương án | Người dùng nền tảng du lịch |

Các quy mô bên dưới mô tả **bản dữ liệu hiện có trong dự án**, dựa trên kết quả kiểm tra đã lưu trong repository.

## 2. MBD-mini — dữ liệu khách hàng ngân hàng

### Bối cảnh và nội dung

[MBD-mini](https://huggingface.co/datasets/ai-lab/MBD-mini) là bản rút gọn của Multimodal Banking Dataset. Bộ dữ liệu được ẩn danh và kết hợp nhiều nguồn thông tin về cùng khách hàng: giao dịch, hoạt động vị trí và hội thoại với ngân hàng. Ngoài lịch sử hành vi, bộ này có sẵn nhãn liên quan đến việc mua bốn sản phẩm ngân hàng.

Bản đang có gồm **100.224 khách hàng** trong bảng phân chia dữ liệu và **1.202.688 dòng nhãn khách hàng–tháng**, tương ứng 12 tháng từ `2022-02-28` đến `2023-01-31`. Lịch sử giao dịch trải từ cuối năm 2020 đến cuối năm 2022; các nguồn vị trí và hội thoại có phạm vi thời gian riêng.

Dữ liệu nằm tại `data/MBD-mini/`, lưu dưới dạng **Parquet**, chia thành các thư mục `fold=0` đến `fold=4`. Fold là nhóm khách hàng dùng cho việc chia tập dữ liệu.

### Các bảng đang chứa gì?

| Bảng/thư mục | Một dòng biểu diễn gì? | Nội dung chính |
|---|---|---|
| `client_split` | Một khách hàng | `client_id` và fold của khách hàng |
| `targets` | Một khách hàng tại một tháng | `mon`, bốn nhãn `target_1`–`target_4`, cùng `trans_count`, `diff_trans_date` |
| `detail/trx` | Một sự kiện giao dịch | Khách hàng, thời gian, số tiền, mã loại giao dịch, mã tiền tệ và các thuộc tính nguồn/đích đã mã hóa |
| `detail/geo` | Một sự kiện vị trí | Khách hàng, thời gian và các mã vị trí `geohash_4`, `geohash_5`, `geohash_6` |
| `detail/dialog` | Một sự kiện hội thoại | Khách hàng, thời gian và vector `embedding` biểu diễn hội thoại |
| `ptls/trx` | Một chuỗi giao dịch của khách hàng | Các mảng thời gian, số tiền và thuộc tính giao dịch |
| `ptls/geo` | Một chuỗi hoạt động vị trí của khách hàng | Các mảng thời gian và mã vị trí |
| `ptls/dialog` | Một chuỗi hội thoại của khách hàng | Mảng thời gian và các vector biểu diễn hội thoại |

`detail` biểu diễn sự kiện theo từng dòng; `ptls` là cách tổ chức dữ liệu theo chuỗi. Đây là hai dạng biểu diễn, không phải các nguồn sự kiện độc lập để cộng lại. Khóa kết nối các nguồn là `client_id`.

Quy mô các bảng sự kiện gồm khoảng **38,8 triệu giao dịch**, **66,3 triệu sự kiện vị trí** và **217 nghìn sự kiện hội thoại**. Hội thoại được cung cấp dưới dạng vector số, không phải nội dung văn bản gốc. Các mã giao dịch, vị trí và thuộc tính đã được ẩn danh/mã hóa, nên không tự diễn giải chúng thành thông tin nghiệp vụ cụ thể khi chưa có từ điển.

### Bài toán, input và output

**Bài toán:** từ lịch sử của khách hàng, dự đoán khả năng mua từng sản phẩm trong tháng đích. Theo mô tả nguồn, đây là bài toán dự đoán mua sản phẩm trong tháng sau thời điểm báo cáo; khi sử dụng cần căn chỉnh đúng ý nghĩa của `mon` với khoảng nhãn.

**Input:** các giao dịch, hoạt động vị trí và biểu diễn hội thoại của khách hàng trước thời điểm dự đoán. Có thể biểu diễn input thành bảng đặc trưng tổng hợp hoặc chuỗi sự kiện theo thời gian.

**Nhãn có sẵn:** `target_1`, `target_2`, `target_3`, `target_4`, mỗi nhãn nhận giá trị 0 hoặc 1. Tên sản phẩm cụ thể chưa được xác định trong tài liệu đang sử dụng, nên giữ nguyên các tên nhãn này.

**Output dự kiến:** bốn điểm xác suất cho mỗi khách hàng tại một thời điểm:

```text
client_id | thời_điểm_dự_đoán | tháng_đích | p_target_1 | p_target_2 | p_target_3 | p_target_4
```

Đây là bài toán **nhiều nhãn**: một khách có thể mua nhiều sản phẩm. Vì vậy, tổng bốn xác suất không bắt buộc bằng 1. Nhãn 0/1 là kết quả quan sát dùng để học; các cột `p_target_*` là đầu ra cần được mô hình ước lượng.

## 3. eCommerce — dữ liệu hành vi mua sắm trực tuyến

### Bối cảnh và nội dung

[eCommerce behavior data from multi category store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) ghi lại hành vi người dùng tại một cửa hàng trực tuyến có nhiều nhóm sản phẩm.

Bản đang có nằm tại `data/ecommerce/`, gồm hai file **CSV**:

- `2019-Oct.csv`: dữ liệu tháng 10/2019, khoảng 42,4 triệu sự kiện.
- `2019-Nov.csv`: dữ liệu tháng 11/2019, khoảng 67,5 triệu sự kiện.

Tổng cộng có **109.950.743 sự kiện**, **5.316.649 người dùng**, trải trên **61 ngày**, từ 01/10 đến 30/11/2019 theo UTC.

### Mỗi dòng đang chứa gì?

Mỗi dòng là **một sự kiện của một người dùng đối với một sản phẩm**, không phải một hồ sơ khách hàng hoặc một đơn hàng hoàn chỉnh.

| Cột | Ý nghĩa |
|---|---|
| `event_time` | Thời gian xảy ra sự kiện, theo UTC |
| `event_type` | Loại hành vi: `view`, `cart` hoặc `purchase` |
| `product_id` | Mã sản phẩm |
| `category_id` | Mã danh mục sản phẩm |
| `category_code` | Nhãn phân loại danh mục; có thể thiếu |
| `brand` | Thương hiệu sản phẩm; có thể thiếu |
| `price` | Giá sản phẩm được ghi trong sự kiện |
| `user_id` | Mã người dùng |
| `user_session` | Mã phiên truy cập |

Ba hành vi được ghi nhận là **xem sản phẩm**, **thêm vào giỏ hàng** và **mua hàng**. Một người dùng có thể xuất hiện nhiều lần, tương tác với nhiều sản phẩm và có nhiều phiên truy cập. `user_id` kết nối lịch sử của người dùng; `user_session` giúp phân biệt các phiên.

### Bài toán, input và output

**Bài toán minh họa:** dự đoán người dùng có phát sinh hành vi mua hàng trong 7 ngày tới hay không. Có thể đặt bài toán riêng theo danh mục hoặc sản phẩm nếu xác định rõ đối tượng cần dự đoán.

**Input:** lịch sử `view`, `cart`, `purchase` trước thời điểm dự đoán, cùng thông tin sản phẩm, danh mục, thương hiệu, giá và phiên truy cập. Ví dụ thông tin có thể biểu diễn từ lịch sử là số lượt xem, số lần thêm giỏ hàng và thời gian kể từ lần mua gần nhất.

**Nhãn cần tạo:** dữ liệu không có sẵn cột “sẽ mua trong 7 ngày tới”. Với mỗi người dùng tại một thời điểm, quan sát 7 ngày tiếp theo: có ít nhất một sự kiện `purchase` thì nhãn bằng 1; không có log mua trong khoảng đó thì nhãn bằng 0.

**Output dự kiến:** một điểm xác suất mua cho mỗi người dùng tại thời điểm dự đoán:

```text
user_id | thời_điểm_dự_đoán | khoảng_tương_lai | p_purchase_7d
```

Một sự kiện `purchase` là hành vi mua được ghi nhận trong log; bản thân dòng này chưa cho biết đầy đủ trạng thái thanh toán, hoàn/hủy hoặc cấu trúc đơn hàng. Nhãn 0 thể hiện không ghi nhận mua trong khoảng đã chọn, không khẳng định người dùng không mua ở kênh khác.

## 4. Fliggy Transfer-Plans — dữ liệu hành vi du lịch

### Bối cảnh và nội dung

[User Behavior Transfer-Plans from Fliggy Trip Platform for Recommendation](https://tianchi.aliyun.com/dataset/140721) ghi lại tương tác của người dùng với các phương án di chuyển/nối chuyến trên nền tảng Fliggy. Đối tượng người dùng tương tác ở đây là **phương án di chuyển**, khác với sản phẩm bán lẻ của eCommerce.

Bản đang có nằm tại `data/Fliggy/`, gồm ba file **CSV không có header**:

| File | Nội dung |
|---|---|
| `user_behavior_transfer_plans.csv` | Log hành vi của người dùng đối với các phương án di chuyển |
| `user_profile.csv` | Hồ sơ và thuộc tính người dùng đã mã hóa |
| `transfer_plan_profile.csv` | Hồ sơ mô tả các phương án di chuyển |

Log có **42.433.919 sự kiện**, **2.524.995 người dùng**. Theo cách chia ngày UTC+8 đang dùng trong dự án, dữ liệu phủ **34 ngày**, từ 16/08 đến 18/09/2022. UTC+8 là giả định để thống kê theo ngày, chưa phải timezone được nguồn xác nhận.

### Các file đang chứa gì?

File hành vi được đọc theo schema sau:

| Cột | Ý nghĩa theo schema đang sử dụng |
|---|---|
| `user_id` | Mã người dùng |
| `query_id` | Mã truy vấn |
| `plan_id` | Mã phương án di chuyển |
| `behavior` | Mã hành vi; bản hiện có gồm các giá trị 1, 2 và 3 |
| `timestamp` | Thời gian sự kiện dưới dạng Unix timestamp, đơn vị giây |

Mỗi dòng liên kết **người dùng – truy vấn – phương án – hành vi – thời gian**. Một người có thể có nhiều dòng trong cùng ngày hoặc ở nhiều ngày khác nhau.

`user_profile.csv` có năm cột; theo schema đang sử dụng, chúng tương ứng với mã người dùng và các mã thuộc tính tuổi, giới tính, nghề nghiệp, thành phố thường trú. Các giá trị được mã hóa, không nên đọc mã số như tuổi thực hoặc tự gán tên nhóm khi chưa có bảng quy đổi.

`transfer_plan_profile.csv` bổ sung hồ sơ phương án. Hai cột đầu đang được đọc là `query_id` và `plan_id`; ý nghĩa và cách mã hóa các thuộc tính còn lại cần được đối chiếu với từ điển nguồn trước khi diễn giải chi tiết. Do CSV không có header, các tên cột đang sử dụng là schema được gán khi đọc file.

### Bài toán, input và output

**Bài toán gốc của nguồn dữ liệu** liên quan đến gợi ý phương án di chuyển. Trong đề tài propensity, có thể chuyển sang câu hỏi: “Người dùng có quay lại tương tác trong 7 ngày tới không?”.

**Input:** chuỗi hành vi trước thời điểm dự đoán, gồm thời gian, mã hành vi, truy vấn và phương án đã tương tác. Hồ sơ người dùng và phương án là thông tin bổ sung khi đã xác định đúng schema và quan hệ khóa.

**Nhãn cần tạo:** với định nghĩa tương tác đã chọn, có ít nhất một log trong 7 ngày tương lai thì nhãn bằng 1; không có log thì bằng 0.

**Output dự kiến:** một điểm xác suất có tương tác trong khoảng tương lai:

```text
user_id | thời_điểm_dự_đoán | khoảng_tương_lai | p_activity_7d
```

Chưa xác minh được mã `behavior=1/2/3` tương ứng với hành vi nghiệp vụ nào. Vì vậy, hiện chỉ mô tả đầu ra theo **hoạt động/tương tác**. Muốn đặt bài toán xác suất mua hoặc đặt dịch vụ, cần xác định chính xác mã hành vi đó trước khi tạo nhãn.

## 5. Cách hiểu chung về input và output

| Thành phần | MBD-mini | eCommerce | Fliggy |
|---|---|---|---|
| ID đối tượng | `client_id` | `user_id` | `user_id` |
| Lịch sử chính | Giao dịch, vị trí, hội thoại | Xem, thêm giỏ hàng, mua | Tương tác với phương án di chuyển |
| Thông tin bổ sung | Thuộc tính sự kiện đã mã hóa | Sản phẩm, danh mục, thương hiệu, giá, phiên | Hồ sơ người dùng và phương án |
| Nhãn có sẵn | Bốn nhãn sản phẩm theo tháng | Không có nhãn propensity tương lai; cần tạo từ log | Không có nhãn propensity tương lai; cần tạo từ log |
| Output minh họa | Bốn xác suất mua sản phẩm | Một xác suất mua trong 7 ngày | Một xác suất tương tác trong 7 ngày |

Ở cả ba bài toán, cần phân biệt **dữ liệu đang chứa** với **đầu ra mong muốn**: các file hiện tại chứa sự kiện, hồ sơ hoặc nhãn quan sát; điểm propensity là kết quả được ước lượng sau khi xây dựng mô hình. Lịch sử đầu vào phải nằm trước khoảng tương lai cần dự đoán.

Ba bộ dữ liệu độc lập về lĩnh vực, người dùng và cách mã hóa. Cùng tên `user_id` không có nghĩa là cùng một người giữa eCommerce và Fliggy. Chúng có thể dùng để nghiên cứu cùng khái niệm propensity, nhưng hành vi đích và ý nghĩa output phải được xác định riêng cho từng bộ.
