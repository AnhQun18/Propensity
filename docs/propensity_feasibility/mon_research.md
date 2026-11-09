# Ý nghĩa `mon` trong MBD-mini

Ngày nghiên cứu: 05/11/2026. Đây là cách diễn giải phục vụ dựng mẫu propensity; chưa có xác nhận trực tiếp từ tác giả bộ dữ liệu về múi giờ hoặc quy trình tạo nhãn.

**Đã đối chiếu thêm mã nguồn ngày 09/11/2026.** Protocol EventGPT xác định raw UTC và
cutoff đầu tháng Moscow; `month_cut` lấy đầu tháng chứa `mon` rồi trừ 10.800 giây.
**Quyết định người dùng sau rà soát:** tạm dùng UTC thống nhất, cutoff đầu tháng lúc
00:00 UTC và không dịch 3 giờ. Đây là quy ước nghiên cứu; protocol Moscow của EventGPT
giữ làm đối chiếu, không phải cấu hình chính hiện tại. Không đổi tháng nhãn hoặc tự
coi các cohort cũ đã tính lại theo Moscow.
Xem [bằng chứng, commit và audit độ nhạy](open_questions/answers.md#7-tháng-nhãn-và-timezone-đã-xác-minh-đến-đâu).

## Kết luận và cách áp dụng

`mon` được dataset card gọi là tháng báo cáo. Trong bản dữ liệu ở máy, giá trị là ngày cuối tháng; cách diễn giải có cơ sở tốt nhất là tháng lịch chứa giá trị đó là tháng nhãn. Lấy ngày đầu tháng làm cutoff và chỉ dùng sự kiện có `event_time < cutoff`.

Ví dụ: `mon = 2022-02-28` → tháng nhãn là 02/2022, khoảng nhãn `[2022-02-01, 2022-03-01)`; lịch sử feature trước `2022-02-01 00:00:00`. Bốn cột target tương ứng bốn sản phẩm, không phải bốn tháng liên tiếp. Với `mon = 2023-01-31`, theo cách hiểu này tháng nhãn là 01/2023 và cutoff là 01/01/2023; có thể dùng nhãn chính thức dù log sự kiện kết thúc vào tháng 12/2022.

## Bằng chứng và giới hạn

- [Dataset card MBD-mini](https://huggingface.co/datasets/ai-lab/MBD-mini) gọi `mon` là reporting month, nêu mục tiêu dự đoán mua bốn sản phẩm trong tháng sau ngày báo cáo. Mô tả riêng từng target dùng từ “first/second/third/fourth reporting month” gây mơ hồ; không xem đây là định nghĩa bốn khoảng thời gian vì các nguồn sau định nghĩa rõ bốn sản phẩm.
- [Bài báo MBD gốc, mục 2.1 và 3.2.1](https://arxiv.org/html/2409.17587v1) mô tả bốn nhãn nhị phân cho bốn sản phẩm, và yêu cầu lịch sử sự kiện phải nằm nghiêm ngặt trước đầu tháng mua. Bài báo nói khoảng ngày nhãn được dịch một tháng, từ 01/02/2022 đến 31/01/2023; đồng thời có chỗ nhắc nhãn của 2022, nên diễn giải câu “dịch” một mình không đủ xác định chính xác mapping từng giá trị `mon`.
- [EventGPT, mục 6.1](https://arxiv.org/html/2611.03302) sử dụng chính MBD-mini: mỗi dòng là một khách–tháng từ 02/2022 đến 01/2023, bốn target cho biết mua sản phẩm trong tháng đó, và lịch sử chỉ gồm sự kiện trước ngày đầu tháng. Bài này giữ đủ 12 tháng cho các nhãn sản phẩm chính thức. Đây là bằng chứng độc lập ủng hộ mapping trên, không phải xác nhận trực tiếp từ tác giả MBD.
- EventGPT nói raw UTC và cutoff theo giờ Moscow; mã `month_cut` thực hiện đúng quy tắc đó. Các shard đại diện ở máy không ghi timezone trong kiểu timestamp. Mã MBD gốc `GetSplit` gọi `datetime(...).timestamp()` không chỉ rõ timezone nên phụ thuộc môi trường. Đã xác minh cách triển khai EventGPT; vẫn chưa có xác nhận trực tiếp về timezone ETL nhãn gốc. Đây là quy ước nghiên cứu có bằng chứng, không phải timezone tự khôi phục từ schema.
- Đã đọc mã benchmark gốc và notebook ghép targets nhưng không tìm thấy ETL tạo nhãn. Mapping tháng có bằng chứng từ paper và code; không tự dịch `mon` thêm một tháng. Audit mới đo 88.976 sự kiện thuộc biên 3 giờ giữa UTC-midnight và Moscow-midnight, tác động 71.467 cặp khách–tháng; cần dựng lại feature/cohort theo cùng protocol trước huấn luyện.

## Phân chia đánh giá

Để đánh giá dự đoán cho tháng tương lai, chia train/validation/test theo thứ tự cutoff và chỉ dùng feature quá khứ. Để so sánh trực tiếp benchmark gốc, dùng 5 fold theo khách hàng như bài MBD; đây là câu hỏi đánh giá khác với kiểm thử theo thời gian. Khi huấn luyện xuyên thời gian, mọi nhãn train phải đã hoàn tất trước cutoff của tập đánh giá.
