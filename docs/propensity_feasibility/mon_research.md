# Ý nghĩa `mon` trong MBD-mini

Ngày nghiên cứu: 05/11/2026. Đây là cách diễn giải phục vụ dựng mẫu propensity; chưa có xác nhận trực tiếp từ tác giả bộ dữ liệu về múi giờ hoặc quy trình tạo nhãn.

## Kết luận và cách áp dụng

`mon` được dataset card gọi là tháng báo cáo. Trong bản dữ liệu ở máy, giá trị là ngày cuối tháng; cách diễn giải có cơ sở tốt nhất là tháng lịch chứa giá trị đó là tháng nhãn. Lấy ngày đầu tháng làm cutoff và chỉ dùng sự kiện có `event_time < cutoff`.

Ví dụ: `mon = 2022-02-28` → tháng nhãn là 02/2022, khoảng nhãn `[2022-02-01, 2022-03-01)`; lịch sử feature trước `2022-02-01 00:00:00`. Bốn cột target tương ứng bốn sản phẩm, không phải bốn tháng liên tiếp. Với `mon = 2023-01-31`, theo cách hiểu này tháng nhãn là 01/2023 và cutoff là 01/01/2023; có thể dùng nhãn chính thức dù log sự kiện kết thúc vào tháng 12/2022.

## Bằng chứng và giới hạn

- [Dataset card MBD-mini](https://huggingface.co/datasets/ai-lab/MBD-mini) gọi `mon` là reporting month, nêu mục tiêu dự đoán mua bốn sản phẩm trong tháng sau ngày báo cáo. Mô tả riêng từng target dùng từ “first/second/third/fourth reporting month” gây mơ hồ; không xem đây là định nghĩa bốn khoảng thời gian vì các nguồn sau định nghĩa rõ bốn sản phẩm.
- [Bài báo MBD gốc, mục 2.1 và 3.2.1](https://arxiv.org/html/2409.17587v1) mô tả bốn nhãn nhị phân cho bốn sản phẩm, và yêu cầu lịch sử sự kiện phải nằm nghiêm ngặt trước đầu tháng mua. Bài báo nói khoảng ngày nhãn được dịch một tháng, từ 01/02/2022 đến 31/01/2023; đồng thời có chỗ nhắc nhãn của 2022, nên diễn giải câu “dịch” một mình không đủ xác định chính xác mapping từng giá trị `mon`.
- [EventGPT, mục 6.1](https://arxiv.org/html/2611.03302) sử dụng chính MBD-mini: mỗi dòng là một khách–tháng từ 02/2022 đến 01/2023, bốn target cho biết mua sản phẩm trong tháng đó, và lịch sử chỉ gồm sự kiện trước ngày đầu tháng. Bài này giữ đủ 12 tháng cho các nhãn sản phẩm chính thức. Đây là bằng chứng độc lập ủng hộ mapping trên, không phải xác nhận trực tiếp từ tác giả MBD.
- EventGPT nói cutoff theo giờ Moscow. Schema timestamp của bản dữ liệu địa phương không ghi timezone, nên chưa đủ căn cứ để tự chuyển timestamp sang Moscow hoặc khẳng định biên chính xác theo múi giờ. Trước mắt dùng lịch/timestamp lưu trong file một cách nhất quán và ghi rõ giả định; cần xác nhận nguồn hoặc kiểm tra mã tiền xử lý trước khi khẳng định múi giờ.
- Chưa xác minh trực tiếp mã benchmark gốc hoặc nguồn tạo nhãn. Vì vậy mapping tháng được xem là lựa chọn triển khai có bằng chứng, không coi là xác nhận tuyệt đối từ nhà phát hành. Không tự dịch `mon` thêm một tháng.

## Phân chia đánh giá

Để đánh giá dự đoán cho tháng tương lai, chia train/validation/test theo thứ tự cutoff và chỉ dùng feature quá khứ. Để so sánh trực tiếp benchmark gốc, dùng 5 fold theo khách hàng như bài MBD; đây là câu hỏi đánh giá khác với kiểm thử theo thời gian. Khi huấn luyện xuyên thời gian, mọi nhãn train phải đã hoàn tất trước cutoff của tập đánh giá.
