# Đính chính EDA: tính ngoại lai trên log

Yêu cầu rà soát ngoại lai trên log đã được ghi và tính trong
[bản kiểm tra 09/11/2026](propensity_feasibility/tail_review/assessment.md).
Bản EDA xuất sau đó dùng IQR/Z-score trên số gốc làm kết quả chính và chỉ dùng
log cho histogram/đề xuất biến đổi. Cách trình bày này chưa kế thừa yêu cầu trước.

## Phương pháp được khôi phục

1. Giữ raw nguyên trạng. Với amount không âm, tính `z=ln(1+DOUBLE(amount))` theo từng currency.
2. Tính Q1/Q3 bằng quantile nội suy tuyến tính **trên các z**, không lấy log của Q1/Q3 gốc.
3. Đếm riêng `z<Q1-1.5*IQR` và `z>Q3+1.5*IQR`, rồi cộng hai đuôi.
4. `ln(amount)` chỉ xét amount dương để kiểm tra độ nhạy. Thang gốc chỉ đối chiếu.
5. Count có 0 dùng log1p. Feature đã log không log lần hai; số có âm dùng signed-log1p.
6. Không kết luận ngoại lai cho cờ/tỷ lệ trong [0,1], biến hằng, IQR log bằng 0,
   nhóm dưới 30 giá trị hoặc currency chưa biết. Ghi trạng thái không áp dụng,
   không nhầm giá trị trống với không có ngoại lai.

Đổi trục biểu đồ sang log không tự đổi hàng rào IQR. Phép biến đổi phải nằm trong
thuật toán tính quartile và gắn cờ. Ngưỡng sau log có thể quy đổi về đơn vị đầu vào
để đọc; việc so sánh và số đếm vẫn phải nhất quán với thang biến đổi.

## Đối soát currency 11

Mẫu số: 38.724.240 giao dịch, tất cả hữu hạn và dương trong snapshot này.

| Thang | Đuôi thấp | Đuôi cao | Tổng ngoài hàng rào | Tỷ lệ |
| --- | ---: | ---: | ---: | ---: |
| Gốc, chỉ đối chiếu | 0 | 5.521.385 | 5.521.385 | 14,2582% |
| ln(amount), độ nhạy | 516.970 | 97.222 | 614.192 | 1,5861% |
| log1p(amount), kết quả chính | 486.868 | 97.271 | 584.139 | 1,5085% |

Kết quả tham chiếu được lưu trong
[CSV ngày 09/11](propensity_feasibility/tail_review/amount_log_iqr_by_currency.csv).
`scripts/validate_eda.py` đối soát lại cả hai đuôi, tổng log1p và tổng ln với CSV này
sau khi notebook thực thi trên raw. Bảng so sánh mới được xuất tại
`deliverables/eda/tables/amount_iqr_scale_comparison_currency11.csv`.

## Giới hạn sử dụng

Ngoài hàng rào log-IQR chưa có nghĩa dữ liệu lỗi. Không diễn giải toàn bộ 584.139
dòng là giao dịch quá lớn, vì phần lớn nằm ở đuôi thấp. Không tự xóa/cap dòng hoặc
khách chỉ dựa trên cờ này. Ngưỡng raw toàn kỳ phục vụ EDA; khi đưa vào mô hình,
phải fit ngưỡng theo currency chỉ từ train và lịch sử trước cutoff, rồi kiểm chứng
bằng validation theo thời gian.
