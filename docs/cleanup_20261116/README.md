# Dọn workspace — 16/11/2026

Đã xóa vĩnh viễn code, config, CLI, notebook, tests riêng, model, prediction,
metric, feature xuất riêng, báo cáo và deck của workflow baseline. Không thay đổi Git history.

- `deletion_plan.json`: 128 mục, 1.665 file, khoảng 6 GiB trước khi xóa.
- `deletion_results.json`: 128/128 mục đã xóa.
- `duplicate_final_plan.json`, `duplicate_final_results.json`: 13 mục EDA/helper cũ bổ sung đã xóa.
- Xóa thêm `docs/workspace_review_20261110.md` vì chỉ còn liên quan baseline đã bỏ.
- `validation.json`: kiểm tra dependency, code, notebook và kết quả test.

Raw, sample, report hiện hành, cache khách–tháng, bảng nguồn EDA và output chuẩn bị
hiện tại được giữ. Các archive cũ đã xóa, không giữ lại bản baseline dưới tên khác.

EDA không còn đọc protocol của model đã xóa. Spearman dùng configs/eda_analysis.json.
Dữ liệu chuẩn bị dùng notebook 03, config propensity_processing.json; notebook 02 cũ đã bỏ.
Scikit-learn còn dùng cho PCA/K-means mô tả embedding, không có trainer propensity.

Lượt unit test trong sandbox gặp lỗi thư mục tạm Windows; các ca export liên quan đã
chạy lại bên ngoài sandbox: 8/8 đạt. Đồng bộ notebook 03/04, parse 115 file Python,
kiểm tra import và packaging đều đạt. Không chạy lại toàn bộ EDA trên raw để dọn file.
