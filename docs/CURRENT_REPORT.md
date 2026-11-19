# Báo cáo hiện hành

Nguồn đối chiếu: [Canva MBD-mini](https://canva.link/156goz4ufsl0xxp), design `DAHalHSbRaU`, đọc ngày 16/11/2026. Bản text được lưu tại [snapshot](current_canva_report_snapshot.txt). Snapshot chỉ chứa text của Canva, không bao gồm nội dung chữ nằm trong ảnh.

## Trạng thái dọn workspace ngày 16/11/2026

Đã xóa vĩnh viễn workflow baseline (code, cấu hình, notebook, tests riêng, model,
dự đoán và báo cáo) theo yêu cầu tiếp theo. Các bản archive cũ cũng đã xóa.
Nhật ký: `docs/cleanup_20261116/deletion_plan.json` và `deletion_results.json`.
Giữ raw và các bảng/cache được code EDA hiện hành đọc. Không coi từ “baseline”
chỉ tỷ lệ nền trong công thức lift là mô hình cần xóa.

## Danh mục biểu đồ hiện hành

Đường dẫn dưới đây là các phiên bản cuối đã thống nhất trong chat; là tài liệu hỗ trợ tái lập, không tuyên bố mọi ảnh đều xuất hiện trong Canva.

| Nội dung | Output | Code |
| --- | --- | --- |
| Độ phủ khách–tháng từng nguồn | `outputs/source_presence_target/customer_month_log_coverage.png` | `src/scripts/plot_customer_month_log_coverage.py` |
| Có/không nguồn và tỷ lệ mua, CI, n, tỷ trọng, Lift | `outputs/source_presence_target/source_presence_purchase_ci.png` | `src/scripts/plot_source_presence_target.py` |
| Nhãn y_any | `outputs/trx_charts/y_any_positive_rate.png` | `outputs/trx_charts/plot_y_any.py` |
| Horizon theo nhiều cutoff | `outputs/trx_charts/buyers_horizon_x_by_cutoff.png` | `outputs/trx_charts/plot_buyers_horizon_x.py` |
| Số tháng có mua | `outputs/trx_charts/purchase_months_per_customer.png` | `outputs/trx_charts/plot_purchase_frequency.py` |
| Returning từ tháng 3 | `outputs/trx_charts/returning_vs_first_observed_buyers_from_march.png` | `outputs/trx_charts/plot_returning_buyers_from_march.py` |
| Returning theo tháng, hai định nghĩa | `outputs/trx_charts/returning_buyer_share_monthly_comparison.png` | `outputs/trx_charts/plot_returning_monthly_comparison.py` |
| Currency | `outputs/trx_charts/trx_currency.png` | `outputs/trx_charts/plot_currency.py` |
| Amount liên tục | `outputs/trx_charts/trx_amount_currency11_continuous.png` | `outputs/trx_charts/plot_amount_continuous.py` |
| RFM | `outputs/trx_charts/trx_rfm_purchase_vs_no_purchase.png` | `outputs/trx_charts/compute_rfm_means.py`, rồi `outputs/trx_charts/plot_rfm.py` |
| Top 5 event_type | `outputs/trx_charts/trx_top5_event_type_portrait.png` | `outputs/trx_charts/plot_event_type_portrait.py` |
| Lift event_type | `outputs/trx_charts/trx_event_type_lift_horizontal.png` | `outputs/trx_charts/generate_event_type_lift_data.py`, rồi `outputs/trx_charts/plot_event_type_lift_horizontal.py` |
| Phân cấp GEO | `outputs/geo_charts/geo_hierarchy_code_counts.png` | `outputs/geo_charts/plot_geo_hierarchy_counts.py` |
| GEO chất lượng / độ phủ | `outputs/geo_charts/geo_record_quality.png`, `geo_customer_coverage.png` | `outputs/geo_charts/plot_geo_coverage_quality.py` |
| GEO theo tháng | `outputs/geo_charts/geo_monthly_coverage.png` | `outputs/geo_charts/plot_geo_monthly.py` |
| GEO cấp 6 có log, đường liên tục | `outputs/geo_charts/geo_level6_positive_continuous.png` | `outputs/geo_charts/plot_geo_unique_distributions.py` |
| GEO và target, các khoảng mã thật | `outputs/geo_charts/geo_signal_vs_target.png`, `geo_level4_signal_vs_target.png`, `geo_level5_signal_vs_target.png` | Các script `plot_geo*_signal_target.py` trong `outputs/geo_charts/` |
| Ví dụ đổi vùng chủ đạo | `outputs/geo_charts/dominant_region_shift_example.png` | `outputs/geo_charts/plot_dominant_region_shift.py` |
| Dialog độ phủ | `outputs/dialog_charts/dialog_customer_coverage.png` | `outputs/dialog_charts/plot_dialog_coverage.py` |
| Norm embedding | `outputs/dialog_charts/embedding_norm_distribution.png` | `outputs/dialog_charts/check_embedding_norms.py` |
| Metadata dialog trên cohort có log | `outputs/dialog_charts/dialog_metadata_vs_target_active.png` | `outputs/dialog_charts/plot_dialog_metadata_target_active.py` |
| Tương tác dialog không dùng log scale | `outputs/dialog_charts/dialog_interaction_patterns_linear.png` | `outputs/dialog_charts/plot_interaction_patterns_linear.py` |
| PCA / cluster mô tả | `outputs/dialog_charts/semantic_embedding_exploration.png` | `outputs/dialog_charts/plot_semantic_exploration.py` |
| Dialog theo giờ / thứ | `outputs/dialog_charts/dialog_temporal_behavior.png` | `outputs/dialog_charts/plot_dialog_temporal.py` |
| Audit lịch dialog / vùng giảm mức ghi nhận | `outputs/dialog_temporal_audit/dialog_timeline_audit.png` | `src/scripts/audit_dialog_timeline.py` |

Chạy Python bằng `.venv/Scripts/python.exe` từ root workspace. Không chạy code dưới `outputs/archive/` trong quy trình hiện hành.

## Dữ liệu nguồn giữ lại

- `data/`, `sample_data/`: dữ liệu gốc/mẫu, không sửa.
- `src/propensity_data/`, notebook 03–04 và `outputs/propensity_processing_v3/`: chuẩn bị dữ liệu, không train mô hình.
- `outputs/dataset_eda/`, `outputs/mbd_review_cache/`, bảng `docs/report/`, `docs/mbd_mini_eda/tables/`, `docs/deliverables/*/tables/`: dependency EDA.
- Các audit trans_count, timestamp, currency và amount còn làm căn cứ cho giới hạn diễn giải.
- Cấp 4/5/6: `compute_unique_geohashes.py` tạo bảng cache; `plot_geo_unique_distributions.py` tạo ba phân phối mới, chỉ trên khách có GEO.
- Horizon trượt: `compute_buyers_rolling_horizon.py` tạo CSV; `plot_buyers_horizon_x.py` vẽ phiên bản mới nhất.
- Spearman dùng `configs/eda_analysis.json`, độc lập workflow model đã xóa.

## Quy ước

`mon` thuộc tháng nhãn T; cutoff là đầu tháng T, UTC của dự án. Feature chỉ dùng
lịch sử trước cutoff. Có log toàn lịch sử khác có log T−1. Lift là tỷ lệ mua nhóm
chia tỷ lệ mua chung. Mã sản phẩm, tiền tệ và vị trí chưa có ánh xạ đầy đủ.
