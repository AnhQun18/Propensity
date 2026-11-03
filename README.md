# New Propensity

Repository lưu trữ tài liệu, notebook phân tích dữ liệu khám phá (EDA), mã nguồn core và dữ liệu ví dụ cho dự án propensity.

## Cấu trúc thư mục

```text
.
├── docs/             # Tài liệu dự án
├── notebooks/        # Notebook EDA
├── src/              # Mã nguồn core
├── sample_data/      # Dữ liệu ví dụ
├── data/             # Script tải, dữ liệu MBD-mini và eCommerce
├── pyproject.toml    # Khai báo thông tin dự án và dependency
└── README.md         # Mô tả repository
```

Các thư mục có file `.gitkeep` để Git lưu lại cấu trúc khi chưa có nội dung.

## Dependency

Khai báo các thư viện cần dùng trong danh sách `dependencies` của `pyproject.toml`.

Script `data/code.py` tải bộ dữ liệu
[eCommerce behavior data from multi category store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)

Bộ [MBD-mini](https://huggingface.co/datasets/ai-lab/MBD-mini) đã tải và giải nén
trước đó nằm riêng tại `data/MBD-mini/`.

[dataset 140721](https://tianchi.aliyun.com/dataset/140721)

- `user_behavior_transfer_plans.csv`
- `user_profile.csv`
- `transfer_plan_profile.csv`

