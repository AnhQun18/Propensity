# `eda_report` — Privacy-Safe EDA Reporting, Headless Charts & Slide Decks

Trực quan hóa và xuất báo cáo EDA an toàn quyền riêng tư cho mọi bộ dữ liệu:
- **`export_report`**: xuất `report.md` (kèm Mermaid ERD tự động), `tables/<section>/*.csv`, `figures/*.png`, `metadata.json`, `manifest.json`; **tự động chặn xuất bảng chứa cột định danh cá nhân** (`forbidden_columns`).
- **Biểu đồ EDA headless (`matplotlib.figure.Figure`)**: `plot_distribution` (histogram + boxplot), `plot_correlation_heatmap`, `plot_missing_bars`, `plot_bars`, `plot_rate_matrix`, `plot_timeline`, `plot_quantile_whisker`.
- **`publish_figure_bundle`**: xuất đồng bộ bộ 4 file (`PNG + SVG + CSV số liệu + MD thuyết minh`).
- **`SlideBuilder`**: dựng slide EDA 16:9 (header bar, KPI card, bottom insight pill) và xuất PDF/PNG.
- **`format_number`**: định dạng số đa ngôn ngữ (`locale="en"` / `"vi"`).

## Sử dụng nhanh
```python
from src.eda_report import export_report, plot_correlation_heatmap, plot_distribution, SlideBuilder

export_report(profile_result, "outputs/eda_report", forbidden_columns=("user_id", "email"))
fig = plot_correlation_heatmap(corr["matrix"])
```
