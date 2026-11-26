"""Generate notebooks/11_baseline_v2_colab.ipynb cleanly via nbformat."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[2]


def _md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def _code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def build_colab_notebook() -> Path:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    nb["cells"] = [
        _md("""
# 11. Baseline v2 — Giao thức Rolling theo tháng × Tách khách (`Protocol B`) trên Google Colab

Notebook chạy trọn vẹn **Baseline v2** theo đặc tả đóng băng [`configs/baseline_v2_evaluation_spec.json`](../configs/baseline_v2_evaluation_spec.json) và [`docs/BASELINE_V2_FROZEN_SPEC.md`](../docs/BASELINE_V2_FROZEN_SPEC.md):
- **Giao thức chính (`Protocol B`):** Huấn luyện lại hàng tháng (`2022-04 .. m-1`) trên `fold != k`, chấm điểm khách hàng mới ở tháng `m` (`fold == k`), ngân sách chiến dịch `K = ceil(1% × N_month) ~ 1.003` cuộc gọi/tháng.
- **Giao thức đối chiếu (`Protocol A`):** Flat OOC để đối chiếu trực tiếp với hệ Baseline v1 cũ (vẫn giữ nguyên các kết quả cũ).
- **6 Mốc Baseline v2:** `L0_base_rate`, `L2_rule_trx_rfm`, `B_logreg_T`, `B_logreg_TGD`, `B_catboost_TGD_tuned`, `L1_rule_target_lag` *(chỉ tham chiếu — kèm cảnh báo độ trễ nhãn)*.
- **Nguyên tắc Val-only / Test-once:** Tuning CatBoost (`lr`, `depth`, `l2_leaf_reg`, early stopping trên `m-1`), ablation `T+G` / `T+D`, ablation `trx_all_days_since_first_event` (`drop` / `cap365`), ensemble `LogReg + CatBoost`, Permutation Importance, Slice Analysis và Shuffled-Label Sanity Check **chỉ chạy trên Val (`2022-09`, `2022-10`)**. **Locked Test (`2022-11`, `2022-12`, `2023-01`) chỉ mở 1 lần** sau khi khóa cấu hình.
"""),
        _md("## 1. Mount Google Drive & Cài thư viện"),
        _code("""
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "catboost", "scikit-learn", "pandas", "pyarrow", "scipy", "matplotlib", "duckdb"
])
print("Environment ready | IN_COLAB =", IN_COLAB)
"""),
        _md("""
## 2. Cấu hình đường dẫn Repo, `panel_moscow.parquet` và Thư mục Output

- **Cách nhanh nhất trên Colab:** Upload sẵn repo (hoặc `git clone`) và đặt `panel_moscow.parquet` (`~185 MB`) lên Google Drive (ví dụ `/content/drive/MyDrive/propensity/outputs/baseline_moscow_v1/panel_moscow.parquet`).
- Nếu chưa có `panel_moscow.parquet` nhưng có thư mục raw `data/MBD-mini/`, cell dưới sẽ tự gọi `src/scripts/build_moscow_features.py` để tạo.
"""),
        _code("""
# Chỉnh đường dẫn REPO_DIR / PANEL_PATH / OUT_DIR theo vị trí thực tế trên Google Drive / Colab
REPO_DIR = Path("/content/drive/MyDrive/propensity") if IN_COLAB else Path.cwd().resolve()
if not (REPO_DIR / "configs" / "baseline_v2_evaluation_spec.json").exists():
    for cand in [Path.cwd().resolve(), Path.cwd().resolve().parent, Path("/content/Propensity"), Path("/content/new propensity")]:
        if (cand / "configs" / "baseline_v2_evaluation_spec.json").exists():
            REPO_DIR = cand
            break

PANEL_PATH = REPO_DIR / "outputs" / "baseline_moscow_v1" / "panel_moscow.parquet"
OUT_DIR = REPO_DIR / "outputs" / "baseline_v2"

if str(REPO_DIR / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "src"))

print("REPO_DIR  :", REPO_DIR)
print("PANEL_PATH:", PANEL_PATH, "| exists =", PANEL_PATH.exists())
print("OUT_DIR   :", OUT_DIR)

if not PANEL_PATH.exists():
    raw_dir = REPO_DIR / "data" / "MBD-mini"
    assert raw_dir.exists(), f"Không thấy {PANEL_PATH} lẫn {raw_dir}. Hãy copy panel_moscow.parquet lên Drive!"
    print("Đang dựng panel_moscow.parquet từ data/MBD-mini/ ...")
    subprocess.check_call([sys.executable, str(REPO_DIR / "src" / "scripts" / "build_moscow_features.py")])
"""),
        _md("## 3. Kiểm tra toàn vẹn Spec Đóng băng & Schema Dữ liệu"),
        _code("""
import json
import pyarrow.parquet as pq
from baseline_v2.spec import load_spec, verify_or_create_freeze_lock

spec = load_spec(REPO_DIR / "configs" / "baseline_v2_evaluation_spec.json")
lock = verify_or_create_freeze_lock(
    OUT_DIR,
    spec_json_path=REPO_DIR / "configs" / "baseline_v2_evaluation_spec.json",
    spec_md_path=REPO_DIR / "docs" / "BASELINE_V2_FROZEN_SPEC.md",
)
pf = pq.ParquetFile(PANEL_PATH)
print("Spec ID     :", spec["spec_id"], "| version:", spec["spec_version"])
print("Freeze Lock :", json.dumps(lock, indent=2))
print("Panel rows  :", f"{pf.metadata.num_rows:,}", "| cols:", len(pf.schema.names))
"""),
        _md("## 4. Phase 1 (Validation Only) — Tuning CatBoost, 6 Mốc Baseline, Ablations & Diagnostics"),
        _code("""
from baseline_v2.runner import run_baseline_v2

# Chạy strictly trên Val (2022-09, 2022-10); Test chưa mở ở bước này
val_manifest = run_baseline_v2(
    phase="val",
    panel_path=PANEL_PATH,
    out_dir=OUT_DIR,
    seeds_override=[42, 142, 242],
    include_protocol_a=True,
    thread_count=-1,
    n_bootstrap_override=1000,
    spec_json_path=REPO_DIR / "configs" / "baseline_v2_evaluation_spec.json",
    spec_md_path=REPO_DIR / "docs" / "BASELINE_V2_FROZEN_SPEC.md",
)
print("Phase 1 (Val) done:", val_manifest["elapsed_seconds"], "s")
"""),
        _md("## 5. Xem Bảng Chẩn đoán trên Validation (Trước khi mở Locked Test)"),
        _code("""
import pandas as pd
from IPython.display import display

m_dir = OUT_DIR / "metrics"
print("=== 5.1. CatBoost Tuning Grid (Val Protocol B) ===")
display(pd.read_csv(m_dir / "val_catboost_tuning_grid.csv"))

print("=== 5.2. Val Monthly Summary ===")
val_m = pd.read_csv(m_dir / "monthly_metrics_all.csv")
display(val_m[val_m["stage"] == "val"][["protocol", "month", "exp_id", "n_features", "lift_1pct_mean", "lift_1pct_ci_low", "lift_1pct_ci_high", "average_precision_mean", "average_precision_ci_low", "average_precision_ci_high", "roc_auc_mean"]])

print("=== 5.3. Val Ablations: T+G / T+D, Tenure (drop/cap365), Linear+Tree Ensemble ===")
display(pd.read_csv(m_dir / "val_ablation_macro.csv")[["group", "exp_id", "n_features", "lift_1pct_mean", "lift_1pct_ci_low", "lift_1pct_ci_high", "average_precision_mean", "average_precision_ci_low", "average_precision_ci_high", "roc_auc_mean"]])

print("=== 5.4. Val Permutation Importance (14 features) ===")
display(pd.read_csv(m_dir / "val_permutation_importance.csv"))

print("=== 5.5. Val Slice Analysis & Shuffled-Label Sanity Check ===")
display(pd.read_csv(m_dir / "val_slice_metrics.csv"))
display(pd.read_csv(m_dir / "val_shuffled_sanity.csv"))
"""),
        _md("## 6. Phase 2 — Mở Locked Test 1 Lần Duy Nhất (`2022-11`, `2022-12`, `2023-01`)"),
        _code("""
# Khóa cấu hình CatBoost đã chọn ở Val và chạy Test 1 lần cho Protocol B (chính) + Protocol A (đối chiếu)
test_manifest = run_baseline_v2(
    phase="test",
    panel_path=PANEL_PATH,
    out_dir=OUT_DIR,
    seeds_override=[42, 142, 242],
    include_protocol_a=True,
    thread_count=-1,
    n_bootstrap_override=1000,
    spec_json_path=REPO_DIR / "configs" / "baseline_v2_evaluation_spec.json",
    spec_md_path=REPO_DIR / "docs" / "BASELINE_V2_FROZEN_SPEC.md",
)
print("Phase 2 (Locked Test) done:", test_manifest["elapsed_seconds"], "s")
"""),
        _md("## 7. Tổng hợp Kết quả Locked Test: Từng tháng (`K=1.003`), Gộp Macro-Monthly, Protocol B vs A & Biểu đồ"),
        _code("""
from IPython.display import Image

m_dir = OUT_DIR / "metrics"
mdf = pd.read_csv(m_dir / "monthly_metrics_all.csv")
macro_df = pd.read_csv(m_dir / "macro_monthly_summary.csv")
pooled_df = pd.read_csv(m_dir / "pooled_oof_summary.csv")

print("=== 7.1. Locked Test — Từng tháng (Protocol B, K ~ 1,003/tháng) ===")
display(mdf[(mdf["protocol"] == "B_rolling_ooc") & (mdf["stage"] == "test")][["month", "exp_id", "is_reference_only", "positives", "lift_1pct_mean", "lift_1pct_ci_low", "lift_1pct_ci_high", "average_precision_mean", "average_precision_ci_low", "average_precision_ci_high", "roc_auc_mean"]])

print("=== 7.2. Macro-Monthly Summary (Protocol B vs Protocol A) ===")
display(macro_df[macro_df["stage"] == "test"][["protocol", "exp_id", "n_features", "is_reference_only", "lift_1pct_mean", "lift_1pct_ci_low", "lift_1pct_ci_high", "average_precision_mean", "average_precision_ci_low", "average_precision_ci_high", "roc_auc_mean"]])

print("=== 7.3. Pooled OOF Summary (Đối chiếu thẳng với Baseline v1) ===")
display(pooled_df[pooled_df["stage"] == "test"][["protocol", "exp_id", "lift_1pct_mean", "lift_1pct_ci_low", "lift_1pct_ci_high", "average_precision_mean", "average_precision_ci_low", "average_precision_ci_high", "roc_auc_mean"]])

print("=== 7.4. Paired Client-Clustered Bootstrap: Protocol B vs A & Model Pairs ===")
display(pd.read_csv(m_dir / "paired_protocol_b_vs_a.csv"))
display(pd.read_csv(m_dir / "paired_models_protocol_b.csv"))

for fig_path in sorted((OUT_DIR / "figures").glob("*.png")):
    print(fig_path.name)
    display(Image(filename=str(fig_path)))
"""),
    ]

    out_nb = REPO_ROOT / "notebooks" / "11_baseline_v2_colab.ipynb"
    nbf.write(nb, out_nb)
    return out_nb


if __name__ == "__main__":
    p = build_colab_notebook()
    print(f"Generated {p}")
