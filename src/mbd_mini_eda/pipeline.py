"""Kế hoạch và thực thi phân tích; ghi file chỉ qua reporting.export_report."""

from dataclasses import asdict, dataclass

from .config import MBDMiniConfig
from .models import AnalysisContext, AnalysisResult, SectionResult
from .schema import TABLES


@dataclass(frozen=True)
class AnalysisStep:
    key: str
    stage: str
    title: str
    grain: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()
    creates_artifacts: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "module": f"mbd_mini_eda.{self.key}",
            "callable": "run",
            "implementation_status": "implemented",
        }


STEPS = (
    AnalysisStep(
        "descriptive.structure", "descriptive", "Cấu trúc và quy mô", "table",
        ("clients", "targets", "transactions", "geo", "dialog"),
        ("table_inventory", "column_roles", "relationship_summary"),
    ),
    AnalysisStep(
        "descriptive.quality", "descriptive", "Chất lượng dữ liệu", "column / key",
        ("clients", "targets", "transactions", "geo", "dialog"),
        ("missing_values", "invalid_values", "duplicates", "key_coverage"),
        depends_on=("descriptive.structure",),
    ),
    AnalysisStep(
        "descriptive.temporal", "descriptive", "Phạm vi thời gian", "source / month / customer",
        ("transactions", "geo", "dialog"),
        ("source_time_ranges", "monthly_activity", "history_length_summary"),
        depends_on=("descriptive.structure",),
    ),
    AnalysisStep(
        "descriptive.transactions", "descriptive", "Phân phối giao dịch", "transaction event",
        ("transactions",), ("amount_summary", "transaction_categories", "transaction_outliers"),
        depends_on=("descriptive.quality",),
    ),
    AnalysisStep(
        "descriptive.modalities", "descriptive", "Vị trí và hội thoại", "event / customer",
        ("geo", "dialog"), ("geo_summary", "location_categories", "dialog_summary", "embedding_quality"),
        depends_on=("descriptive.quality",),
    ),
    AnalysisStep(
        "descriptive.targets", "descriptive", "Phân bố bốn nhãn", "customer-month / customer",
        ("targets", "clients"),
        ("target_prevalence", "target_by_month_fold", "target_cooccurrence", "customer_positive_history"),
        depends_on=("descriptive.quality",),
    ),
    AnalysisStep(
        "features.customers", "features", "Tạo bảng khách hàng", "customer",
        ("clients", "transactions", "geo", "dialog"), ("customer_feature_checks",),
        depends_on=("descriptive.quality", "descriptive.temporal"),
        creates_artifacts=("customer_features",),
    ),
    AnalysisStep(
        "descriptive.customers", "descriptive", "Phân bố hoạt động khách hàng", "customer",
        ("customer_features",), ("customer_activity", "modality_coverage"),
        depends_on=("features.customers",), required_artifacts=("customer_features",),
    ),
    AnalysisStep(
        "features.customer_month", "features", "Tạo bảng khách hàng–tháng", "customer-month",
        ("targets", "label_windows", "transactions", "geo", "dialog"), ("customer_month_checks",),
        depends_on=("descriptive.quality", "descriptive.temporal", "descriptive.targets"),
        required_artifacts=("label_windows",), creates_artifacts=("customer_month_features",),
    ),
    AnalysisStep(
        "exploration.temporal", "exploration", "Mẫu hình hoạt động theo thời gian", "source / period",
        ("transactions", "geo", "dialog"), ("temporal_patterns", "activity_changes"),
        depends_on=("descriptive.temporal",),
    ),
    AnalysisStep(
        "exploration.segments", "exploration", "Khác biệt giữa nhóm khách", "customer / customer-month",
        ("customer_features", "customer_month_features"), ("segment_profiles", "segment_target_rates"),
        depends_on=("features.customers", "features.customer_month"),
        required_artifacts=("customer_features", "customer_month_features"),
    ),
    AnalysisStep(
        "exploration.modalities", "exploration", "Quan hệ giữa các nguồn", "customer / customer-month",
        ("customer_features", "customer_month_features"), ("modality_overlap", "modality_target_rates"),
        depends_on=("descriptive.customers", "features.customer_month"),
        required_artifacts=("customer_features", "customer_month_features"),
    ),
    AnalysisStep(
        "exploration.missingness", "exploration", "Mẫu hình thiếu thuộc tính và không có log", "source / period / group",
        ("transactions", "geo", "dialog", "customer_features"),
        ("missingness_patterns", "missingness_by_group"),
        depends_on=("descriptive.quality", "descriptive.temporal", "features.customers"),
        required_artifacts=("customer_features",),
    ),
    AnalysisStep(
        "exploration.target_relationships", "exploration", "Hành vi lịch sử và nhãn", "customer-month",
        ("customer_month_features",), ("feature_by_target", "target_rates_by_feature_group"),
        depends_on=("features.customer_month", "descriptive.targets"),
        required_artifacts=("customer_month_features",),
    ),
    AnalysisStep(
        "exploration.correlations", "exploration", "Quan hệ giữa biến tổng hợp", "customer-month",
        ("customer_month_features",), ("feature_correlations", "correlated_feature_pairs"),
        depends_on=("features.customer_month",), required_artifacts=("customer_month_features",),
    ),
    AnalysisStep(
        "exploration.stability", "exploration", "Độ ổn định của các mẫu hình", "month / fold / history window",
        ("customer_month_features", "section_results"), ("pattern_stability", "small_group_flags"),
        depends_on=(
            "exploration.temporal", "exploration.segments", "exploration.modalities",
            "exploration.missingness", "exploration.target_relationships", "exploration.correlations",
        ),
        required_artifacts=("customer_month_features",),
    ),
)


def build_plan(config: MBDMiniConfig | None = None) -> dict:
    """Trả kế hoạch có thể xuất JSON/Markdown, không đọc dữ liệu hoặc ghi file."""
    config = config if config is not None else MBDMiniConfig()
    return {
        "package": "mbd_mini_eda",
        "implementation_status": "implemented",
        "reads_raw_data": False,
        "config": config.to_dict(),
        "tables": [asdict(table) for table in TABLES],
        "steps": [step.to_dict() for step in STEPS],
    }


def run_analysis(config: MBDMiniConfig, *, label_windows=None, progress=None) -> AnalysisResult:
    """Tính các kết quả; không xuất artifact cá nhân hoặc sửa dữ liệu nguồn.

    Nếu chưa có mapping nhãn, chỉ bỏ qua các bước cần customer_month_features;
    thống kê mô tả, xu hướng và missingness vẫn được tính đầy đủ.
    """
    from datetime import datetime, timezone
    import importlib
    import platform
    import pandas as pd
    import pyarrow
    from .io import source_paths, read_reference_tables, load_label_windows, validate_label_windows

    with AnalysisContext(config, source_paths(config.data_root)) as context:
        context.artifacts.update(read_reference_tables(context.sources))
        windows = (validate_label_windows(label_windows, config.timezone) if label_windows is not None
                   else load_label_windows(config, context.artifacts["targets"]))
        context.runtime["label_alignment"] = "provided_windows" if label_windows is not None else config.label_alignment
        if windows is not None:
            context.artifacts["label_windows"] = windows
        for step in STEPS:
            if progress is not None:
                progress(step.key)
            missing = set(step.required_artifacts) - set(context.artifacts)
            if missing:
                section = SectionResult(step.key, findings=["Bỏ qua do thiếu artifact: " + ", ".join(sorted(missing))],
                                        metadata={"status": "skipped", "missing_artifacts": sorted(missing)})
            else:
                module = importlib.import_module("mbd_mini_eda." + step.key)
                section = module.run(context)
                section.metadata["status"] = "completed"
                section.metadata.setdefault("grain", step.grain)
            context.section_results[step.key] = section
        store = context.runtime.get("engine")
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "pandas": pd.__version__, "pyarrow": pyarrow.__version__, "config": config.to_dict(),
            "label_alignment": "provided_windows" if label_windows is not None else config.label_alignment,
            "event_scope": "prefix" if config.max_batches_per_source is not None else "full",
            "sources": {name: {"scanned_rows": state.rows, "sample_rows": len(state.sample.frame),
                "sample_seed": config.random_seed + index, "sampling": "uniform_bottom_k_random_priorities"}
                for index, (name, state) in enumerate(store.stats.items())} if store else {},
            "sections": {name: item.metadata for name, item in context.section_results.items()},
            "label_windows": windows.astype(str).to_dict("records") if windows is not None else None,
        }
        return AnalysisResult(config, dict(context.section_results), metadata)
