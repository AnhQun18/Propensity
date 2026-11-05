"""Kế hoạch và thực thi phân tích; ghi file chỉ qua reporting.export_report."""

from dataclasses import asdict, dataclass

from .config import MBDMiniConfig
from .models import AnalysisContext, AnalysisResult
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
        "descriptive.customers", "descriptive", "Phân bố hoạt động khách hàng", "customer",
        ("clients", "transactions", "geo", "dialog"),
        ("customer_activity", "modality_coverage", "customer_aggregation_checks"),
        depends_on=("descriptive.quality", "descriptive.temporal"),
    ),
    AnalysisStep(
        "descriptive.targets", "descriptive", "Phân bố bốn nhãn", "customer-month / customer",
        ("targets", "clients"),
        ("target_prevalence", "target_by_month_fold", "target_cooccurrence", "customer_positive_history"),
        depends_on=("descriptive.quality",),
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


def run_analysis(config: MBDMiniConfig, *, progress=None) -> AnalysisResult:
    """Chạy đúng bảy bước mô tả, dùng chung một lần quét sự kiện."""
    from datetime import datetime, timezone
    import importlib
    import platform
    from time import perf_counter
    import pandas as pd
    import pyarrow
    from .io import source_paths, read_reference_tables

    started = perf_counter()
    with AnalysisContext(config, source_paths(config.data_root)) as context:
        context.runtime["progress"] = progress
        context.artifacts.update(read_reference_tables(context.sources))
        for step in STEPS:
            if progress is not None:
                progress(step.key)
            step_started = perf_counter()
            module = importlib.import_module("mbd_mini_eda." + step.key)
            section = module.run(context)
            section.metadata["status"] = "completed"
            section.metadata.setdefault("grain", step.grain)
            section.metadata["duration_seconds"] = round(perf_counter() - step_started, 3)
            context.section_results[step.key] = section
            if progress is not None:
                progress(f"  Hoàn thành {step.key}: {section.metadata['duration_seconds']:.1f}s")
        store = context.runtime.get("engine")
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "pandas": pd.__version__, "pyarrow": pyarrow.__version__, "config": config.to_dict(),
            "analysis_scope": "seven_descriptive_steps", "duration_seconds": round(perf_counter() - started, 3),
            "event_scope": "prefix" if config.max_batches_per_source is not None else "full",
            "sources": {name: {"scanned_rows": state.rows, "sample_rows": len(state.sample.frame),
                "sample_seed": config.random_seed + index, "sampling": "uniform_bottom_k_random_priorities",
                "duration_seconds": state.duration_seconds, "batch_size": state.batch_size,
                "batches": state.batches}
                for index, (name, state) in enumerate(store.stats.items())} if store else {},
            "sections": {name: item.metadata for name, item in context.section_results.items()},
        }
        return AnalysisResult(config, dict(context.section_results), metadata)
