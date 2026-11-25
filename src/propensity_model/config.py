"""Configuration loading and temporal validation for propensity_model."""
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


def month_to_index(month: str) -> int:
    year_s, mon_s = month.split("-")
    year, mon = int(year_s), int(mon_s)
    if not (1 <= mon <= 12):
        raise ValueError(f"Invalid YYYY-MM month: {month}")
    return year * 12 + (mon - 1)


def month_diff(later: str, earlier: str) -> int:
    return month_to_index(later) - month_to_index(earlier)


@dataclass(frozen=True)
class FoldSpec:
    name: str
    train_months: tuple[str, ...]
    eval_month: str
    horizon: int
    stage: str = "validation"  # "validation" | "calibration" | "test"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "train_months": list(self.train_months),
            "eval_month": self.eval_month,
            "horizon": self.horizon,
            "stage": self.stage,
        }


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    model_type: str  # prevalence | heuristic_rf | ridge_logreg | catboost | catboost_cold_fallback
    feature_set: str
    stage: str = "M2"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "model_type": self.model_type,
            "feature_set": self.feature_set,
            "stage": self.stage,
            "description": self.description,
        }


@dataclass(frozen=True)
class BaselineConfig:
    run_name: str
    processed_dir: Path
    output_dir: Path
    burn_in_months: tuple[str, ...] = ("2022-02", "2022-03", "2022-04")
    h1_validation_folds: tuple[FoldSpec, ...] = ()
    h3_validation_folds: tuple[FoldSpec, ...] = ()
    h1_final_train_months: tuple[str, ...] = (
        "2022-05", "2022-06", "2022-07", "2022-08", "2022-09", "2022-10", "2022-11"
    )
    h1_calibration_month: str = "2022-12"
    h1_test_month: str = "2023-01"
    h3_final_monthly_train_months: tuple[str, ...] = ("2022-05", "2022-06", "2022-07", "2022-08")
    h3_final_quarterly_train_months: tuple[str, ...] = ("2022-05", "2022-08")
    h3_test_month: str = "2022-11"
    experiments: tuple[ExperimentSpec, ...] = ()
    paired_comparisons: tuple[tuple[str, str], ...] = ()
    budget_fractions: tuple[float, ...] = (0.005, 0.01, 0.05)
    primary_budget_fraction: float = 0.01
    seed: int = 42
    bootstrap_replicates: int = 1000
    cold_start_alpha: float = 20.0
    duckdb_memory_limit: str = "2GB"
    duckdb_threads: int = 2
    catboost_params: dict[str, Any] = field(default_factory=lambda: {
        "iterations": 400,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 10.0,
        "rsm": 0.6,
        "border_count": 128,
        "thread_count": 4,
        "used_ram_limit": "3gb",
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "verbose": False,
    })
    learning_curve_checkpoints: tuple[int, ...] = (50, 100, 200, 300, 400)
    logreg_c: float = 1.0
    logreg_max_iter: int = 300

    def __post_init__(self):
        proc = self.processed_dir.resolve()
        out = self.output_dir.resolve()
        if out == proc or out in proc.parents or proc in out.parents:
            raise ValueError("output_dir must not overlap processed_dir")
        if self.cold_start_alpha < 0:
            raise ValueError("cold_start_alpha must be non-negative")
        if self.bootstrap_replicates < 10:
            raise ValueError("bootstrap_replicates must be >= 10")
        if not self.experiments:
            raise ValueError("At least one experiment must be configured")

        burn_set = set(self.burn_in_months)
        for fold in (*self.h1_validation_folds, *self.h3_validation_folds):
            assert_temporal_embargo(fold.train_months, fold.eval_month, fold.horizon, burn_set)
        assert_temporal_embargo(self.h1_final_train_months, self.h1_calibration_month, 1, burn_set)
        assert_temporal_embargo(self.h1_final_train_months, self.h1_test_month, 1, burn_set)
        assert_temporal_embargo(
            (self.h1_calibration_month,), self.h1_test_month, 1, burn_set
        )
        assert_temporal_embargo(self.h3_final_monthly_train_months, self.h3_test_month, 3, burn_set)
        assert_temporal_embargo(self.h3_final_quarterly_train_months, self.h3_test_month, 3, burn_set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "processed_dir": str(self.processed_dir),
            "output_dir": str(self.output_dir),
            "burn_in_months": list(self.burn_in_months),
            "h1_validation_folds": [f.to_dict() for f in self.h1_validation_folds],
            "h3_validation_folds": [f.to_dict() for f in self.h3_validation_folds],
            "h1_final_train_months": list(self.h1_final_train_months),
            "h1_calibration_month": self.h1_calibration_month,
            "h1_test_month": self.h1_test_month,
            "h3_final_monthly_train_months": list(self.h3_final_monthly_train_months),
            "h3_final_quarterly_train_months": list(self.h3_final_quarterly_train_months),
            "h3_test_month": self.h3_test_month,
            "experiments": [e.to_dict() for e in self.experiments],
            "paired_comparisons": [list(p) for p in self.paired_comparisons],
            "budget_fractions": list(self.budget_fractions),
            "primary_budget_fraction": self.primary_budget_fraction,
            "seed": self.seed,
            "bootstrap_replicates": self.bootstrap_replicates,
            "cold_start_alpha": self.cold_start_alpha,
            "duckdb_memory_limit": self.duckdb_memory_limit,
            "duckdb_threads": self.duckdb_threads,
            "catboost_params": dict(self.catboost_params),
            "learning_curve_checkpoints": list(self.learning_curve_checkpoints),
            "logreg_c": self.logreg_c,
            "logreg_max_iter": self.logreg_max_iter,
        }


def assert_temporal_embargo(
    train_months: tuple[str, ...] | list[str],
    eval_month: str,
    horizon: int,
    burn_in_months: set[str] | tuple[str, ...] | list[str],
) -> None:
    """Strictly enforce zero burn-in inclusion and H-month embargo between train and eval."""
    if horizon not in (1, 3):
        raise ValueError(f"Unsupported horizon {horizon}; expected 1 or 3")
    if not train_months:
        raise ValueError("train_months cannot be empty")
    burn = set(burn_in_months)
    if eval_month in burn:
        raise ValueError(f"Evaluation month {eval_month} overlaps burn-in months {sorted(burn)}")
    for m in train_months:
        if m in burn:
            raise ValueError(f"Training month {m} overlaps burn-in months {sorted(burn)}")
        gap = month_diff(eval_month, m)
        if gap < horizon:
            raise ValueError(
                f"Temporal embargo violation for H={horizon}: train month {m} is only "
                f"{gap} month(s) before eval month {eval_month} (requires >= {horizon} "
                f"so train labels close strictly before eval cutoff)"
            )


def load_config(path: str | Path) -> BaselineConfig:
    cfg_path = Path(path).resolve()
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    base = cfg_path.parent

    def resolve_path(val: str) -> Path:
        p = Path(val)
        return p if p.is_absolute() else (base / p).resolve()

    def parse_folds(items: list[dict[str, Any]], horizon: int) -> tuple[FoldSpec, ...]:
        return tuple(
            FoldSpec(
                name=item["name"],
                train_months=tuple(item["train_months"]),
                eval_month=item["eval_month"],
                horizon=horizon,
                stage=item.get("stage", "validation"),
            )
            for item in items
        )

    experiments = tuple(
        ExperimentSpec(
            experiment_id=e["experiment_id"],
            model_type=e["model_type"],
            feature_set=e["feature_set"],
            stage=e.get("stage", "M2"),
            description=e.get("description", ""),
        )
        for e in raw["experiments"]
    )
    paired = tuple((pair[0], pair[1]) for pair in raw.get("paired_comparisons", []))

    return BaselineConfig(
        run_name=raw["run_name"],
        processed_dir=resolve_path(raw["processed_dir"]),
        output_dir=resolve_path(raw["output_dir"]),
        burn_in_months=tuple(raw.get("burn_in_months", ["2022-02", "2022-03", "2022-04"])),
        h1_validation_folds=parse_folds(raw["h1_validation_folds"], horizon=1),
        h3_validation_folds=parse_folds(raw["h3_validation_folds"], horizon=3),
        h1_final_train_months=tuple(raw["h1_final_train_months"]),
        h1_calibration_month=raw["h1_calibration_month"],
        h1_test_month=raw["h1_test_month"],
        h3_final_monthly_train_months=tuple(raw["h3_final_monthly_train_months"]),
        h3_final_quarterly_train_months=tuple(raw["h3_final_quarterly_train_months"]),
        h3_test_month=raw["h3_test_month"],
        experiments=experiments,
        paired_comparisons=paired,
        budget_fractions=tuple(raw.get("budget_fractions", [0.005, 0.01, 0.05])),
        primary_budget_fraction=float(raw.get("primary_budget_fraction", 0.01)),
        seed=int(raw.get("seed", 42)),
        bootstrap_replicates=int(raw.get("bootstrap_replicates", 1000)),
        cold_start_alpha=float(raw.get("cold_start_alpha", 20.0)),
        duckdb_memory_limit=str(raw.get("duckdb_memory_limit", "2GB")),
        duckdb_threads=int(raw.get("duckdb_threads", 2)),
        catboost_params=dict(raw.get("catboost_params", {})),
        learning_curve_checkpoints=tuple(raw.get("learning_curve_checkpoints", [50, 100, 200, 300, 400])),
        logreg_c=float(raw.get("logreg_c", 1.0)),
        logreg_max_iter=int(raw.get("logreg_max_iter", 300)),
    )
