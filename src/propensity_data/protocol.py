"""Offline evaluation calendar with explicit assumed label availability."""
from datetime import date, timedelta

from .config import month_start


def next_month(day: date) -> date:
    return date(day.year + (day.month == 12), day.month % 12 + 1, 1)


def evaluation_protocol(config, months: list[str]) -> dict:
    expected = []
    day = month_start(config.first_month)
    while day <= month_start(config.last_month):
        expected.append(day.strftime("%Y-%m"))
        day = next_month(day)
    if months != expected:
        raise ValueError("Targets do not cover every configured month")
    available = {
        m: next_month(month_start(m)) + timedelta(days=config.label_delay_days)
        for m in months
    }

    def training_at(cutoff_month):
        cutoff = month_start(cutoff_month)
        result = [m for m in months if m < cutoff_month and available[m] <= cutoff]
        if not result:
            raise ValueError(f"No labels available to train at {cutoff_month}")
        return result

    calibration = config.calibration_month
    test = config.test_month
    # Do not silently retain an impossible calibration schedule when delay > 0.
    if available[calibration] > month_start(test):
        raise ValueError("Calibration labels unavailable at test cutoff; choose an earlier calibration month")
    for m in config.validation_months:
        if available[m] > month_start(calibration):
            raise ValueError("Validation labels unavailable at final base-model fit")
    return {
        "timezone": "UTC",
        "feature_rule": "event_time < cutoff; fixed-day windows are [cutoff - days, cutoff)",
        "label_availability": "Assumption, not verified publisher SLA",
        "label_delay_days": config.label_delay_days,
        "assumed_label_available_at": {m: d.isoformat() + "T00:00:00Z" for m, d in available.items()},
        "rolling_validation": [{"train": training_at(m), "validate": m} for m in config.validation_months],
        "final_train": training_at(calibration),
        "base_model_fit_at": calibration + "-01T00:00:00Z",
        "calibration": calibration,
        "test": test,
        "refit_after_calibration": False,
        "primary_budget_fraction": 0.01,
        "secondary_budget_fractions": [0.005, 0.05],
        "metrics": ["recall_at_k", "precision_at_k", "lift_at_k", "average_precision",
                    "roc_auc", "log_loss", "brier", "calibration"],
        "bootstrap_unit": "client_id",
        "note": "Existing-customer temporal evaluation; never use client_id or source fold as predictors.",
    }
