"""Advanced EDA for both general tabular and longitudinal panel datasets."""

from .bipartite import analyze_bipartite_graph
from .categorical import periodic_categorical_profile
from .decomposition import period_change_decomposition
from .drift import compare_distributions, population_stability_index, temporal_drift_report
from .event_study import run_event_study, run_stratified_event_study
from .inference import cluster_bootstrap_rates, compare_two_groups, wilson_ci
from .relationships import (
    categorical_associations,
    compute_vif,
    correlation_analysis,
    target_signal_scan,
    woe_iv_table,
)
from .target_panel import (
    build_multi_horizon_targets,
    buyer_composition_and_transitions,
    compute_surge_ratio,
    stratified_lift_table,
)

__all__ = [
    "correlation_analysis",
    "categorical_associations",
    "compute_vif",
    "woe_iv_table",
    "target_signal_scan",
    "population_stability_index",
    "compare_distributions",
    "temporal_drift_report",
    "wilson_ci",
    "cluster_bootstrap_rates",
    "compare_two_groups",
    "periodic_categorical_profile",
    "analyze_bipartite_graph",
    "run_event_study",
    "run_stratified_event_study",
    "period_change_decomposition",
    "build_multi_horizon_targets",
    "stratified_lift_table",
    "buyer_composition_and_transitions",
    "compute_surge_ratio",
]
