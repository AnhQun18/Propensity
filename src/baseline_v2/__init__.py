"""Baseline v2 package: protocol-decoupled monthly rolling OOC evaluation framework."""

from .spec import (
    load_spec,
    verify_or_create_freeze_lock,
    validate_feature_set,
    extract_feature_matrix,
)

__all__ = [
    "load_spec",
    "verify_or_create_freeze_lock",
    "validate_feature_set",
    "extract_feature_matrix",
]
