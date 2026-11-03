"""Pairwise-complete correlation with effective pair sizes."""
from ..utils.stats import correlations


def spearman(frame, columns):
    return correlations(frame, list(columns))
