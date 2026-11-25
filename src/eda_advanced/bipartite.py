"""Bipartite entity-item sparse graph: event-weighted IDF, connected components, and TF-IDF SVD."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import bmat, coo_matrix
from scipy.sparse.csgraph import connected_components


def analyze_bipartite_graph(
    entity_ids: Any,
    edges_df: pd.DataFrame,
    *,
    entity_col: str,
    item_col: str,
    weight_col: str = "events",
    svd_components: int | None = 32,
    random_state: int = 42,
) -> dict[str, Any]:
    """Analyze sparse bipartite entity-item interactions (degree, IDF, graph components, TF-IDF SVD)."""
    ids = pd.Index(np.sort(pd.Index(entity_ids).astype(str).unique()))
    n_ent = len(ids)

    clean = edges_df[[entity_col, item_col, weight_col]].dropna().copy()
    clean[entity_col] = clean[entity_col].astype(str)
    clean[item_col] = clean[item_col].astype(str)
    clean = clean[clean[entity_col].isin(ids)]

    items = pd.Index(np.sort(clean[item_col].unique()))
    n_items = len(items)
    if n_ent == 0 or n_items == 0:
        return {
            "entity_metrics": pd.DataFrame({entity_col: ids, "degree": 0, "event_weighted_idf": np.nan}),
            "components": pd.DataFrame(columns=["component", "nodes", "entities"]),
            "embeddings": np.zeros((n_ent, 0), dtype=np.float32),
            "explained_variance_ratio": np.array([], dtype=float),
        }

    ei = ids.get_indexer(clean[entity_col])
    ii = items.get_indexer(clean[item_col])
    weights = clean[weight_col].to_numpy(dtype=float)

    mat = coo_matrix((weights, (ei, ii)), shape=(n_ent, n_items)).tocsr()
    degree = np.asarray((mat > 0).sum(axis=1)).ravel()
    item_degree = np.asarray((mat > 0).sum(axis=0)).ravel()
    idf = np.log((n_ent + 1.0) / (item_degree + 1.0)) + 1.0
    total_w = np.asarray(mat.sum(axis=1)).ravel()
    mean_idf = np.divide(mat @ idf, total_w, out=np.full(n_ent, np.nan), where=total_w > 0)

    adj_bin = (mat > 0).astype(np.int8)
    bip = bmat([[None, adj_bin], [adj_bin.T, None]], format="csr")
    n_comp, labels = connected_components(bip, directed=False)
    node_counts = np.bincount(labels, minlength=n_comp)
    ent_comp_counts = np.bincount(labels[:n_ent], minlength=n_comp)

    entity_metrics = pd.DataFrame({
        entity_col: ids,
        "degree": degree.astype(int),
        "event_weighted_idf": mean_idf,
        "component": labels[:n_ent].astype(int),
        "component_nodes": node_counts[labels[:n_ent]].astype(int),
        "component_entities": ent_comp_counts[labels[:n_ent]].astype(int),
    })
    comp_df = pd.DataFrame({
        "component": np.arange(n_comp, dtype=int),
        "nodes": node_counts.astype(int),
        "entities": ent_comp_counts.astype(int),
    })

    embeddings = np.zeros((n_ent, 0), dtype=np.float32)
    explained = np.array([], dtype=float)
    if svd_components is not None and svd_components > 0 and min(n_ent, n_items) >= 2:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfTransformer

        k = min(int(svd_components), min(n_ent, n_items) - 1)
        tfidf = TfidfTransformer().fit_transform(mat)
        svd = TruncatedSVD(n_components=k, random_state=random_state)
        embeddings = svd.fit_transform(tfidf).astype(np.float32)
        explained = svd.explained_variance_ratio_.astype(float)

    return {
        "entity_metrics": entity_metrics,
        "components": comp_df,
        "embeddings": embeddings,
        "explained_variance_ratio": explained,
    }
