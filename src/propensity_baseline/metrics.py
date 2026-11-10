"""Monthly metrics with explicit deterministic ties and paired client bootstrap."""
import hashlib
import math

import numpy as np


def tie_keys(ids, seed):
    return np.array([int.from_bytes(hashlib.sha256(f'{seed}:{x}'.encode()).digest()[:8], 'big')
                     for x in ids], dtype=np.uint64)


def ranked_indices(ids, scores, seed):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError('Ranking scores must be finite')
    # Client string breaks the extremely unlikely hash collision deterministically.
    return np.lexsort((np.asarray(ids, dtype=str), tie_keys(ids, seed), -scores))


def evaluate(y, probabilities, ids, *, seed, budgets=(.005, .01, .05), ranking_score=None):
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    y, p = np.asarray(y), np.asarray(probabilities, dtype=float)
    if not len(y) or len(y) != len(p) or len(y) != len(ids) or not np.isin(y, (0, 1)).all():
        raise ValueError('Metrics require nonempty, aligned, known binary labels')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must be finite in [0,1]')
    order = ranked_indices(ids, p if ranking_score is None else ranking_score, seed)
    ranking = p if ranking_score is None else np.asarray(ranking_score, dtype=float)
    total = int(y.sum())
    prevalence = float(y.mean())
    result = dict(rows=len(y), positives=total, prevalence=prevalence,
                  average_precision=float(average_precision_score(y, ranking)) if total else None,
                  roc_auc=float(roc_auc_score(y, ranking)) if 0 < total < len(y) else None,
                  log_loss=float(log_loss(y, np.clip(p, 1e-15, 1-1e-15), labels=[0, 1])),
                  brier=float(brier_score_loss(y, p)), mean_prediction=float(p.mean()), top_k={})
    for fraction in budgets:
        if not 0 < fraction <= 1:
            raise ValueError('Budget must be in (0,1]')
        k = math.ceil(fraction * len(y))
        hits = int(y[order[:k]].sum())
        result['top_k'][str(fraction)] = dict(k=k, hits=hits, precision=hits/k,
                                             recall=hits/total if total else None,
                                             lift=(hits/k)/prevalence if total else None,
                                             random_expected_hits=k*prevalence,
                                             random_expected_recall=k/len(y) if total else None)
    # Fixed bins compare calibration across folds without choosing bins from labels.
    bins = np.minimum((p * 10).astype(int), 9)
    result['calibration_bins'] = [dict(bin=i, rows=int((bins == i).sum()),
                                     mean_prediction=float(p[bins == i].mean()),
                                     observed_rate=float(y[bins == i].mean()))
                                  for i in range(10) if (bins == i).any()]
    return result


def paired_recall_bootstrap(frame, variant, reference, *, seed, replicates):
    """Resample IDs once across months; rerank each resampled monthly population."""
    if replicates == 0:
        return {'replicates': 0, 'status': 'disabled'}
    # A row represents one ID/month/model. Resampling duplicates retain a unique
    # position key for ties while all months of a sampled client stay together.
    left = frame[frame.model == variant][['client_id', 'month', 'y_any', 'score']]
    right = frame[frame.model == reference][['client_id', 'month', 'y_any', 'score']]
    joined = left.merge(right, on=['client_id', 'month'], suffixes=('_a', '_b'), validate='one_to_one')
    if len(joined) != len(left) or len(joined) != len(right) or not joined.y_any_a.equals(joined.y_any_b):
        raise ValueError('Bootstrap model populations/labels differ')
    clients = sorted(joined.client_id.unique())
    if not clients:
        raise ValueError('No paired predictions')
    lookup = {c: i for i, c in enumerate(clients)}
    monthly = []
    for _, group in joined.groupby('month', sort=True):
        monthly.append((np.array([lookup[c] for c in group.client_id]), group.y_any_a.to_numpy(),
                        ranked_indices(group.client_id.to_numpy(), group.score_a, seed),
                        ranked_indices(group.client_id.to_numpy(), group.score_b, seed)))

    def delta(weights):
        values = []
        for indices, y, a, b in monthly:
            counts = weights[indices]
            n, positives = int(counts.sum()), float(counts @ y)
            if not n or not positives:
                continue
            k = math.ceil(.01*n)
            hits = []
            for order in (a, b):
                ordered = counts[order]
                before = np.cumsum(ordered) - ordered
                selected = np.minimum(ordered, np.maximum(k-before, 0))
                hits.append(float(selected @ y[order]))
            values.append((hits[0]-hits[1])/positives)
        return float(np.mean(values)) if values else None

    rng = np.random.default_rng(seed)
    samples = [delta(np.bincount(rng.integers(0, len(clients), len(clients)), minlength=len(clients)))
               for _ in range(replicates)]
    valid = [x for x in samples if x is not None]
    return dict(reference=reference, variant=variant, metric='macro monthly Recall@1% difference',
                estimate=delta(np.ones(len(clients), dtype=int)), replicates=replicates,
                valid_replicates=len(valid),
                interval_95=[float(x) for x in np.quantile(valid, [.025, .975])] if valid else None,
                scope='Conditional on fitted models and observed months; not uncertainty across unseen months')
