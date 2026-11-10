"""Audit saved validation predictions; export aggregate diagnostics without fitting or opening test."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from propensity_baseline.metrics import ranked_indices


BUDGETS = (.005, .01, .02, .05, .10)
MONTHS = ('2022-09', '2022-10', '2022-11')
LABELS = ['y_any', 'target_1', 'target_2', 'target_3', 'target_4']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def top_metrics(y, selected):
    """Metrics of a global selection, optionally restricted to a customer subgroup."""
    n, positives, k = len(y), int(y.sum()), int(selected.sum())
    hits = int(y[selected].sum())
    precision = hits / k if k else None
    return dict(rows=n, positives=positives, selected=k, hits=hits,
                precision=precision, recall=hits / positives if positives else None,
                lift=precision / (positives / n) if precision is not None and positives else None)


def review(run, source, out, monthly_path):
    state = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    if state['status'] != 'fitted' or state['test_opened']:
        raise ValueError('This review requires the fitted, unopened-test baseline run')
    inputs = [run / 'manifest.json', run / 'selection.json', run / 'validation_metrics.json',
              run / 'validation_predictions.parquet', run / 'rule_validation/rule_predictions.parquet',
              monthly_path]
    from catboost import CatBoostClassifier
    model_path = run / 'final/model.cbm'
    if digest(model_path) != state['model_sha256']:
        raise ValueError('Final model hash differs from fitted manifest')
    model = CatBoostClassifier().load_model(str(model_path))
    learning = model.get_evals_result().get('learn', {}).get('Logloss', [])
    if len(learning) != model.tree_count_:
        raise ValueError('Expected one saved training Logloss value per final-model tree')
    inputs.append(model_path)
    source_protocol = json.loads((source / 'protocol.json').read_text(encoding='utf-8'))
    inputs.append(source / 'protocol.json')
    prediction = pd.read_parquet(inputs[3])
    rule = pd.read_parquet(inputs[4])
    if set(prediction.month) != set(MONTHS) or set(rule.month) != set(MONTHS):
        raise ValueError('Only saved September-November validation predictions are permitted')
    if prediction.duplicated(['client_id', 'month', 'model']).any() or rule.duplicated(['client_id', 'month']).any():
        raise ValueError('Duplicate prediction keys')
    selection = json.loads(inputs[1].read_text(encoding='utf-8'))
    seed = state['config']['seed']
    winner = selection['variant']
    if not rule.rule_available.all() or rule.prior_y_any.isna().any():
        raise ValueError('Review requires the full common rule/model cohort')
    base = prediction.loc[prediction.model.eq(winner), ['client_id', 'month', 'y_any']]
    aligned = base.merge(rule[['client_id', 'month', 'y_any']], on=['client_id', 'month'],
                         validate='one_to_one', suffixes=('', '_rule'))
    if len(aligned) != len(base) or not aligned.y_any.eq(aligned.y_any_rule).all():
        raise ValueError('Rule and selected model outcomes differ')

    # Read only labels BEFORE validation cutoffs, never December or January features.
    history = {}
    for month in pd.period_range('2022-02', '2022-10', freq='M').astype(str):
        path = source / f'features/month={month}/part.parquet'
        part = pd.read_parquet(path, columns=['client_id', *LABELS]).set_index('client_id')
        if not part.index.is_unique:
            raise ValueError('Duplicate historical label keys')
        history[month] = part
        inputs.append(path)
    before = {str(p.relative_to(run.parent.parent)) if p.is_relative_to(run.parent.parent) else p.name: digest(p)
              for p in inputs}
    records, segments, overlaps, gains = [], [], [], []
    for month in MONTHS:
        actual = base.loc[base.month.eq(month)].set_index('client_id').y_any
        ids = actual.index.to_numpy()
        y = actual.to_numpy(dtype=int)
        prior_month = str(pd.Period(month, freq='M') - 1)
        cutoff = pd.Timestamp(month + '-01', tz='UTC')
        available = source_protocol['assumed_label_available_at']
        if any(pd.Timestamp(available[m]) > cutoff for m in history if m < month):
            raise ValueError('Historical labels unavailable at validation cutoff')
        lag = history[prior_month].reindex(ids).y_any
        saved_lag = rule.loc[rule.month.eq(month)].set_index('client_id').reindex(ids).prior_y_any
        if not lag.eq(saved_lag).all():
            raise ValueError('Historical lag and saved rule do not match')
        past = [frame.reindex(ids).y_any for m, frame in history.items() if m < month]
        known = pd.concat(past, axis=1).notna().all(axis=1).to_numpy()
        ever = pd.concat(past, axis=1).eq(1).any(axis=1).to_numpy()
        if not known.all():
            raise ValueError('Cannot call a customer never-observed-positive with incomplete history')
        masks = {'bought_previous_month': lag.eq(1).to_numpy(),
                 'no_purchase_previous_month': lag.eq(0).to_numpy(),
                 'ever_observed_purchase': ever, 'no_observed_purchase_since_2022_02': ~ever}
        scores = {}
        for model in ('multi_recent', 'multi_history'):
            frame = prediction.loc[prediction.month.eq(month) & prediction.model.eq(model)].set_index('client_id')
            if len(frame) != len(ids) or not frame.reindex(ids).y_any.eq(actual).all():
                raise ValueError('Prediction cohorts/outcomes differ')
            scores[model] = frame.reindex(ids).score.to_numpy()
        scores['previous_month_purchase'] = saved_lag.to_numpy()
        orders = {model: ranked_indices(ids, score, seed) for model, score in scores.items()}
        for model, order in orders.items():
            cumulative = np.cumsum(y[order])
            fractions = np.unique(np.r_[np.linspace(.001, 1., 250), BUDGETS])
            for fraction in fractions:
                k = math.ceil(fraction * len(y))
                gains.append(dict(month=month, model=model, selected_fraction=k/len(y),
                                  recall=float(cumulative[k-1]/y.sum())))
            for fraction in BUDGETS:
                selected = np.zeros(len(y), dtype=bool)
                selected[order[:math.ceil(fraction * len(y))]] = True
                records.append(dict(month=month, model=model, budget=fraction, **top_metrics(y, selected)))
                if fraction == .01:
                    for name, mask in masks.items():
                        segments.append(dict(month=month, model=model, segment=name,
                                             selection_policy='global_top_1pct_then_restrict',
                                             **top_metrics(y[mask], selected[mask])))
        k = math.ceil(.01 * len(y))
        model_set, rule_set = set(orders[winner][:k]), set(orders['previous_month_purchase'][:k])
        intersection = model_set & rule_set
        ml_mask = np.zeros(len(y), dtype=bool)
        ml_mask[list(model_set)] = True
        overlaps.append(dict(month=month, k=k, common=len(intersection),
                             overlap_fraction=len(intersection)/k,
                             jaccard=len(intersection)/len(model_set | rule_set),
                             ml_selected_prior_buyers=int((ml_mask & masks['bought_previous_month']).sum()),
                             ml_selected_observed_ever_buyers=int((ml_mask & ever).sum()),
                             ml_hits_prior_buyers=int(y[ml_mask & masks['bought_previous_month']].sum()),
                             ml_hits_no_prior_purchase=int(y[ml_mask & masks['no_purchase_previous_month']].sum()),
                             ml_hits_no_observed_purchase=int(y[ml_mask & ~ever].sum()),
                             common_hits=int(y[list(intersection)].sum()),
                             ml_only_hits=int(y[list(model_set-rule_set)].sum()),
                             rule_only_hits=int(y[list(rule_set-model_set)].sum())))
    # Cross-check new budget calculations against the original persisted metrics.
    metrics = json.loads((run / 'validation_metrics.json').read_text(encoding='utf-8'))
    for row in records:
        if row['model'] == 'previous_month_purchase' or row['budget'] not in (.005, .01, .05):
            continue
        original = next(m for m in metrics if m['month'] == row['month'] and m['model'] == row['model'])
        old = original['top_k'][str(row['budget'])]
        assert row['hits'] == old['hits'] and row['selected'] == old['k']
        assert np.isclose(row['recall'], old['recall'])
    table = pd.DataFrame(records)
    macro = table.groupby(['model', 'budget'], as_index=False)[['precision', 'recall', 'lift']].mean()
    monthly = pd.read_csv(monthly_path)
    # This is pre-existing label-count metadata, not test predictions/performance.
    if monthly.month.tolist() != pd.period_range('2022-02', '2023-01', freq='M').astype(str).tolist():
        raise ValueError('Unexpected audited label-summary months')
    if {str(p.relative_to(run.parent.parent)) if p.is_relative_to(run.parent.parent) else p.name: digest(p)
        for p in inputs} != before:
        raise ValueError('Inputs changed during review')
    out.mkdir(parents=True, exist_ok=False)
    table.to_csv(out/'budget_metrics.csv', index=False)
    macro.to_csv(out/'budget_macro.csv', index=False)
    pd.DataFrame(segments).to_csv(out/'segments_top1.csv', index=False)
    pd.DataFrame(overlaps).to_csv(out/'overlap_top1.csv', index=False)
    pd.DataFrame(gains).to_csv(out/'cumulative_gain.csv', index=False)
    monthly.to_csv(out/'label_prevalence.csv', index=False)
    pd.DataFrame({'trees': np.arange(1, len(learning)+1), 'train_logloss': learning}).to_csv(
        out/'final_training_logloss.csv', index=False)
    manifest = dict(scope='Saved validation predictions only; no fitting or test prediction evaluation',
                    months=list(MONTHS), winner=winner, seed=seed, input_sha256=before,
                    history_definition='Observed labels from 2022-02 strictly before each cutoff; not lifetime purchases',
                    segment_selection='Select global top 1%, then restrict; subgroup denominators kept separate',
                    validation_selection_bias='Exploratory: these months already selected the model',
                    learning_curve='Final base training Logloss only; no saved fold validation/AP curves; not evidence of underfit',
                    test_summary='January aggregate label counts were already published; no test prediction file read')
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    plot_diagnostics(pd.DataFrame(gains), monthly, learning, out)
    print(macro.to_string(index=False))
    print(pd.DataFrame(overlaps).to_string(index=False))


def plot_diagnostics(gains, monthly, learning, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    names = {'multi_history': 'Behavior: recent + older', 'multi_recent': 'Behavior: recent',
             'previous_month_purchase': 'Previous-month rule'}
    for model, group in gains.groupby('model', sort=True):
        macro = group.groupby('selected_fraction', as_index=False).recall.mean()
        axes[0].plot(macro.selected_fraction*100, macro.recall*100, label=names[model])
    axes[0].plot([0, 10], [0, 10], '--', color='gray', label='Random expectation')
    axes[0].set(xlim=(0, 10), ylim=(0, 50), xlabel='Customers selected (%)',
                ylabel='Buyers captured (%)', title='Cumulative gain: mean Sep-Nov 2022')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2)
    axes[1].plot(np.arange(len(monthly)), monthly.positive_rate*100, 'o-', color='#267969')
    axes[1].set(xticks=np.arange(len(monthly)), xticklabels=monthly.month.str[2:],
                ylabel='Observed monthly purchase rate (%)', title='Previously audited target prevalence')
    axes[1].tick_params(axis='x', rotation=50)
    axes[1].grid(alpha=.2)
    fig.savefig(out/'diagnostics.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    trees = np.arange(1, len(learning)+1)
    for axis in axes:
        axis.plot(trees, learning, color='#267969')
        axis.set(xlabel='Trees', ylabel='Training Logloss')
        axis.grid(alpha=.2)
    axes[0].set_title('Saved final base: train Feb-Nov 2022')
    axes[1].set(xlim=(50, len(learning)), ylim=(min(learning)*.995, learning[49]*1.005),
                title='Zoom after 50 trees; no validation curve')
    fig.savefig(out/'final_training_logloss.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('outputs/propensity_baseline_geo_dialog_v1'))
    parser.add_argument('--source', type=Path, default=Path('outputs/propensity_calendar_v2'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--monthly-summary', type=Path,
                        default=Path('docs/propensity_feasibility/any_product/monthly.csv'))
    args = parser.parse_args()
    review(args.run.resolve(), args.source.resolve(), args.output.resolve(), args.monthly_summary.resolve())
