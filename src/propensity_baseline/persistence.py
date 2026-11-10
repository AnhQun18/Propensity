"""Previous calendar month's purchase as an explicitly available hard rule."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from propensity_data.storage import write_json
from .data import previous_month, read_json
from .metrics import evaluate, paired_recall_bootstrap, ranked_indices


RULE_NAME = 'previous_month_purchase'


def persistence_predictions(config, frame, schema, manifest, protocol):
    """Forecast y(M) = y(M-1); never replace unavailable prior labels with 0."""
    from .runner import read_months
    columns = [c for c in ['client_id','month','y_any','cold_start','trx_ever_seen_source',
                           'geo_ever_seen_source','dialog_ever_seen_source'] if c in frame]
    if not {'client_id','month','y_any'} <= set(columns):
        raise ValueError('Rule scoring requires client/month keys and current evaluation label')
    if frame.duplicated(['client_id','month']).any():
        raise ValueError('Duplicate rule scoring keys')
    parts = []
    provided = {r['month'] for r in manifest['months']}
    for month, current in frame.groupby('month', sort=True):
        result = current[columns].copy()
        prior_month = previous_month(month).strftime('%Y-%m')
        result['prior_month'] = prior_month
        result['prior_y_any'] = np.nan
        reason = 'prior_month_not_provided'
        cutoff = pd.Timestamp(month+'-01', tz='UTC')
        available_at = protocol['assumed_label_available_at'].get(prior_month)
        if prior_month in provided:
            reason = 'prior_labels_not_available_at_cutoff'
            if available_at is not None and pd.Timestamp(available_at) <= cutoff:
                # One approved feature is projected only to reuse source/label checks;
                # it never affects the rule prediction.
                prior = read_months(config, [prior_month], schema['predictors'][:1], schema, manifest)
                lookup = prior.set_index('client_id').y_any
                result['prior_y_any'] = result.client_id.map(lookup).to_numpy(dtype=float, na_value=np.nan)
                reason = 'prior_label_unknown_or_customer_absent'
        result['rule_available'] = result.prior_y_any.notna()
        result['unavailable_reason'] = np.where(result.rule_available, '', reason)
        result['model'] = RULE_NAME
        result['score'] = result.prior_y_any
        # This is a hard 0/1 decision, not a fitted/calibrated probability.
        result['hard_prediction'] = result.prior_y_any
        parts.append(result)
    return pd.concat(parts, ignore_index=True)


def classification_counts(y, selected):
    y, selected = np.asarray(y, dtype=int), np.asarray(selected, dtype=bool)
    tp = int(((y==1)&selected).sum())
    fp = int(((y==0)&selected).sum())
    fn = int(((y==1)&~selected).sum())
    tn = int(((y==0)&~selected).sum())
    return dict(selected=int(selected.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def compare_predictions(model_predictions, rule_predictions, *, seed):
    """Same cohort, same top-K, plus the rule's natural decision-set budget."""
    if rule_predictions.duplicated(['client_id','month']).any():
        raise ValueError('Duplicate rule prediction keys')
    if model_predictions.duplicated(['model','client_id','month']).any():
        raise ValueError('Duplicate model prediction keys')
    rule = rule_predictions[['client_id','month','y_any','prior_y_any','rule_available']].rename(columns={'y_any':'rule_actual'})
    joined = model_predictions.merge(rule, on=['client_id','month'], how='left', validate='many_to_one')
    if len(joined) != len(model_predictions) or joined.rule_available.isna().any():
        raise ValueError('Model and rule scoring populations differ')
    same = joined.y_any.eq(joined.rule_actual) | (joined.y_any.isna() & joined.rule_actual.isna())
    if not same.all():
        raise ValueError('Model and rule evaluation labels differ')
    records, paired = [], []
    for (model, month), full in joined.groupby(['model','month'], sort=True):
        current_known = full.y_any.notna()
        subset = full.loc[current_known & full.rule_available]
        coverage = dict(model=model, month=month, scoring_rows=len(full),
                        current_labelled_rows=int(current_known.sum()), comparison_rows=len(subset),
                        excluded_unavailable_prior=int((current_known & ~full.rule_available).sum()))
        if not len(subset):
            records.append(dict(**coverage, status='no_common_labels'))
            continue
        y, ids = subset.y_any.to_numpy(dtype=int), subset.client_id.to_numpy()
        model_metrics = evaluate(y, subset.p_any.to_numpy(), ids, seed=seed, ranking_score=subset.score.to_numpy())
        rule_metrics = evaluate(y, subset.prior_y_any.to_numpy(), ids, seed=seed)
        # Don't present the binary rule's hard decisions as calibrated probabilities.
        rule_metrics = {k:v for k,v in rule_metrics.items() if k not in
                        ('log_loss','brier','mean_prediction','calibration_bins')}
        rule_selected = subset.prior_y_any.eq(1).to_numpy()
        budget = int(rule_selected.sum())
        selected = np.zeros(len(subset), dtype=bool)
        selected[ranked_indices(ids, subset.score.to_numpy(), seed)[:budget]] = True
        records.append(dict(**coverage, status='compared', model_metrics=model_metrics,
                            rule_ranking_metrics=rule_metrics,
                            rule_natural_decisions=classification_counts(y, rule_selected),
                            model_at_rule_budget=classification_counts(y, selected)))
        left = subset[['client_id','month','y_any','score','p_any']].copy()
        left['model'] = model
        paired.append(left)
    return records, pd.concat(paired, ignore_index=True) if paired else pd.DataFrame()


def comparison_table(records):
    rows = []
    for record in records:
        if record['status'] != 'compared':
            continue
        common = {k:record[k] for k in ['model','month','comparison_rows','excluded_unavailable_prior']}
        for name, metrics in [('model',record['model_metrics']),('rule',record['rule_ranking_metrics'])]:
            top = metrics['top_k']['0.01']
            rows.append(dict(**common, method=name, k=top['k'], hits=top['hits'],
                             precision_at_1pct=top['precision'], recall_at_1pct=top['recall'],
                             lift_at_1pct=top['lift'], average_precision=metrics['average_precision']))
    return pd.DataFrame(rows)


def compare_rule(config, *, stage='validation'):
    """Compare saved predictions; validation never opens the held-out month."""
    from .runner import locked_run, source_contract
    if stage not in ('validation','test'):
        raise ValueError('Comparison stage must be validation or test')
    state = read_json(config.output_dir / 'manifest.json')
    allowed = ('tuned','fitted','tested') if stage=='validation' else ('tested',)
    if state.get('status') not in allowed:
        raise ValueError('Requested comparison predictions are not yet available')
    _, manifest, schema, protocol, _ = locked_run(config, state['status'])
    path = config.output_dir / ('validation_predictions.parquet' if stage=='validation' else 'test/predictions.parquet')
    predictions = pd.read_parquet(path)
    predictions = predictions.loc[predictions.model.isin(config.variants)].copy()
    if predictions.empty:
        raise ValueError('No behavior model predictions to compare')
    expected_months = {f['validate'] for f in protocol['rolling_validation']} if stage=='validation' else {protocol['test']}
    if set(predictions.month) != expected_months:
        raise ValueError('Saved predictions differ from comparison calendar')
    keys = predictions.drop_duplicates(['client_id','month'])
    rule = persistence_predictions(config, keys, schema, manifest, protocol)
    records, paired = compare_predictions(predictions, rule, seed=config.seed)
    out = config.output_dir / f'rule_{stage}'
    out.mkdir(exist_ok=False)
    report = dict(status='running', stage=stage, rule='y_hat(M)=y_any(M-1)',
                  label_delay_days=protocol['label_delay_days'], timezone=protocol['timezone'],
                  test_opened=state['test_opened'],
                  missing_prior_policy='Unknown/absent/unavailable labels remain unknown; compare the same eligible cohort',
                  probability_note='Hard rule has no calibrated probability; no rule log loss reported',
                  tie_break='SHA256(seed:client_id), then client_id; rule positives rank before negatives',
                  started_at=datetime.now(timezone.utc).isoformat())
    write_json(out / 'manifest.json', report)
    try:
        rule.to_parquet(out / 'rule_predictions.parquet', index=False)
        write_json(out / 'monthly_comparison.json', records)
        table = comparison_table(records)
        table.to_csv(out / 'monthly_comparison.csv', index=False)
        comparisons = []
        for model, group in paired.groupby('model',sort=True):
            rule_rows = rule.merge(group[['client_id','month']], on=['client_id','month'], validate='one_to_one')
            rule_rows = rule_rows[['client_id','month','y_any','model','score']]
            comparison = paired_recall_bootstrap(pd.concat([group,rule_rows],ignore_index=True),model,RULE_NAME,
                                                seed=config.seed,replicates=config.bootstrap_replicates)
            comparison['interpretation'] = 'Validation is exploratory after model selection' if stage=='validation' else 'Conditional on fitted models and this held-out month'
            comparisons.append(comparison)
        write_json(out / 'paired_comparison.json',comparisons)
        if source_contract(config)[3] != state['source_inventory']:
            raise ValueError('Feature inputs changed during rule comparison')
        report.update(status='ready', finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(out / 'manifest.json',report)
    return out
