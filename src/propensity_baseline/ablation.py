"""Validation-only purchase-history ablation. Never opens calibration or test."""
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd

from propensity_data.storage import write_json
from .data import fingerprint
from .metrics import paired_recall_bootstrap
from .runner import (environment, fit_model, monthly_metrics, predict_frame,
                     read_months, sha256_file, source_contract)


PURCHASE_FEATURES = (
    'purchase_any_lag1', 'purchase_target_1_lag1', 'purchase_target_2_lag1',
    'purchase_target_3_lag1', 'purchase_target_4_lag1',
    'purchase_any_lag2', 'purchase_any_lag3', 'purchase_known_months_3',
    'purchase_positive_months_3', 'purchase_fraction_3',
    'purchase_known_months_history', 'purchase_positive_months_history',
    'purchase_fraction_history', 'purchase_months_since_last_positive',
    'purchase_has_previous_label',
)
VARIANTS = ('behavior_original', 'behavior', 'purchase', 'combined')


def purchase_features(rows, history, protocol):
    """Exact calendar lags, explicit availability, unknown stays unknown."""
    keys = ['client_id', 'month']
    if rows.duplicated(keys).any() or history.duplicated(keys).any():
        raise ValueError('Duplicate purchase-history keys')
    labels = ['y_any', *[f'target_{i}' for i in range(1, 5)]]
    values = history[labels]
    if (~(values.isna() | values.isin([0, 1]))).any().any():
        raise ValueError('Invalid purchase-history labels')
    outputs = []
    for month, current in rows.groupby('month', sort=False):
        cutoff = pd.Timestamp(month+'-01', tz='UTC')
        eligible = [m for m, at in protocol['assumed_label_available_at'].items()
                    if m < month and pd.Timestamp(at) <= cutoff]
        past = history.loc[history.month.isin(eligible)]
        ids = pd.Index(current.client_id)
        result = pd.DataFrame(index=current.index)
        recent = []
        for lag in (1, 2, 3):
            prior = str(pd.Period(month, freq='M') - lag)
            lookup = past.loc[past.month.eq(prior)].set_index('client_id')
            mapped = ids.map(lookup.y_any).to_numpy(dtype=float, na_value=np.nan)
            result[f'purchase_any_lag{lag}'] = mapped
            recent.append(mapped)
            if lag == 1:
                for i in range(1, 5):
                    result[f'purchase_target_{i}_lag1'] = ids.map(lookup[f'target_{i}']).to_numpy(dtype=float, na_value=np.nan)
        recent = np.stack(recent, axis=1)
        count = np.isfinite(recent).sum(axis=1)
        positives = np.nansum(recent, axis=1)
        result['purchase_known_months_3'] = count
        result['purchase_positive_months_3'] = np.where(count > 0, positives, np.nan)
        result['purchase_fraction_3'] = np.divide(positives, count, out=np.full(len(ids), np.nan), where=count > 0)
        grouped = past.groupby('client_id').y_any.agg(['count', 'sum'])
        count_all = ids.map(grouped['count']).to_numpy(dtype=float, na_value=np.nan)
        count_all = np.nan_to_num(count_all, nan=0.)
        positives_all = ids.map(grouped['sum']).to_numpy(dtype=float, na_value=np.nan)
        result['purchase_known_months_history'] = count_all
        result['purchase_positive_months_history'] = np.where(count_all > 0, positives_all, np.nan)
        result['purchase_fraction_history'] = np.divide(positives_all, count_all, out=np.full(len(ids), np.nan), where=count_all > 0)
        last = past.loc[past.y_any.eq(1)].groupby('client_id').month.max()
        last_index = ids.map(last)
        now = pd.Period(month, freq='M').ordinal
        result['purchase_months_since_last_positive'] = [
            now - pd.Period(m, freq='M').ordinal if pd.notna(m) else np.nan for m in last_index]
        result['purchase_has_previous_label'] = np.isfinite(recent[:, 0]).astype(int)
        outputs.append(result.loc[:, list(PURCHASE_FEATURES)])
    if not outputs:
        raise ValueError('Empty scoring rows')
    return pd.concat(outputs).reindex(rows.index)


def feature_variants(schema):
    original = schema['variants']['multi_history']
    behavior = [c for c in original if not c.endswith(('_source_window_overlap_days', '_history_truncated'))]
    return {'behavior_original': original, 'behavior': behavior,
            'purchase': list(PURCHASE_FEATURES), 'combined': [*behavior, *PURCHASE_FEATURES]}


def run_ablation(config):
    """Fixed hyperparameters and paired comparisons, isolated from locked old run."""
    if len(config.depths) != 1 or len(config.iterations) != 1:
        raise ValueError('Ablation requires one predeclared configuration')
    manifest, schema, protocol, inventory = source_contract(config)
    validation = [f['validate'] for f in protocol['rolling_validation']]
    needed = sorted(set(m for f in protocol['rolling_validation'] for m in [*f['train'], f['validate']]))
    if set(needed) & {protocol['calibration'], protocol['test']}:
        raise ValueError('Ablation cannot open calibration/test')
    out = config.output_dir
    out.mkdir(parents=True, exist_ok=False)
    variants = feature_variants(schema)
    started = time.monotonic()
    state = dict(status='running', config=config.to_dict(), environment=environment(),
                 started_at=datetime.now(timezone.utc).isoformat(), test_opened=False,
                 calibration_opened=False, protocol=protocol, feature_variants=variants,
                 source_inventory=inventory, source_fingerprint=fingerprint(inventory),
                 code_sha256={p.name: sha256_file(p) for p in Path(__file__).parent.glob('*.py')},
                 selection_note='Exploratory validation comparisons; no test model selected or fitted',
                 parameters=dict(depth=config.depths[0], iterations=config.iterations[0]),
                 class_weights=None, negative_sampling=False, early_stopping=False)
    write_json(out/'manifest.json', state)
    records, predictions = [], []
    try:
        # Only earlier labels are used as history; the function also checks availability per row.
        prior_months = [m for m in needed if m < max(validation)]
        history = read_months(config, prior_months, schema['predictors'][:1], schema, manifest)
        history = history[['client_id', 'month', 'y_any', *[f'target_{i}' for i in range(1,5)]]]
        for fold in protocol['rolling_validation']:
            train = read_months(config, fold['train'], variants['behavior_original'], schema, manifest)
            valid = read_months(config, [fold['validate']], variants['behavior_original'], schema, manifest)
            train = pd.concat([train, purchase_features(train, history, protocol)], axis=1)
            valid = pd.concat([valid, purchase_features(valid, history, protocol)], axis=1)
            for variant, features in variants.items():
                print(f"Ablation {fold['validate']} {variant}: {len(features)} predictors", flush=True)
                model = fit_model(config, train, features, state['parameters'])
                prediction = predict_frame(config, model, valid, features, variant)
                prediction.to_parquet(out/f"validation_{fold['validate']}_{variant}.parquet", index=False)
                predictions.append(prediction)
                records.extend(monthly_metrics(prediction, config.seed))
                write_json(out/'monthly_metrics.json', records)
                model.save_model(str(out/f"model_{fold['validate']}_{variant}.cbm"))
                del model
            # Same-cohort persistence rule, only where prior labels are available.
            if not valid.loc[valid.y_any.notna(), 'purchase_any_lag1'].notna().all():
                raise ValueError('Full validation cohort requires available previous labels for rule comparison')
            rule = valid[['client_id', 'month', 'y_any', 'cold_start', 'trx_ever_seen_source',
                          'geo_ever_seen_source', 'dialog_ever_seen_source']].copy()
            rule['model'] = 'previous_month_purchase'
            rule['score'] = valid.purchase_any_lag1.to_numpy()
            rule['p_any'] = rule.score
            predictions.append(rule)
            for record in monthly_metrics(rule, config.seed):
                # A binary rule is not a calibrated probability model.
                record.pop('log_loss'); record.pop('brier'); record.pop('calibration_bins')
                records.append(record)
            del train, valid
        prediction = pd.concat(predictions, ignore_index=True)
        prediction.to_parquet(out/'validation_predictions.parquet', index=False)
        write_json(out/'monthly_metrics.json', records)
        summaries = []
        for name in [*variants, 'previous_month_purchase']:
            monthly = [r for r in records if r['model'] == name]
            summaries.append(dict(model=name, predictors=len(variants[name]) if name in variants else 1,
                                 mean_recall_at_1pct=float(np.mean([r['top_k']['0.01']['recall'] for r in monthly])),
                                 mean_ap=float(np.mean([r['average_precision'] for r in monthly])),
                                 mean_roc_auc=float(np.mean([r['roc_auc'] for r in monthly])),
                                 mean_log_loss=float(np.mean([r['log_loss'] for r in monthly])) if name in variants else None))
        write_json(out/'summary.json', summaries)
        print('Computing paired client bootstrap comparisons', flush=True)
        comparisons = [paired_recall_bootstrap(prediction.loc[prediction.y_any.notna()], a, b, seed=config.seed,
                                              replicates=config.bootstrap_replicates)
                       for a,b in [('behavior','behavior_original'), ('purchase','previous_month_purchase'),
                                   ('combined','purchase'), ('combined','behavior')]]
        write_json(out/'paired_comparisons.json', comparisons)
        if source_contract(config)[3] != inventory:
            raise ValueError('Feature inputs changed during ablation')
        if {p.name: sha256_file(p) for p in Path(__file__).parent.glob('*.py')} != state['code_sha256']:
            raise ValueError('Baseline code changed during ablation')
        # Publish aggregates only. Client predictions/models remain in ignored outputs/.
        state.update(status='complete', elapsed_seconds=time.monotonic()-started,
                     finished_at=datetime.now(timezone.utc).isoformat(),
                     artifact_sha256={p.name: sha256_file(p) for p in out.iterdir() if p.suffix in ('.cbm','.parquet','.json') and p.name != 'manifest.json'})
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(out/'manifest.json', state)
    return out
