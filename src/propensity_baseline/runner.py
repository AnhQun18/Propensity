"""Explicit tune -> fit/calibrate -> test stages, with locked feature contracts."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from propensity_data.storage import literal, write_json
from propensity_data.schema import validate_feature_columns
from .data import calendar_schema, file_inventory, fingerprint, read_json
from .metrics import evaluate, paired_recall_bootstrap


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def environment():
    return {name: importlib.metadata.version(name)
            for name in ('catboost', 'scikit-learn', 'pandas', 'numpy', 'duckdb', 'pyarrow')}


def source_contract(config):
    source = config.source_dir
    manifest = read_json(source / 'manifest.json')
    if manifest.get('status') != 'ready' or manifest.get('history_mode') != 'calendar_split':
        raise ValueError('Baseline requires ready calendar_split features')
    schema, protocol = (read_json(source / name) for name in ('feature_schema.json', 'protocol.json'))
    if schema.get('schema_version') != 3 or not set(config.variants) <= set(schema.get('variants', {})):
        raise ValueError('Unsupported calendar schema/variants')
    if protocol['primary_budget_fraction'] != .01:
        raise ValueError('This experiment predeclares top 1% as primary budget')
    prepared = read_json(Path(manifest['source_dir']) / 'manifest.json')
    from propensity_data.config import PreparationConfig
    preparation_config = PreparationConfig(**prepared['config'])
    if schema != calendar_schema(preparation_config):
        raise ValueError('Calendar predictor/schema contract was changed')
    for protected in (preparation_config.data_root, Path(manifest['source_dir']).resolve()):
        if config.output_dir == protected or protected in config.output_dir.parents or config.output_dir in protected.parents:
            raise ValueError('Model output must not overlap raw/prepared data')
    names = ('manifest.json', 'feature_schema.json', 'protocol.json')
    paths = [*(source / n for n in names),
             *(source / f"features/month={m['month']}/part.parquet" for m in manifest['months'])]
    return manifest, schema, protocol, file_inventory(paths)


def read_months(config, months, features, schema, manifest):
    """Read selected numeric predictors, labels and cohort fields; never all X."""
    import duckdb
    if not months:
        raise ValueError('Empty month selection')
    summaries = {r['month']: r for r in manifest['months']}
    allowed = set(schema['predictors'])
    if not features or not set(features) <= allowed:
        raise ValueError('Unapproved predictor list')
    # Source labels stay outside X and verify the persisted any-product target.
    meta = ['client_id', 'cutoff', 'y_any', 'target_1', 'target_2', 'target_3', 'target_4',
            'supervised_eligible', 'cold_start', 'trx_ever_seen_source',
            'geo_ever_seen_source', 'dialog_ever_seen_source']
    columns = list(dict.fromkeys([*meta, *features]))
    parts = []
    with duckdb.connect() as con:
        con.execute("SET memory_limit='2GB'")
        con.execute(f'SET threads={config.threads}')
        for month in months:
            if month not in summaries:
                raise ValueError('Month outside source contract')
            file = config.source_dir / f'features/month={month}/part.parquet'
            scan = f'read_parquet({literal(file.as_posix())}, hive_partitioning=false)'
            validate_feature_columns(con.execute('DESCRIBE SELECT * FROM ' + scan).fetchall(), schema)
            part = con.execute('SELECT ' + ','.join('"'+c+'"' for c in columns) + ' FROM ' + scan).df()
            if len(part) != summaries[month]['rows'] or part.client_id.isna().any() or part.client_id.duplicated().any():
                raise ValueError('Month population/keys differ from manifest')
            if not part.cutoff.eq(pd.Timestamp(month+'-01')).all():
                raise ValueError('Invalid prediction cutoff')
            labels = part[[f'target_{i}' for i in range(1,5)]]
            if (~(labels.isna() | labels.isin([0,1]))).any().any():
                raise ValueError('Invalid source label')
            expected = np.where(labels.eq(1).any(axis=1), 1., np.where(labels.eq(0).all(axis=1), 0., np.nan))
            if not np.array_equal(expected, part.y_any.to_numpy(dtype=float, na_value=np.nan), equal_nan=True):
                raise ValueError('Any-product labels do not reconcile')
            if not part.supervised_eligible.eq(part.y_any.notna()).all():
                raise ValueError('Label eligibility mismatch')
            if int(part.y_any.notna().sum()) != summaries[month]['labelled_rows'] or int(part.y_any.sum()) != summaries[month]['positives']:
                raise ValueError('Persisted label counts differ from manifest')
            part['month'] = month
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def numeric_x(frame, features):
    # Training matrix is bounded to selected features, and is not stored twice.
    with np.errstate(over='ignore', invalid='ignore'):
        values = frame[features].to_numpy(dtype=np.float32, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError('Feature overflow/non-finite values in float32 model input')
    return pd.DataFrame(values, columns=features)


def fit_model(config, frame, features, parameters):
    from catboost import CatBoostClassifier
    known = frame.loc[frame.y_any.notna()]
    if known.y_any.nunique() != 2:
        raise ValueError('Training requires both classes')
    model = CatBoostClassifier(loss_function='Logloss', learning_rate=config.learning_rate,
                              l2_leaf_reg=config.l2_leaf_reg, random_seed=config.seed,
                              thread_count=config.threads, task_type='CPU', nan_mode='Min',
                              allow_writing_files=False, verbose=False, **parameters)
    model.fit(numeric_x(known, features), known.y_any.astype(int))
    return model


def predict_frame(config, model, frame, features, model_name):
    from scipy.special import expit
    result = frame[['client_id', 'month', 'y_any', 'cold_start', 'trx_ever_seen_source',
                    'geo_ever_seen_source', 'dialog_ever_seen_source']].copy()
    result['model'] = model_name
    result['score'] = model.predict(numeric_x(frame, features), prediction_type='RawFormulaVal')
    result['p_any'] = expit(result.score.to_numpy())
    return result


def monthly_metrics(frame, seed):
    records = []
    for (model, month), full in frame.groupby(['model', 'month'], sort=True):
        group = full.loc[full.y_any.notna()]
        if not len(group):
            raise ValueError('No labels for monthly evaluation')
        metric = evaluate(group.y_any.to_numpy(dtype=int), group.p_any.to_numpy(), group.client_id.to_numpy(),
                          seed=seed, ranking_score=group.score.to_numpy())
        records.append(dict(model=model, month=month, scoring_rows=len(full), labelled_rows=len(group), **metric))
    return records


def constant_predictions(frame, probability, name='base_rate'):
    result = frame[['client_id', 'month', 'y_any', 'cold_start', 'trx_ever_seen_source',
                    'geo_ever_seen_source', 'dialog_ever_seen_source']].copy()
    result['model'], result['p_any'], result['score'] = name, probability, probability
    return result


def candidate_summary(variant, params, records, n_features):
    recalls = [m['top_k']['0.01']['recall'] for m in records]
    if any(x is None for x in recalls):
        raise ValueError('Each validation month must contain positive labels')
    return dict(variant=variant, parameters=params, predictors=n_features,
                mean_recall_at_1pct=float(np.mean(recalls)),
                mean_ap=float(np.mean([m['average_precision'] for m in records])),
                mean_log_loss=float(np.mean([m['log_loss'] for m in records])),
                recalls=recalls)


def selection_key(record):
    # Minimize tuple: primary Recall@1%, then AP/logloss, then simpler models.
    return (-record['mean_recall_at_1pct'], -record['mean_ap'], record['mean_log_loss'],
            record['predictors'], record['parameters']['depth'], record['parameters']['iterations'], record['variant'])


def tune(config):
    """Read train/rolling-validation only. Write a frozen selection, never test."""
    manifest, schema, protocol, inputs = source_contract(config)
    versions = environment()  # Check dependencies before creating output.
    out = config.output_dir
    out.mkdir(parents=True, exist_ok=False)
    state = dict(status='tuning', config=config.to_dict(), environment=versions,
                 source_inventory=inputs, source_fingerprint=fingerprint(inputs),
                 started_at=datetime.now(timezone.utc).isoformat(), test_opened=False,
                 selection_rule='macro Recall@1%, AP, logloss, fewer predictors/depth/trees',
                 calibration_rule='L2 sigmoid on raw margin, C=1; require positive slope; December fit only',
                 class_weights=None, negative_sampling=False, early_stopping=False,
                 code_sha256={p.name: sha256_file(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(out / 'manifest.json', state)
    records, all_predictions, baseline_predictions = [], [], []
    try:
        for variant in config.variants:
            features = schema['variants'][variant]
            best = None
            best_predictions = None
            for depth, iterations in product(config.depths, config.iterations):
                params = dict(depth=depth, iterations=iterations)
                fold_metrics, predictions = [], []
                for fold in protocol['rolling_validation']:
                    print(f"{variant}: depth={depth}, trees={iterations}, validate={fold['validate']}", flush=True)
                    train = read_months(config, fold['train'], features, schema, manifest)
                    rate = float(train.y_any.mean())
                    model = fit_model(config, train, features, params)
                    del train
                    validate = read_months(config, [fold['validate']], features, schema, manifest)
                    prediction = predict_frame(config, model, validate, features, variant)
                    predictions.append(prediction)
                    fold_metrics.extend(monthly_metrics(prediction, config.seed))
                    if variant == config.variants[0] and depth == config.depths[0] and iterations == config.iterations[0]:
                        baseline_predictions.append(constant_predictions(validate, rate))
                    del model, validate
                summary = candidate_summary(variant, params, fold_metrics, len(features))
                records.append(dict(**summary, monthly_metrics=fold_metrics))
                if best is None or selection_key(summary) < selection_key(best):
                    best, best_predictions = summary, pd.concat(predictions, ignore_index=True)
                write_json(out / 'validation_candidates.json', records)
            all_predictions.append(best_predictions)
        summaries = [min((r for r in records if r['variant'] == v), key=selection_key) for v in config.variants]
        winner = min(summaries, key=selection_key)
        predictions = pd.concat([*all_predictions, *baseline_predictions], ignore_index=True)
        predictions.to_parquet(out / 'validation_predictions.parquet', index=False)
        write_json(out / 'validation_metrics.json', monthly_metrics(predictions, config.seed))
        reference = 'trx_recent' if 'trx_recent' in config.variants else config.variants[0]
        known = predictions.loc[predictions.y_any.notna()]
        comparison = paired_recall_bootstrap(known, winner['variant'], reference, seed=config.seed,
                                             replicates=config.bootstrap_replicates)
        comparison['interpretation'] = 'Exploratory: the same validation months selected the models; confirm on locked test'
        write_json(out / 'paired_comparison.json', comparison)
        comparisons = []
        for variant, reference_name in [('trx_history','trx_recent'), ('multi_recent','trx_recent'),
                                        ('multi_history','trx_history'), ('multi_history','multi_recent')]:
            if variant in config.variants and reference_name in config.variants:
                comparisons.append(paired_recall_bootstrap(known, variant, reference_name, seed=config.seed,
                                                           replicates=config.bootstrap_replicates))
        write_json(out / 'ablation_comparisons.json', comparisons)
        selection = {k: v for k, v in winner.items() if k != 'monthly_metrics'}
        selection.update(features=schema['variants'][winner['variant']], protocol=protocol,
                         source_fingerprint=fingerprint(inputs))
        write_json(out / 'selection.json', selection)
        if source_contract(config)[3] != inputs:
            raise ValueError('Feature inputs changed during tuning')
        state.update(status='tuned', selection_sha256=sha256_file(out / 'selection.json'),
                     variant_summaries=[{k:v for k,v in r.items() if k != 'monthly_metrics'} for r in summaries],
                     finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        state.update(status='failed_tuning', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(out / 'manifest.json', state)
    return out / 'selection.json'


def locked_run(config, expected_status):
    state = read_json(config.output_dir / 'manifest.json')
    # JSON lists and dataclass tuples deliberately compare through a hash.
    if fingerprint(state['config']) != fingerprint(config.to_dict()):
        raise ValueError('Configuration changed after tuning')
    if state.get('status') != expected_status:
        raise ValueError(f'Expected run status {expected_status}')
    manifest, schema, protocol, inputs = source_contract(config)
    if fingerprint(inputs) != state['source_fingerprint']:
        raise ValueError('Feature inputs changed after tuning')
    current_code = {p.name: sha256_file(p) for p in Path(__file__).parent.glob('*.py')}
    if current_code != state['code_sha256'] or environment() != state['environment']:
        raise ValueError('Baseline code or library versions changed; start a new run')
    if sha256_file(config.output_dir / 'selection.json') != state['selection_sha256']:
        raise ValueError('Selection changed after tuning')
    selection = read_json(config.output_dir / 'selection.json')
    if selection['features'] != schema['variants'][selection['variant']] or selection['protocol'] != protocol:
        raise ValueError('Selection no longer matches input contract')
    return state, manifest, schema, protocol, selection


def fit_final(config):
    """Fit through November, calibrate December; never read the test partition."""
    from scipy.special import expit
    from sklearn.linear_model import LogisticRegression
    state, manifest, schema, protocol, selected = locked_run(config, 'tuned')
    out = config.output_dir
    model_dir = out / 'final'
    model_dir.mkdir(exist_ok=False)
    state['status'] = 'fitting'
    write_json(out / 'manifest.json', state)
    try:
        features = selected['features']
        train = read_months(config, protocol['final_train'], features, schema, manifest)
        state['train_base_rate'] = float(train.y_any.mean())
        model = fit_model(config, train, features, selected['parameters'])
        del train
        model.save_model(str(model_dir / 'model.cbm'))
        calibration = read_months(config, [protocol['calibration']], features, schema, manifest)
        prediction = predict_frame(config, model, calibration, features, selected['variant'])
        known = prediction.loc[prediction.y_any.notna()]
        if known.y_any.nunique() != 2:
            raise ValueError('Calibration requires both classes')
        calibrator = LogisticRegression(C=1., solver='lbfgs', max_iter=1000, random_state=config.seed)
        calibrator.fit(known[['score']], known.y_any.astype(int))
        if calibrator.n_iter_[0] >= calibrator.max_iter:
            raise ValueError('Sigmoid calibration did not converge')
        slope, intercept = float(calibrator.coef_[0,0]), float(calibrator.intercept_[0])
        if not np.isfinite([slope, intercept]).all() or slope <= 0:
            raise ValueError('Sigmoid calibration is not increasing; inspect validation/calibration before continuing')
        write_json(model_dir / 'calibrator.json', dict(method='sigmoid', slope=slope, intercept=intercept, C=1.))
        prediction['p_any_raw'] = prediction.p_any
        prediction['p_any'] = expit(slope*prediction.score.to_numpy()+intercept)
        prediction.to_parquet(model_dir / 'calibration_predictions.parquet', index=False)
        write_json(model_dir / 'calibration_fit_diagnostics.json', dict(
            warning='In-sample calibrator fit diagnostics; not an unbiased performance estimate',
            monthly_metrics=monthly_metrics(prediction, config.seed)))
        importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
        importance.sort_values('importance', ascending=False).to_csv(model_dir / 'feature_importance.csv', index=False)
        if source_contract(config)[3] != state['source_inventory']:
            raise ValueError('Feature inputs changed during final fit')
        state.update(status='fitted', model_sha256=sha256_file(model_dir / 'model.cbm'),
                     calibrator_sha256=sha256_file(model_dir / 'calibrator.json'),
                     fitted_at=datetime.now(timezone.utc).isoformat(), test_opened=False)
    except BaseException as exc:
        state.update(status='failed_fit', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(out / 'manifest.json', state)
    return model_dir


def test_final(config):
    """Explicit final holdout opening. Output refuses overwrite and further tuning."""
    from catboost import CatBoostClassifier
    from scipy.special import expit
    state, manifest, schema, protocol, selected = locked_run(config, 'fitted')
    out = config.output_dir
    model_dir = out / 'final'
    for filename, key in [('model.cbm','model_sha256'), ('calibrator.json','calibrator_sha256')]:
        if sha256_file(model_dir / filename) != state[key]:
            raise ValueError('Final model/calibrator changed after fit')
    test_dir = out / 'test'
    test_dir.mkdir(exist_ok=False)
    state.update(status='testing', test_opened=True)
    write_json(out / 'manifest.json', state)
    try:
        frame = read_months(config, [protocol['test']], selected['features'], schema, manifest)
        model = CatBoostClassifier()
        model.load_model(str(model_dir / 'model.cbm'))
        if model.feature_names_ != selected['features']:
            raise ValueError('Final model feature names/order differ from frozen selection')
        prediction = predict_frame(config, model, frame, selected['features'], selected['variant'])
        calibration = read_json(model_dir / 'calibrator.json')
        prediction['p_any_raw'] = prediction.p_any
        prediction['p_any'] = expit(calibration['slope']*prediction.score.to_numpy()+calibration['intercept'])
        baseline = constant_predictions(frame, state['train_base_rate'])
        combined = pd.concat([prediction, baseline], ignore_index=True)
        combined.to_parquet(test_dir / 'predictions.parquet', index=False)
        write_json(test_dir / 'metrics.json', monthly_metrics(combined, config.seed))
        raw = prediction.copy()
        raw['p_any'] = raw.p_any_raw
        write_json(test_dir / 'uncalibrated_metrics.json', monthly_metrics(raw, config.seed))
        cohorts = []
        for name, mask in [('cold_start', prediction.cold_start.eq(1)),
                           ('has_history', prediction.cold_start.eq(0)),
                           ('no_trx', prediction.trx_ever_seen_source.eq(0)),
                           ('no_geo', prediction.geo_ever_seen_source.eq(0)),
                           ('no_dialog', prediction.dialog_ever_seen_source.eq(0))]:
            if prediction.loc[mask].y_any.notna().any():
                cohorts.append(dict(cohort=name, definition='Top-K recomputed within this cohort; groups may overlap',
                                    metrics=monthly_metrics(prediction.loc[mask], config.seed)))
        write_json(test_dir / 'cohort_metrics.json', cohorts)
        state.update(status='tested', tested_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        state.update(status='failed_test', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(out / 'manifest.json', state)
    return test_dir
