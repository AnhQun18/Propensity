"""Re-aggregate a read-only normalized cache into disjoint calendar histories."""
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path

from propensity_data.config import PreparationConfig, month_start
from propensity_data.features import window_features
from propensity_data.protocol import evaluation_protocol
from propensity_data.schema import LABELS, SOURCES, feature_schema, validate_feature_columns
from propensity_data.storage import assert_empty, export, literal, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_inventory(paths):
    return [{'path': str(p.resolve()), 'bytes': p.stat().st_size,
             'mtime_ns': p.stat().st_mtime_ns} for p in paths]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def previous_month(month):
    day = month_start(month)
    return date(day.year - (day.month == 1), (day.month - 2) % 12 + 1, 1)


def calendar_schema(config):
    schema = feature_schema(config, window_tags=('recent', 'older', 'all'))
    for source in SOURCES:
        for name in ('recent_event_rate', 'older_event_rate', 'event_rate_difference'):
            column = source + '_' + name
            schema['columns'].append(column)
            schema['dtypes'][column] = 'DOUBLE'
            schema['predictors'].append(column)
            schema['groups'][source].append(column)
    recent = lambda s: [c for c in schema['predictors'] if c.startswith(s + '_recent_')]
    history = lambda s: [c for c in schema['predictors']
                         if c.startswith((s + '_recent_', s + '_older_'))
                         or c == s + '_event_rate_difference']
    schema['variants'] = {
        'trx_recent': recent('trx'), 'trx_history': history('trx'),
        'multi_recent': sum((recent(s) for s in SOURCES), []),
        'multi_history': sum((history(s) for s in SOURCES), []),
    }
    schema['history_definition'] = {
        'recent': '[start of previous calendar month, cutoff)',
        'older': 'all recorded events before start of previous calendar month',
        'all': 'audit/cohort only; excluded from model variants',
        'recency_anchor': 'prediction cutoff, including older-window recency',
        'rates': 'events / days overlapping the source extract; not continuous client observation',
    }
    return schema


def validate_calendar(con, month, out, schema):
    """Check labels, count conservation, currency, time bounds and finite values."""
    file = out / f'features/month={month}/part.parquet'
    con.execute(f"CREATE OR REPLACE TEMP VIEW checked AS SELECT * FROM read_parquet({literal(file.as_posix())}, hive_partitioning=false)")
    validate_feature_columns(con.execute('DESCRIBE checked').fetchall(), schema)
    cutoff = month_start(month)
    recent_start = previous_month(month)
    days = (cutoff - recent_start).days
    assert_empty(con, 'SELECT client_id FROM checked GROUP BY client_id HAVING client_id IS NULL OR count(*)<>1',
                 'Invalid calendar customer keys')
    expected_rows = con.execute('SELECT count(*) FROM clients').fetchone()[0]
    rows, known, positives = con.execute('SELECT count(*),count(y_any),coalesce(sum(y_any),0) FROM checked').fetchone()
    if rows != expected_rows:
        raise ValueError('Calendar features lost customer rows')
    assert_empty(con, 'SELECT 1 FROM checked ANTI JOIN clients USING(client_id)', 'Unexpected calendar customer')
    conditions = [f"cutoff IS DISTINCT FROM TIMESTAMP {literal(cutoff)}",
                  'y_any NOT IN (0,1)', 'supervised_eligible IS DISTINCT FROM (y_any IS NOT NULL)',
                  'cold_start IS DISTINCT FROM cast(trx_all_has_event=0 AND geo_all_has_event=0 AND dialog_all_has_event=0 AS TINYINT)']
    conditions += [f'("{c}" IS NOT NULL AND NOT isfinite("{c}"::DOUBLE))' for c in schema['predictors']]
    for s in SOURCES:
        for tag in ('recent', 'older', 'all'):
            p = s + '_' + tag + '_'
            conditions += [f'{p}event_count<0', f'{p}active_days<0', f'{p}active_days>{p}event_count',
                           f'{p}days_since_last_event<0',
                           f'{p}days_since_first_recorded_event<{p}days_since_last_event',
                           f'{p}has_event IS DISTINCT FROM cast(coalesce({p}event_count,0)>0 AS TINYINT)',
                           f'({p}source_window_overlap_days=0 AND {p}event_count IS NOT NULL)',
                           f'({p}source_window_overlap_days>0 AND {p}event_count IS NULL)']
            if s == 'trx':
                conditions += [f'{p}main_count+{p}foreign_count+{p}currency_missing_count+{p}currency_invalid_count<>{p}event_count',
                               f'{p}known_currency_count<>{p}main_count+{p}foreign_count',
                               f'{p}main_amount_count>{p}main_count']
        conditions += [f'coalesce({s}_recent_event_count,0)+coalesce({s}_older_event_count,0)<>coalesce({s}_all_event_count,0)',
                       f'coalesce({s}_recent_active_days,0)+coalesce({s}_older_active_days,0)<>coalesce({s}_all_active_days,0)',
                       f'{s}_recent_days_since_first_recorded_event>{days}',
                       f'{s}_older_days_since_last_event<{days}',
                       f'{s}_recent_source_window_overlap_days>{days}']
    conditions += [f'("{c}"<0 OR "{c}">1)' for c in schema['predictors'] if 'fraction' in c or 'share' in c]
    assert_empty(con, 'SELECT 1 FROM checked WHERE ' + ' OR '.join(conditions), 'Calendar feature invariants failed')
    differences = ' OR '.join(f'd.{c} IS DISTINCT FROM t.{c}' for c in (*LABELS, 'y_any'))
    assert_empty(con, 'SELECT 1 FROM checked d LEFT JOIN selected_targets t ON d.client_id=t.client_id AND d.cutoff=t.cutoff '
                 f'WHERE {differences}', 'Calendar labels differ from target references')
    expected_y = ('CASE WHEN target_1=1 OR target_2=1 OR target_3=1 OR target_4=1 THEN 1 '
                  'WHEN target_1=0 AND target_2=0 AND target_3=0 AND target_4=0 THEN 0 ELSE NULL END')
    assert_empty(con, f'SELECT 1 FROM checked WHERE y_any IS DISTINCT FROM ({expected_y})', 'Invalid any-product label')
    return dict(month=month, rows=rows, labelled_rows=known, positives=positives, unknown_labels=rows-known)


def build_calendar(source_dir, output_dir, *, reuse_prepared_aggregates=True):
    """Use normalized tables from a ready run; never write raw or prior outputs."""
    import duckdb
    if type(reuse_prepared_aggregates) is not bool:
        raise ValueError('reuse_prepared_aggregates must be boolean')
    source, out = Path(source_dir).resolve(), Path(output_dir).resolve()
    old = read_json(source / 'manifest.json')
    if old.get('status') != 'ready':
        raise ValueError('Source preparation must be ready')
    config = PreparationConfig(**old['config'])
    if source != config.output_dir:
        raise ValueError('Source path differs from preparation manifest')
    if config.include_dialog_embeddings:
        raise ValueError('Calendar baseline v1 uses dialog metadata; use a metadata-only prepared source')
    for protected in (source, config.data_root):
        if out == protected or out in protected.parents or protected in out.parents:
            raise ValueError('Calendar output must not overlap input trees')
    if read_json(source / 'feature_schema.json') != feature_schema(config):
        raise ValueError('Prepared source schema differs from approved schema')
    months = [r['month'] for r in old['months']]
    protocol = evaluation_protocol(config, months)
    if read_json(source / 'protocol.json') != protocol:
        raise ValueError('Prepared source protocol mismatch')
    paths = [source / n for n in ('manifest.json', 'feature_schema.json', 'protocol.json', 'work.duckdb')]
    cached_months = {m: source / f'features/month={m}/part.parquet' for m in months}
    # Reuse exact-cutoff all-history aggregates only from the original completed
    # run. If normalized/cache files were changed later, recompute from the DB.
    finished = datetime.fromisoformat(old['finished_at']).timestamp()
    reuse_cache = reuse_prepared_aggregates and all(p.is_file() and p.stat().st_mtime <= finished + 2
                      for p in [source / 'work.duckdb', *cached_months.values()])
    if reuse_cache:
        paths += list(cached_months.values())
    before = file_inventory(paths)
    out.mkdir(parents=True, exist_ok=False)
    schema = calendar_schema(config)
    manifest = dict(status='running', history_mode='calendar_split', source_dir=str(source),
                    source_inventory=before, source_fingerprint=fingerprint(before),
                    fingerprint_note='Paths, sizes and mtimes; not full byte checksums',
                    started_at=datetime.now(timezone.utc).isoformat(), months=[], models_fitted=False,
                    reused_windows=[], cache_reuse_enabled=reuse_cache,
                    sidecar_note='Reused all/older aggregates reference source sidecars; newly computed windows have local sidecars')
    write_json(out / 'manifest.json', manifest)
    con = None
    try:
        con = duckdb.connect()
        con.execute("SET threads=2")
        con.execute("SET memory_limit='2GB'")
        con.execute(f"SET temp_directory={literal((out / 'spill').as_posix())}")
        con.execute(f"ATTACH {literal((source / 'work.duckdb').as_posix())} AS prepared (READ_ONLY)")
        for table in ('clients', 'selected_targets', 'source_bounds', *SOURCES):
            con.execute(f'CREATE TEMP VIEW {table} AS SELECT * FROM prepared.{table}')
        for month in months:
            cutoff = month_start(month).isoformat()
            previous = previous_month(month).isoformat()
            con.execute(f"CREATE OR REPLACE TEMP TABLE features AS SELECT c.client_id, TIMESTAMP {literal(cutoff)} cutoff, "
                        f"t.y_any, {', '.join('t.'+c for c in LABELS)}, (t.y_any IS NOT NULL) supervised_eligible "
                        f"FROM clients c LEFT JOIN selected_targets t ON c.client_id=t.client_id AND t.cutoff=TIMESTAMP {literal(cutoff)}")
            for s in SOURCES:
                for tag, lower, upper in [('recent', previous, cutoff), ('older', None, previous), ('all', None, cutoff)]:
                    cached_month = month if tag == 'all' else previous[:7] if tag == 'older' else None
                    if reuse_cache and cached_month in cached_months:
                        cached_file = cached_months[cached_month]
                        scan = f'read_parquet({literal(cached_file.as_posix())}, hive_partitioning=false)'
                        validate_feature_columns(con.execute('DESCRIBE SELECT * FROM ' + scan).fetchall(), feature_schema(config))
                        prefix = s + '_all_'
                        expressions = ['client_id']
                        for name in feature_schema(config)['columns']:
                            if not name.startswith(prefix):
                                continue
                            suffix = name[len(prefix):]
                            expression = name
                            if tag == 'older' and suffix in ('days_since_first_recorded_event','days_since_last_event'):
                                lag_days = (month_start(month)-previous_month(month)).days
                                expression += f' + {lag_days}'
                            expressions.append(f'{expression} AS {s}_{tag}_{suffix}')
                        con.execute('CREATE OR REPLACE TEMP TABLE window_result AS SELECT ' + ','.join(expressions) + ' FROM ' + scan)
                        manifest['reused_windows'].append(dict(month=month,source=s,window=tag,cache_month=cached_month))
                    else:
                        window_features(con, s, cutoff, None, config, out, tag=tag, lower_bound=lower, upper_bound=upper)
                    con.execute('CREATE OR REPLACE TEMP TABLE joined AS SELECT f.*, w.* EXCLUDE(client_id) '
                                'FROM features f LEFT JOIN window_result w USING(client_id)')
                    con.execute('DROP TABLE features')
                    con.execute('ALTER TABLE joined RENAME TO features')
                con.execute(f'ALTER TABLE features ADD COLUMN {s}_ever_seen_source TINYINT')
                con.execute(f'UPDATE features SET {s}_ever_seen_source={s}_all_has_event')
                if s == 'dialog':
                    con.execute('ALTER TABLE features ADD COLUMN dialog_has_dialog TINYINT')
                    con.execute('UPDATE features SET dialog_has_dialog=dialog_all_has_event')
            con.execute('ALTER TABLE features ADD COLUMN cold_start TINYINT')
            con.execute('UPDATE features SET cold_start=cast(trx_all_has_event=0 AND geo_all_has_event=0 AND dialog_all_has_event=0 AS TINYINT)')
            for s in SOURCES:
                for tag in ('recent', 'older'):
                    con.execute(f'ALTER TABLE features ADD COLUMN {s}_{tag}_event_rate DOUBLE')
                    con.execute(f'UPDATE features SET {s}_{tag}_event_rate={s}_{tag}_event_count::DOUBLE/nullif({s}_{tag}_source_window_overlap_days,0)')
                con.execute(f'ALTER TABLE features ADD COLUMN {s}_event_rate_difference DOUBLE')
                con.execute(f'UPDATE features SET {s}_event_rate_difference={s}_recent_event_rate-{s}_older_event_rate')
            # Construct schema order explicitly, including rates after cohort flags.
            columns = ','.join('"'+c+'"' for c in schema['columns'])
            export(con, f'SELECT {columns} FROM features ORDER BY client_id', out / f'features/month={month}/part.parquet')
            manifest['months'].append(validate_calendar(con, month, out, schema))
            write_json(out / 'manifest.json', manifest)
            print(f'Calendar features {month}: validated', flush=True)
        if file_inventory(paths) != before:
            raise ValueError('Prepared cache changed during calendar aggregation')
        write_json(out / 'feature_schema.json', schema)
        write_json(out / 'protocol.json', protocol)
        manifest.update(status='ready', finished_at=datetime.now(timezone.utc).isoformat(), predictors=len(schema['predictors']))
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if con is not None:
            con.close()
        write_json(out / 'manifest.json', manifest)
    return out / 'manifest.json'
