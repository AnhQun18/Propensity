"""Verify labels, auxiliary fields and the UTC/Moscow cutoff sensitivity.

Reads the public MBD-mini files without changing raw data. Outputs aggregates only.
The Moscow interpretation is a documented EventGPT protocol, not timezone metadata
recovered from the source files.
"""
from pathlib import Path
import argparse
import json

import duckdb
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


def audit(data_root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect()
    db.execute("SET threads=2")
    db.execute("SET memory_limit='2GB'")
    metadata = {}
    for source, folder in {
        'targets': 'targets', 'trx': 'detail/trx',
        'geo': 'detail/geo', 'dialog': 'detail/dialog',
    }.items():
        files = sorted((data_root / folder).rglob('*.parquet'))
        if not files:
            raise FileNotFoundError(data_root / folder)
        db.read_parquet([p.as_posix() for p in files], hive_partitioning=False).create_view(source)
        schema = pq.ParquetFile(files[0]).schema_arrow
        metadata[source] = {
            'files': len(files),
            'representative_file': files[0].relative_to(data_root).as_posix(),
            'event_time_type': str(schema.field('event_time').type) if 'event_time' in schema.names else None,
            'metadata_keys': sorted(k.decode() for k in (schema.metadata or {})),
            'scope': 'Schema metadata from the first shard; not proof of the original timezone.',
        }

    positive = '(' + ' OR '.join(f'target_{i}=1' for i in range(1, 5)) + ')'
    negative = '(' + ' AND '.join(f'target_{i}=0' for i in range(1, 5)) + ')'
    invalid = '(' + ' OR '.join(f'(target_{i} IS NOT NULL AND target_{i} NOT IN (0,1))' for i in range(1, 5)) + ')'
    nulls = ' + '.join(f'(CASE WHEN target_{i} IS NULL THEN 1 ELSE 0 END)' for i in range(1, 5))
    labels = db.execute(f"""
        SELECT count(*) rows, count(DISTINCT client_id) clients,
               count(DISTINCT mon) AS "months",
               count(*) - count(DISTINCT (client_id, mon)) duplicate_keys,
               count(*) FILTER (WHERE {positive}) positive_rows,
               count(*) FILTER (WHERE {negative}) negative_rows,
               count(*) FILTER (WHERE NOT coalesce({positive},false)
                                 AND NOT coalesce({negative},false)) unknown_rows,
               count(*) FILTER (WHERE {invalid}) invalid_label_rows,
               sum({nulls}) null_label_cells
        FROM targets
    """).df().iloc[0].to_dict()
    labels = {k: int(v) for k, v in labels.items()}
    assert labels['rows'] == labels['clients'] * labels['months'] == 1202688
    assert labels['duplicate_keys'] == labels['unknown_rows'] == labels['invalid_label_rows'] == labels['null_label_cells'] == 0
    assert labels['positive_rows'] == 12854
    labels['positive_rate'] = labels['positive_rows'] / labels['rows']
    auxiliary = db.execute("""
        SELECT count(*) rows, count(*) FILTER (WHERE diff_trans_date IS NULL) diff_null,
               min(diff_trans_date) diff_min, max(diff_trans_date) diff_max,
               count(*) FILTER (WHERE trans_count=0 AND diff_trans_date IS NULL) zero_count_null_diff,
               count(*) FILTER (WHERE trans_count>0 AND diff_trans_date IS NULL) positive_count_null_diff,
               count(*) FILTER (WHERE trans_count=0 AND diff_trans_date IS NOT NULL) zero_count_present_diff,
               count(*) FILTER (WHERE trans_count=1 AND diff_trans_date>0) one_count_positive_diff
        FROM targets
    """).df().iloc[0].to_dict()
    auxiliary = {k: int(v) for k, v in auxiliary.items()}
    assert auxiliary['diff_null'] == auxiliary['zero_count_null_diff'] == 120786
    assert auxiliary['positive_count_null_diff'] == auxiliary['zero_count_present_diff'] == 0

    db.execute('CREATE TEMP TABLE boundary_events(source VARCHAR, client_id VARCHAR, label_month TIMESTAMP)')
    for source in ['trx', 'geo', 'dialog']:
        print(f'Scanning month-boundary timestamps: {source}', flush=True)
        db.execute(f"""
            INSERT INTO boundary_events
            SELECT '{source}', client_id, date_trunc('month', event_time + INTERVAL '3 hours')
            FROM {source}
            WHERE event_time >= TIMESTAMP '2022-01-31 21:00:00'
              AND event_time < TIMESTAMP '2023-01-01 00:00:00'
              AND event_time >= date_trunc('month', event_time + INTERVAL '3 hours') - INTERVAL '3 hours'
              AND event_time < date_trunc('month', event_time + INTERVAL '3 hours')
        """)
    monthly = db.execute("""
        SELECT strftime(label_month, '%Y-%m') AS "month", source, count(*) events,
               count(DISTINCT client_id) clients
        FROM boundary_events GROUP BY ALL
    """).df()
    grid = pd.MultiIndex.from_product(
        [pd.period_range('2022-02', '2023-01', freq='M').astype(str), ['trx', 'geo', 'dialog']],
        names=['month', 'source'],
    )
    monthly = monthly.set_index(['month', 'source']).reindex(grid, fill_value=0).reset_index()
    monthly.to_csv(output / 'timezone_boundary_by_source.csv', index=False)
    combined = db.execute("""
        SELECT strftime(label_month, '%Y-%m') AS "month",
               count(DISTINCT client_id) clients
        FROM boundary_events GROUP BY ALL ORDER BY "month"
    """).df().set_index('month').reindex(grid.levels[0], fill_value=0).reset_index()
    combined.to_csv(output / 'timezone_boundary_clients.csv', index=False)
    boundary = {
        'events': int(monthly.events.sum()),
        'client_months': int(combined.clients.sum()),
        'clients': int(db.execute('SELECT count(DISTINCT client_id) FROM boundary_events').fetchone()[0]),
        'test_2023_01_events': int(monthly.loc[monthly.month.eq('2023-01'), 'events'].sum()),
        'meaning': 'Events included by naive UTC midnight but excluded by Moscow midnight; this measures a protocol difference, not proof of original label timezone.',
    }
    result = {
        'checked_on': '2026-11-09', 'labels': labels, 'auxiliary': auxiliary,
        'timezone_metadata': metadata, 'boundary_sensitivity': boundary,
        'raw_edited': False, 'model_fitted': False,
    }
    (output / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    db.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/MBD-mini')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'docs/propensity_feasibility/open_questions')
    args = parser.parse_args()
    audit(args.data_root, args.output_dir)
