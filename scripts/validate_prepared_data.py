"""Validate existing prepared Parquet; optionally upgrade its schema contract.

No feature regeneration, raw writes or model fitting. Legacy JSON contracts are
backed up before an upgrade; feature Parquet and execution evidence stay intact.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from propensity_data.config import PreparationConfig
from propensity_data.protocol import evaluation_protocol
from propensity_data.schema import SOURCES, feature_schema
from propensity_data.storage import files_at, write_json
from propensity_data.validation import validate_month


def main(argv=None):
    import duckdb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--upgrade-contract', action='store_true')
    args = parser.parse_args(argv)
    out = args.output_dir.resolve()
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    manifest = read(out / 'manifest.json')
    if manifest.get('status') != 'ready':
        raise ValueError('Only a completed ready run can be validated/upgraded')
    config = PreparationConfig(**manifest['config'])
    if config.output_dir != out:
        raise ValueError('Output path differs from manifest configuration')
    schema = feature_schema(config)
    protocol = evaluation_protocol(config, [r['month'] for r in manifest['months']])
    if protocol != read(out / 'protocol.json'):
        raise ValueError('Saved protocol differs from the configured evaluation calendar')
    sources = {'clients': 'client_split', 'targets': 'targets', **SOURCES}
    inventory = [{'source': key, 'path': p.relative_to(config.data_root).as_posix(),
                  'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                 for key, rel in sources.items() for p in files_at(config.data_root / rel)]
    if inventory != read(out / 'source_inventory.json'):
        raise ValueError('Raw inventory differs from the prepared run')
    fingerprint = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
    if fingerprint != manifest['source_inventory_sha256']:
        raise ValueError('Manifest inventory fingerprint differs from the prepared run')
    checks = []
    with duckdb.connect(str(out / 'work.duckdb'), read_only=True) as con:
        con.execute("SET memory_limit='2GB'")
        con.execute('SET threads=2')
        for summary in manifest['months']:
            check = validate_month(con, summary['month'], config, out, schema)
            if (check['scoring_rows'] != summary['rows'] or
                    check['supervised_rows'] != summary['labelled_rows'] or
                    check['unlabelled_rows'] != summary['unknown_labels']):
                raise ValueError('Manifest counts differ from persisted data')
            positive = con.execute('SELECT coalesce(sum(y_any),0) FROM month_data').fetchone()[0]
            if positive != summary['positives']:
                raise ValueError('Manifest positives differ from persisted data')
            checks.append(check)
            print(summary['month'] + ': validated', flush=True)
    result = {'status': 'passed', 'validated_at': datetime.now(timezone.utc).isoformat(),
              'schema_version': schema['schema_version'], 'predictors': len(schema['predictors']),
              'dataset_checks': checks, 'features_regenerated': False,
              'validation_code_sha256': {name: hashlib.sha256(
                  (Path(__file__).resolve().parents[1] / 'src/propensity_data' / name).read_bytes()).hexdigest()
                  for name in ('schema.py', 'validation.py')}}
    if args.upgrade_contract:
        # All checks complete before any contract is changed. Preserve old evidence.
        backup = out / 'contract_before_schema_v2'
        backup.mkdir(exist_ok=False)
        paths = ['manifest.json', 'feature_schema.json']
        if (out / 'model_input/contract.json').is_file():
            paths.append('model_input/contract.json')
        for relative in paths:
            destination = backup / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((out / relative).read_bytes())
        write_json(out / 'feature_schema.json', schema)
        if 'model_input/contract.json' in paths:
            contract = read(out / 'model_input/contract.json')
            model_columns = [*schema['keys'], *schema['predictors'], schema['target']]
            contract.update(schema_version=schema['schema_version'],
                            dtypes={name: schema['dtypes'][name] for name in model_columns},
                            keys=schema['keys'], target=schema['target'], protocol=protocol,
                            predictors=schema['predictors'], ablation_groups=schema['ablation_groups'])
            write_json(out / 'model_input/contract.json', contract)
        manifest.update(feature_schema_version=schema['schema_version'], dataset_checks=checks,
                        contract_validated_at=result['validated_at'])
        write_json(out / 'validation_v2.json', result)
        write_json(out / 'manifest.json', manifest)
    print(json.dumps({'status': result['status'], 'predictors': result['predictors'],
                      'months': len(checks), 'contract_upgraded': args.upgrade_contract}))


if __name__ == '__main__':
    main()
