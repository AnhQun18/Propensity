"""Recompute raw TRX null cells, affected rows and joint patterns without filtering."""
from pathlib import Path
import json
import duckdb

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trx_null_audit_20261117'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / 'data/MBD-mini/detail/trx').rglob('*.parquet'))
    before = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET memory_limit='2GB'")
    con.read_parquet([p.as_posix() for p in files], hive_partitioning=True).create_view('trx')
    cols = [r[0] for r in con.execute('DESCRIBE trx').fetchall()]
    q = lambda c: '"' + c.replace('"', '""') + '"'
    null_count = ' + '.join(f'CAST({q(c)} IS NULL AS INTEGER)' for c in cols)
    expr = ', '.join(f'count(*) FILTER (WHERE {q(c)} IS NULL)' for c in cols)
    result = con.execute(f'SELECT count(*), {expr}, count(*) FILTER (WHERE ({null_count})>0) FROM trx').fetchone()
    n, counts, affected = result[0], result[1:-1], result[-1]
    import pandas as pd
    per_col = pd.DataFrame([dict(column=c, rows=n, null_n=v, null_pct=100*v/n) for c, v in zip(cols, counts)])
    per_col.to_csv(OUT / 'columns.csv', index=False)
    patterns = con.execute(f'SELECT {null_count} AS null_columns, count(*) AS rows FROM trx GROUP BY 1 ORDER BY 1').df()
    patterns['row_pct'] = patterns.rows / n * 100
    patterns.to_csv(OUT / 'null_count_per_row.csv', index=False)
    nullable = [c for c, v in zip(cols, counts) if v]
    pattern_expr = " || ',' || ".join(f"CASE WHEN {q(c)} IS NULL THEN '{c}' ELSE '' END" for c in nullable)
    con.execute(f'SELECT {pattern_expr} AS null_pattern, count(*) AS rows FROM trx GROUP BY 1 ORDER BY 2 DESC').df().to_csv(OUT / 'joint_patterns.csv', index=False)
    for group in ['fold', 'event_type']:
        table = con.execute(f'SELECT {q(group)}, count(*) AS rows, count(*) FILTER (WHERE ({null_count})>0) AS rows_any_null, {expr} FROM trx GROUP BY 1 ORDER BY 1').df()
        table.columns = [group, 'rows', 'rows_any_null'] + [c+'_null_n' for c in cols]
        table['rows_any_null_pct'] = table.rows_any_null / table.rows * 100
        table.to_csv(OUT / f'by_{group}.csv', index=False)
    old = pd.read_csv(ROOT / 'docs/report/tables/quality/raw_missing.csv')
    old = old[old.source.eq('trx')][['column', 'rows', 'missing_n']]
    check = per_col.merge(old, on='column', suffixes=('_raw', '_report'), how='outer', indicator=True)
    check['matches'] = check['_merge'].eq('both') & check.rows_raw.eq(check.rows_report) & check.null_n.eq(check.missing_n)
    check.to_csv(OUT / 'reconciliation.csv', index=False)
    assert check.matches.all(), 'Existing report differs; inspect reconciliation.csv'
    after = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted((ROOT / 'data/MBD-mini/detail/trx').rglob('*.parquet'))]
    assert before == after
    physical_cols = [c for c in cols if c != 'fold']
    summary = dict(source='All raw detail/trx shards; no sampling, deduplication or filtering', files=len(files), rows=n,
                   columns_including_partition=len(cols), physical_columns=len(physical_cols), null_cells=sum(counts),
                   cell_null_pct=100*sum(counts)/(n*len(cols)), physical_cell_null_pct=100*sum(counts)/(n*len(physical_cols)),
                   rows_any_null=affected, rows_any_null_pct=100*affected/n, complete_rows=n-affected,
                   existing_column_report_matches=True, raw_inventory_unchanged=True,
                   definition='SQL IS NULL; zero, negative values and category codes are not treated as null. Fold is a partition column. No time grouping; timezone does not affect these counts.')
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(per_col.to_string(index=False))
    print(patterns.to_string(index=False))
    con.close()


if __name__ == '__main__':
    main()
