"""Chunk readers; paths are explicit and independent of project layout."""
from pathlib import Path
import pandas as pd
import hashlib
from .config import FileSource


def iter_chunks(source: FileSource, chunk_size=100_000):
    expected = None
    for filename in source.paths:
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(path)
        kind = source.format
        if kind == "auto":
            kind = "parquet" if path.is_dir() or path.suffix.lower() in {".parquet", ".pq"} else "csv"
        if kind == "csv":
            reader = pd.read_csv(path, chunksize=chunk_size, **source.read_options)
            with reader:
                for chunk in reader:
                    if expected is not None and list(chunk.columns) != expected:
                        raise ValueError(f"Schema changed between files/chunks: {path}")
                    expected = list(chunk.columns)
                    yield chunk
        else:
            import pyarrow.dataset as ds
            options = dict(source.read_options)
            columns = options.pop("columns", None)
            partitioning = options.pop("partitioning", "hive")
            if options:
                raise ValueError(f"Unsupported Parquet options: {sorted(options)}")
            dataset = ds.dataset(path, format="parquet", partitioning=partitioning)
            scanner = dataset.scanner(columns=columns, batch_size=chunk_size)
            seen = False
            for batch in scanner.to_batches():
                seen = True
                chunk = batch.to_pandas()
                if expected is not None and list(chunk.columns) != expected:
                    raise ValueError(f"Schema changed between files/chunks: {path}")
                expected = list(chunk.columns)
                yield chunk
            if not seen:
                chunk = dataset.head(0, columns=columns).to_pandas()
                if expected is not None and list(chunk.columns) != expected:
                    raise ValueError(f"Schema changed between files/chunks: {path}")
                expected = list(chunk.columns)
                yield chunk


def source_manifest(source, hash_sources=True):
    """Inventory physical files once, including partition files in directories."""
    rows = []
    seen = set()
    for name in source.paths:
        path = Path(name)
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".parquet", ".pq"}) if path.is_dir() else [path]
        for file in files:
            resolved = file.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            digest = None
            if hash_sources:
                hasher = hashlib.sha256()
                with file.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        hasher.update(block)
                digest = hasher.hexdigest()
            rows.append(dict(path=str(resolved), bytes=file.stat().st_size, sha256=digest,
                             hash_status="computed" if hash_sources else "disabled"))
    return pd.DataFrame(rows, columns=["path", "bytes", "sha256", "hash_status"])
