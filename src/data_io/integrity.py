"""Source fingerprinting, non-overlapping path guards, and atomic run manifests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def json_safe(value: Any) -> Any:
    """Recursively convert values into strict JSON-serializable structures (no NaN/Inf)."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(json_safe(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: str | Path, value: Any, *, allow_nan: bool = False) -> Path:
    """Write formatted UTF-8 JSON with strict NaN rejection by default."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = value if allow_nan else json_safe(value)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=allow_nan) + "\n",
        encoding="utf-8",
    )
    return target


def assert_non_overlapping_paths(output_dir: str | Path, protected_paths: Sequence[str | Path]) -> None:
    """Prevent writing outputs inside or overwriting any protected input directory."""
    out = Path(output_dir).resolve()
    for raw in protected_paths:
        if raw is None:
            continue
        protected = Path(raw).resolve()
        if out == protected or protected in out.parents or out in protected.parents:
            raise ValueError(f"Output path {out} must not overlap protected path {protected}")


def compute_file_inventory(
    source_files: Mapping[str, Sequence[Path]] | Sequence[Path],
    *,
    base_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic (source, path, bytes, mtime_ns) records for immutability checks."""
    base = Path(base_dir).resolve() if base_dir is not None else None
    records: list[dict[str, Any]] = []

    if isinstance(source_files, Mapping):
        items = [(key, paths) for key, paths in sorted(source_files.items())]
    else:
        items = [("source", source_files)]

    for source_name, paths in items:
        for path in sorted(Path(p) for p in paths):
            resolved = path.resolve()
            rel = resolved.relative_to(base).as_posix() if base is not None and resolved.is_relative_to(base) else resolved.as_posix()
            stat = resolved.stat()
            records.append({
                "source": source_name,
                "path": rel,
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
    return records


def inventory_sha256(inventory: list[dict[str, Any]]) -> str:
    """Deterministic SHA-256 fingerprint over a file inventory list."""
    blob = json.dumps(inventory, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def verify_inventory_unchanged(
    expected_inventory: list[dict[str, Any]],
    source_files: Mapping[str, Sequence[Path]] | Sequence[Path],
    *,
    base_dir: str | Path | None = None,
) -> None:
    """Raise ValueError if any input file was added, removed, resized, or modified."""
    current = compute_file_inventory(source_files, base_dir=base_dir)
    if current != expected_inventory:
        raise ValueError("Source file inventory changed during execution")


class RunManifest:
    """Context manager tracking an isolated run's lifecycle: running -> ready / failed."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        protected_paths: Sequence[str | Path] = (),
        exist_ok: bool = False,
        initial: dict[str, Any] | None = None,
        filename: str = "manifest.json",
    ):
        self.output_dir = Path(output_dir).resolve()
        self.protected_paths = tuple(protected_paths)
        self.exist_ok = exist_ok
        self.filename = filename
        self.data: dict[str, Any] = {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            **(initial or {}),
        }

    @property
    def path(self) -> Path:
        return self.output_dir / self.filename

    def save(self) -> Path:
        return write_json(self.path, self.data)

    def update(self, **fields: Any) -> Path:
        self.data.update(fields)
        return self.save()

    def __enter__(self) -> "RunManifest":
        assert_non_overlapping_paths(self.output_dir, self.protected_paths)
        self.output_dir.mkdir(parents=True, exist_ok=self.exist_ok)
        self.save()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.data["status"] = "failed"
            self.data["error"] = f"{exc_type.__name__}: {exc}"
        else:
            self.data["status"] = "ready"
            self.data["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.save()
        return False
