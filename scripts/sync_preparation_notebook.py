"""Update visible helper cells as text; never execute a notebook or read raw."""
import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/02_data_preparation.ipynb"


def render_library_cell(root: Path, metadata: dict) -> str:
    path = (root / metadata["source_file"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Cell source must stay within the repository")
    source = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    if metadata["source_kind"] == "module":
        # Current relative imports occupy one line; notebook definitions share globals.
        source = "\n".join(line for line in source.splitlines() if not line.startswith("from ."))
    elif metadata["source_kind"] == "function":
        node = next((n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                     and n.name == metadata["symbol"]), None)
        if node is None:
            raise ValueError(f"Missing function: {metadata['symbol']}")
        source = ast.get_source_segment(source, node)
    else:
        raise ValueError("Unknown source_kind")
    return source.strip() + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare only; do not write")
    args = parser.parse_args(argv)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    stale = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        if "source_file" in cell["metadata"]:
            expected = render_library_cell(ROOT, cell["metadata"])
            if "".join(cell["source"]) != expected:
                stale.append(cell["id"])
            cell["source"] = expected.splitlines(keepends=True)
        if cell.get("execution_count") is not None or cell.get("outputs"):
            stale.append(cell["id"] + ": output")
        cell["execution_count"] = None
        cell["outputs"] = []
        cell["metadata"].pop("execution", None)
    if args.check:
        if stale:
            raise SystemExit("Notebook needs synchronization: " + ", ".join(stale))
        print("Source cells match; no execution outputs.")
        return
    notebook["metadata"]["workflow"]["execution_status"] = "not_run"
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Updated {NOTEBOOK}; no notebook code was executed.")


if __name__ == "__main__":
    main()
