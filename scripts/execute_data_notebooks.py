"""Execute notebooks 02 then 03 locally, checkpointing copies outside source control."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("02_data_preparation.ipynb", "03_data_processing.ipynb")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--only", choices=NOTEBOOKS)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(ROOT / "outputs"):
        parser.error("Execution artifacts must stay under repository outputs/")
    run_dir.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    os.environ["PYTHONPATH"] = str(ROOT / "src")
    os.environ["IPYTHONDIR"] = str(run_dir / "ipython")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(run_dir / "runtime")
    (run_dir / "runtime").mkdir()
    spec_dir = run_dir / "kernels" / "propensity-local"
    spec_dir.mkdir(parents=True)
    write_json(spec_dir / "kernel.json", {
        "argv": [sys.executable, "-B", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Propensity local venv", "language": "python",
    })
    report = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable, "python_version": sys.version,
        "versions": {p: importlib.metadata.version(p) for p in
                     ("duckdb", "pandas", "pyarrow", "nbformat", "nbclient", "ipykernel")},
        "notebooks": [], "raw_modified": False,
    }
    write_json(run_dir / "execution.json", report)
    event_log = (run_dir / "cells.jsonl").open("a", encoding="utf-8", buffering=1)

    def event(data):
        data["at"] = datetime.now(timezone.utc).isoformat()
        line = json.dumps(data, ensure_ascii=False)
        event_log.write(line + "\n")
        print(line, flush=True)

    try:
        for filename in ((args.only,) if args.only else NOTEBOOKS):
            source = ROOT / "notebooks" / filename
            destination = run_dir / filename
            notebook = nbformat.read(source, as_version=4)
            item = {"notebook": filename, "status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
            report["notebooks"].append(item)
            write_json(run_dir / "execution.json", report)
            started = {}

            def on_start(cell, cell_index):
                if cell.cell_type == "code":
                    started[cell_index] = time.monotonic()
                    item["active_cell"] = cell.id
                    write_json(run_dir / "execution.json", report)
                    event({"notebook": filename, "event": "cell_start", "index": cell_index, "id": cell.id})

            def on_executed(cell, cell_index, execute_reply):
                seconds = round(time.monotonic() - started[cell_index], 3)
                event({"notebook": filename, "event": "cell_finish", "index": cell_index,
                       "id": cell.id, "seconds": seconds, "status": execute_reply["content"]["status"]})
                nbformat.write(notebook, destination)

            manager = KernelManager(
                kernel_name="propensity-local",
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(run_dir / "kernels")]),
                connection_file=str(run_dir / "runtime" / (source.stem + ".json")),
            )
            client = NotebookClient(
                notebook, km=manager, timeout=None, startup_timeout=120, iopub_timeout=120,
                resources={"metadata": {"path": str(ROOT)}}, store_widget_state=False,
                record_timing=True, on_cell_start=on_start, on_cell_executed=on_executed,
            )
            try:
                client.execute()
                item["status"] = "ready"
                item["executed_code_cells"] = sum(c.cell_type == "code" and c.execution_count is not None
                                                  for c in notebook.cells)
                notebook.metadata.setdefault("workflow", {})["execution_status"] = "executed_successfully"
            except BaseException as exc:
                item["status"] = "failed"
                item["error"] = f"{type(exc).__name__}: {exc}"
                notebook.metadata.setdefault("workflow", {})["execution_status"] = "failed"
                raise
            finally:
                item["finished_at"] = datetime.now(timezone.utc).isoformat()
                nbformat.write(notebook, destination)
                write_json(run_dir / "execution.json", report)
                # A supplied KernelManager is caller-owned in nbclient.
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
            event({"notebook": filename, "event": "notebook_finished", "status": item["status"]})
        report["status"] = "ready"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        (run_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(run_dir / "execution.json", report)
        event_log.close()


if __name__ == "__main__":
    main()
