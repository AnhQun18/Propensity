"""Execute EDA in the local Python, checkpointing every cell and recording failures."""
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

ROOT=Path(__file__).resolve().parents[1]
run=ROOT/"outputs/eda_execution"
run.mkdir(parents=True,exist_ok=True)
os.environ["IPYTHONDIR"]=str(run/"ipython")
os.environ["JUPYTER_RUNTIME_DIR"]=str(run/"runtime")
(run/"runtime").mkdir(exist_ok=True)
kernel=run/"kernels/eda-local"
kernel.mkdir(parents=True,exist_ok=True)
(kernel/"kernel.json").write_text(json.dumps({"argv":[sys.executable,"-B","-m","ipykernel_launcher","-f","{connection_file}"],"display_name":"EDA local","language":"python"}),encoding="utf-8")
source=ROOT/"notebooks/eda.ipynb"
nb=nbformat.read(source,as_version=4)
nb.metadata["eda_execution"]={"status":"running"}
started={}
events=[]
def save():
    nbformat.write(nb,source)
    nbformat.write(nb,run/"eda.executed.ipynb")
    (run/"cells.json").write_text(json.dumps(events,indent=2),encoding="utf-8")
def on_start(cell,cell_index):
    started[cell_index]=time.monotonic()
    if cell.cell_type=="code":print(f"Cell {cell_index}: {cell.source.splitlines()[0]}",flush=True)
def on_done(cell,cell_index,execute_reply):
    event={"index":cell_index,"seconds":round(time.monotonic()-started[cell_index],2),"status":execute_reply["content"]["status"]}
    events.append(event); print(event,flush=True);save()
manager=KernelManager(kernel_name="eda-local",kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(run/"kernels")]),connection_file=str(run/"runtime/connection.json"))
client=NotebookClient(nb,km=manager,timeout=None,startup_timeout=120,iopub_timeout=120,
                      resources={"metadata":{"path":str(ROOT)}},store_widget_state=False,
                      on_cell_start=on_start,on_cell_executed=on_done)
try:
    client.execute()
    nb.metadata["eda_execution"]={"status":"ready","code_cells":sum(c.cell_type=="code" for c in nb.cells)}
    if (run/"error.log").exists():(run/"error.log").unlink()
except BaseException:
    nb.metadata["eda_execution"]={"status":"failed"}
    (run/"error.log").write_text(traceback.format_exc(),encoding="utf-8")
    raise
finally:
    save()
    if manager.has_kernel:manager.shutdown_kernel(now=True)
print("EDA notebook executed successfully",flush=True)
