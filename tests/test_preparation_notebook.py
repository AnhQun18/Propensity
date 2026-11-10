"""Notebook text contract. These checks never execute its cells."""
import ast
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PreparationNotebookTests(unittest.TestCase):
    def test_structure_and_python_text(self):
        self.check_notebook("02_data_preparation.ipynb", "prep-")

    def test_processing_notebook_structure_and_source_text(self):
        self.check_notebook("03_data_processing.ipynb", "processing-")

    def check_notebook(self, filename, prefix):
        notebook = json.loads((ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        spec = importlib.util.spec_from_file_location("notebook_sync", ROOT / "scripts/sync_preparation_notebook.py")
        sync = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sync)  # Only imports authoring helpers; no notebook execution.
        identifiers = []
        linked = 0
        for cell in notebook["cells"]:
            identifiers.append(cell["id"])
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            self.assertNotIn("execution", cell["metadata"])
            text = "".join(cell["source"])
            ast.parse(text)  # Parse only, never compile/exec/evaluate.
            self.assertNotIn("prepare(config)", text)
            if "source_file" in cell["metadata"]:
                linked += 1
                self.assertEqual(text, sync.render_library_cell(ROOT, cell["metadata"]), cell["id"])
        self.assertGreaterEqual(linked, 8)
        self.assertEqual(len(identifiers), len(set(identifiers)))
        for name in ("execute-labels", "normalize-trx", "normalize-geo",
                     "normalize-dialog", "build-months", "supervised-reader", "finish", "validation-source"):
            self.assertIn(prefix + name, identifiers)
        if prefix == "processing-":
            self.assertIn(prefix + "export-datasets", identifiers)
            text = "".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
            self.assertIn('ready_manifest.get("status") != "ready"', text)
            self.assertIn("all_four_targets_missing", text)


if __name__ == "__main__":
    unittest.main()
