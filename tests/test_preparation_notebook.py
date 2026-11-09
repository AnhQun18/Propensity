"""Notebook text contract. These checks never execute its cells."""
import ast
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PreparationNotebookTests(unittest.TestCase):
    def test_structure_and_python_text(self):
        notebook = json.loads((ROOT / "notebooks/02_data_preparation.ipynb").read_text(encoding="utf-8"))
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
        for name in ("prep-execute-labels", "prep-normalize-trx", "prep-normalize-geo",
                     "prep-normalize-dialog", "prep-build-months", "prep-supervised-reader", "prep-finish"):
            self.assertIn(name, identifiers)


if __name__ == "__main__":
    unittest.main()
