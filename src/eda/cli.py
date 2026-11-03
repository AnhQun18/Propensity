"""JSON-configured CLI. Relative input paths resolve beside the config file."""
import argparse
import json
from pathlib import Path
from .config import DatasetSpec, FileSource, Relationship
from .pipeline import profile, profile_dataset
from .report import export_report


def load_config(path):
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    def source(config):
        item = FileSource(**config)
        item.paths = [str((path.parent / p).resolve()) if not Path(p).is_absolute() else p for p in item.paths]
        return item
    if "tables" in document:
        sources, specs = {}, {}
        for name, config in document["tables"].items():
            sources[name] = source(config["source"])
            specs[name] = DatasetSpec(**{"dataset": name, **config.get("spec", {})})
        return profile_dataset(sources, specs, dataset=document.get("dataset", "dataset"),
                               relationships=[Relationship(**r) for r in document.get("relationships", [])])
    return profile(source(document["source"]), DatasetSpec(**document.get("spec", {})))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Descriptive statistics and EDA; default output: docs/<dataset>/")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plots", action="store_true", help="Export PNG charts (requires plots extra)")
    parser.add_argument("--notebook", action="store_true", help="Export a static report notebook")
    args = parser.parse_args(argv)
    result = load_config(args.config)
    destination = export_report(result, args.output_dir, plots=args.plots, notebook=args.notebook)
    print(f"EDA saved to {destination.resolve()}")
    return 0
