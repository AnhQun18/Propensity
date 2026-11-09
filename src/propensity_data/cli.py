"""No work happens on import or without an explicit prepare command."""
import argparse
import json
import sys

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare point-in-time MBD-mini features; never train a model")
    parser.add_argument("command", choices=["plan", "prepare"])
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):
        parser.error("propensity_data requires Python >=3.11; the EDA package remains available on Python 3.10")
    config = load_config(args.config)
    if args.command == "plan":
        # Only configuration is read; no raw-data reads or output writes.
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
        return
    from .pipeline import prepare
    print(prepare(config))
