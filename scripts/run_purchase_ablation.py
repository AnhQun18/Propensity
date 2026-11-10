"""Run fixed, validation-only purchase/history comparisons without opening test."""
import argparse
from propensity_baseline.config import load_baseline_config
from propensity_baseline.ablation import run_ablation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/propensity_ablation.json')
    args = parser.parse_args()
    print(run_ablation(load_baseline_config(args.config)))
