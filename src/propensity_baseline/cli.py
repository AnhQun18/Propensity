"""No data processing or training occurs without an explicit command."""
import argparse
import json

from .config import load_baseline_config, load_paths


def main(argv=None):
    parser = argparse.ArgumentParser(description='Calendar-history propensity baseline')
    parser.add_argument('command', choices=['calendar-plan', 'calendar', 'plan', 'tune', 'fit', 'test', 'compare-rule'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=['validation','test'], default='validation', help='compare-rule only')
    args = parser.parse_args(argv)
    if args.command.startswith('calendar'):
        paths = load_paths(args.config)
        if args.command == 'calendar-plan':
            print(json.dumps({k:str(v) for k,v in paths.items()}, indent=2))
            return
        from .data import build_calendar
        print(build_calendar(**paths))
        return
    config = load_baseline_config(args.config)
    if args.command == 'plan':
        print(json.dumps(config.to_dict(), indent=2))
        return
    if args.command == 'compare-rule':
        from .persistence import compare_rule
        print(compare_rule(config,stage=args.stage))
        return
    from .runner import fit_final, test_final, tune
    action = {'tune': tune, 'fit': fit_final, 'test': test_final}[args.command]
    print(action(config))


if __name__ == '__main__':
    main()
