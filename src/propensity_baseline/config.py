"""Small, explicit experiment grid; test is a separate command."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

from .data import read_json


VARIANTS = ('trx_recent', 'trx_history', 'multi_recent', 'multi_history')


@dataclass(frozen=True)
class BaselineConfig:
    source_dir: Path
    output_dir: Path
    seed: int = 20261110
    threads: int = 2
    depths: tuple[int, ...] = (4, 6)
    iterations: tuple[int, ...] = (200, 500)
    learning_rate: float = .04
    l2_leaf_reg: float = 5.
    variants: tuple[str, ...] = ('multi_recent', 'multi_history')
    bootstrap_replicates: int = 500

    def __post_init__(self):
        for name in ('source_dir', 'output_dir'):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        if (self.source_dir == self.output_dir or self.source_dir in self.output_dir.parents
                or self.output_dir in self.source_dir.parents):
            raise ValueError('Model artifacts and feature inputs must not overlap')
        for name in ('depths', 'iterations', 'variants'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
            if not getattr(self, name) or len(set(getattr(self, name))) != len(getattr(self, name)):
                raise ValueError(f'{name} must be nonempty and unique')
        if any(type(d) is not int or not 1 <= d <= 10 for d in self.depths):
            raise ValueError('Depth must be an integer in [1,10]')
        if any(type(n) is not int or n < 1 for n in self.iterations):
            raise ValueError('Iterations must be positive integers')
        if any(v not in VARIANTS for v in self.variants):
            raise ValueError('Unknown feature variant')
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError('Seed must be an integer in [0,2**31)')
        if type(self.threads) is not int or not 1 <= self.threads <= 8:
            raise ValueError('threads must be an integer in [1,8]')
        if type(self.bootstrap_replicates) is not int or self.bootstrap_replicates < 0:
            raise ValueError('bootstrap_replicates must be a nonnegative integer')
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1:
            raise ValueError('Invalid learning rate')
        if not math.isfinite(self.l2_leaf_reg) or self.l2_leaf_reg <= 0:
            raise ValueError('Invalid l2_leaf_reg')

    def to_dict(self):
        result = asdict(self)
        for key in ('source_dir', 'output_dir'):
            result[key] = str(result[key])
        return result


def load_baseline_config(path):
    path = Path(path).resolve()
    options = read_json(path)
    for key in ('source_dir', 'output_dir'):
        options[key] = (path.parent / options[key]).resolve()
    return BaselineConfig(**options)


def load_paths(path):
    path = Path(path).resolve()
    options = read_json(path)
    if set(options) != {'source_dir', 'output_dir'}:
        raise ValueError('Calendar config requires source_dir and output_dir only')
    return {key: (path.parent / value).resolve() for key, value in options.items()}
