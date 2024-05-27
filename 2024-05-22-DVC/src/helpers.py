import pathlib
import pickle
from typing import Any

import yaml


def load_yaml(path: pathlib.Path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def dump_pickle(obj, path: pathlib.Path) -> None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: pathlib.Path) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)
