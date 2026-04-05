import os
import yaml
from pathlib import Path


def load_config(path: str = None) -> dict:
    if path is None:
        root = Path(__file__).resolve().parents[2]
        path = root / "configs" / "config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)
