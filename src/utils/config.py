from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(name: str) -> dict:
    """Load a YAML config by filename, e.g. load_config('base.yaml')."""
    path = CONFIG_DIR / name
    with open(path) as f:
        return yaml.safe_load(f)


def horizons_to_steps(horizons_minutes, dt_seconds: int):
    steps = []
    for minutes in horizons_minutes:
        seconds = minutes * 60
        if seconds % dt_seconds != 0:
            raise ValueError(
                f"horizon {minutes}min is not a whole multiple of dt_seconds={dt_seconds}"
            )
        steps.append(seconds // dt_seconds)
    return steps
