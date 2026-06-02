"""Shared config loading for all pipeline stages.

Every stage reads a global ``config/base.yaml`` plus its own
``config/stage{N}.yaml``. ``load_configs`` is the single implementation used by
Stages 1-4 (previously copy-pasted into each stage entry point).
"""

from pathlib import Path

import yaml


def load_configs(base_path: str, stage_path: str) -> tuple[dict, dict]:
    """Load base and per-stage YAML configs.

    Args:
        base_path: Path to config/base.yaml.
        stage_path: Path to the stage's config (e.g. config/stage3.yaml).

    Returns:
        Tuple of (base_cfg, stage_cfg) as plain dicts.

    Raises:
        FileNotFoundError: If either config file is missing.
    """
    for p in (base_path, stage_path):
        if not Path(p).exists():
            raise FileNotFoundError(f"Config file not found: {p}")
    with open(base_path) as f:
        base_cfg = yaml.safe_load(f)
    with open(stage_path) as f:
        stage_cfg = yaml.safe_load(f)
    return base_cfg, stage_cfg
