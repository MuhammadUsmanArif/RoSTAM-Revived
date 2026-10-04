"""
RoSTAM Configuration Loader
============================
Loads YAML configuration files into typed dataclasses for safe, IDE-friendly
parameter access throughout the codebase.

Usage
-----
    from rostam.config import load_config
    cfg = load_config("config/default.yaml")
    print(cfg.ea.num_generations)
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ExperimentConfig:
    name: str = "rostam_experiment"
    output_dir: str = "./results"
    problem_distribution: str = "ST-SR-TA"


@dataclass
class EnvironmentConfig:
    num_robots: int = 3
    num_tasks: int = 50
    map_dim: int = 6
    task_type: int = 1
    speed_matrix: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    failed_robots: List[int] = field(default_factory=lambda: [1, 1, 1])
    hidden_task_percentage: float = 0.0


@dataclass
class EAConfig:
    population_size: int = 100
    num_generations: int = 1500
    crossover_prob: float = 1.0
    mutation_prob: float = 0.3
    seed: Optional[int] = None
    ais_injection_freq: int = 10
    ais_inject_count: int = 20


@dataclass
class WindowConfig:
    enabled: bool = False
    window_size: int = 3


@dataclass
class PenaltyConfig:
    mode: str = "adaptive"
    initial_value: float = 1.0
    adaptive_window: int = 10
    adaptive_rate: float = 0.15
    max_value: float = 20.0


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_frequency: int = 100


@dataclass
class RoSTAMConfig:
    """Top-level configuration container for a RoSTAM experiment."""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    ea: EAConfig = field(default_factory=EAConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    penalty: PenaltyConfig = field(default_factory=PenaltyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _dict_to_dataclass(cls, data: dict):
    """Recursively converts a nested dict into a nested dataclass instance."""
    import dataclasses
    if not dataclasses.is_dataclass(cls):
        return data
    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for f in dataclasses.fields(cls):
        value = data.get(f.name)
        if value is None:
            continue
        # Resolve forward references stored as strings
        ftype = field_types[f.name]
        if isinstance(ftype, str):
            ftype = eval(ftype)
        if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _dict_to_dataclass(ftype, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> RoSTAMConfig:
    """
    Load a YAML configuration file and return a RoSTAMConfig object.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    RoSTAMConfig
        Fully populated configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = RoSTAMConfig()
    if "experiment" in raw:
        cfg.experiment = _dict_to_dataclass(ExperimentConfig, raw["experiment"])
    if "environment" in raw:
        cfg.environment = _dict_to_dataclass(EnvironmentConfig, raw["environment"])
    if "ea" in raw:
        cfg.ea = _dict_to_dataclass(EAConfig, raw["ea"])
    if "window" in raw:
        cfg.window = _dict_to_dataclass(WindowConfig, raw["window"])
    if "penalty" in raw:
        cfg.penalty = _dict_to_dataclass(PenaltyConfig, raw["penalty"])
    if "logging" in raw:
        cfg.logging = _dict_to_dataclass(LoggingConfig, raw["logging"])
    return cfg
