from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

@dataclass
class ExperimentConfig:
    project_root: Path
    experiment_path: Path
    experiment_name: str
    n_outer_folds: int
    n_inner_folds: int
    num_trials: int
    random_seed: int
    device: str