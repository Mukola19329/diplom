from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "out"


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    n_services: int = 24
    n_edges: int = 36
    n_steps: int = 12
    n_repeats: int = 5
    capacity: float = 1.0
    load_levels: tuple[float, ...] = (1.05, 1.15, 1.25)
    cascade_proximities: tuple[float, ...] = (0.20, 0.70)
    decision_budgets: tuple[int, ...] = (1, 3, 6)
    inference_errors: tuple[float, ...] = (0.00, 0.15, 0.30)
    dependency_floor_far: float = 0.60
    dependency_floor_near: float = 0.25
    overload_penalty_exponent: float = 1.5


CONFIG = ExperimentConfig()
POLICIES = (
    "fixed_declared",
    "perfect_inference",
    "inferred_independent",
    "inferred_correlated",
)
