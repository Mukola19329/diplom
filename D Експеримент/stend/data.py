from dataclasses import dataclass

import numpy as np

from config import ExperimentConfig


@dataclass(frozen=True)
class ServiceSystem:
    dependencies: np.ndarray
    true_criticality: np.ndarray
    declared_priority: np.ndarray
    base_demand: np.ndarray


def generate_system(rng: np.random.Generator, config: ExperimentConfig) -> ServiceSystem:
    n = config.n_services
    dependencies = np.zeros((n, n), dtype=bool)
    order = rng.permutation(n)

    for position in range(1, n):
        child = order[position]
        parent = order[rng.integers(0, position)]
        dependencies[parent, child] = True

    extra_edges = max(0, config.n_edges - (n - 1))
    for _ in range(extra_edges):
        parent, child = rng.choice(n, size=2, replace=False)
        if parent != child:
            dependencies[parent, child] = True

    true_criticality = rng.beta(2.5, 1.8, size=n)
    declared_priority = np.clip(
        0.70 * true_criticality + 0.30 * rng.normal(0.5, 0.25, size=n),
        0.0,
        1.0,
    )
    base_demand = rng.uniform(0.02, 0.08, size=n)
    base_demand /= base_demand.sum()

    return ServiceSystem(
        dependencies=dependencies,
        true_criticality=true_criticality,
        declared_priority=declared_priority,
        base_demand=base_demand,
    )


def delivery_from_active(
    active: np.ndarray,
    dependencies: np.ndarray,
    proximity: float,
    config: ExperimentConfig,
) -> np.ndarray:
    floor = (
        config.dependency_floor_far * (1.0 - proximity)
        + config.dependency_floor_near * proximity
    )
    delivery = active.astype(float).copy()

    for _ in range(config.n_steps):
        next_delivery = delivery.copy()
        for child in range(len(active)):
            if not active[child]:
                next_delivery[child] = 0.0
                continue
            parents = dependencies[:, child]
            if not parents.any():
                next_delivery[child] = 1.0
                continue
            support = float(delivery[parents].mean())
            next_delivery[child] = floor + (1.0 - floor) * support
        if np.allclose(next_delivery, delivery, atol=1e-8):
            break
        delivery = next_delivery

    return delivery
