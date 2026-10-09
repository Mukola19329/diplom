from dataclasses import dataclass

import numpy as np

from config import ExperimentConfig
from data import ServiceSystem, delivery_from_active


@dataclass(frozen=True)
class SimulationResult:
    policy: str
    load_factor: float
    proximity: float
    decision_budget: int
    inference_error: float
    critical_value_loss: float
    direct_loss: float
    cascade_loss: float
    dropped_services: int
    decision_steps: int
    unresolved_overload: float


def _scores(
    system: ServiceSystem,
    policy: str,
    active: np.ndarray,
    delivery: np.ndarray,
    inference_error: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if policy == "fixed_declared":
        return system.declared_priority.copy()
    if policy == "perfect_inference":
        return system.true_criticality.copy()

    noise = rng.normal(0.0, inference_error, size=len(active))
    if policy == "inferred_independent":
        visibility = np.ones(len(active))
    elif policy == "inferred_correlated":
        visibility = delivery
    else:
        raise ValueError(f"Unknown policy: {policy}")

    return np.clip(system.true_criticality * visibility + noise, 0.0, 1.0)


def simulate_policy(
    system: ServiceSystem,
    policy: str,
    load_factor: float,
    proximity: float,
    decision_budget: int,
    inference_error: float,
    rng: np.random.Generator,
    config: ExperimentConfig,
) -> SimulationResult:
    active = np.ones(config.n_services, dtype=bool)
    demand = system.base_demand * load_factor
    decision_steps = 0

    while decision_steps < decision_budget:
        delivery = delivery_from_active(active, system.dependencies, proximity, config)
        carried_demand = float(np.dot(demand, delivery))
        if carried_demand <= config.capacity:
            break

        scores = _scores(
            system,
            policy,
            active,
            delivery,
            inference_error,
            rng,
        )
        scores[~active] = np.inf
        drop_index = int(np.argmin(scores))
        active[drop_index] = False
        decision_steps += 1

    delivery_before_overload = delivery_from_active(
        active,
        system.dependencies,
        proximity,
        config,
    )
    carried_demand = float(np.dot(demand, delivery_before_overload))
    unresolved_overload = max(0.0, carried_demand / config.capacity - 1.0)

    if unresolved_overload > 0.0:
        overload_factor = max(
            0.0,
            1.0 - unresolved_overload ** config.overload_penalty_exponent,
        )
        delivery = delivery_before_overload * overload_factor
    else:
        delivery = delivery_before_overload

    total_criticality = float(system.true_criticality.sum())
    critical_value_loss = float(
        np.dot(system.true_criticality, 1.0 - delivery) / total_criticality
    )
    direct_loss = float(
        np.dot(system.true_criticality, ~active) / total_criticality
    )
    cascade_loss = max(0.0, critical_value_loss - direct_loss)

    return SimulationResult(
        policy=policy,
        load_factor=load_factor,
        proximity=proximity,
        decision_budget=decision_budget,
        inference_error=inference_error,
        critical_value_loss=critical_value_loss,
        direct_loss=direct_loss,
        cascade_loss=cascade_loss,
        dropped_services=int((~active).sum()),
        decision_steps=decision_steps,
        unresolved_overload=unresolved_overload,
    )
