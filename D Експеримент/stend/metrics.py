from dataclasses import asdict

import pandas as pd

from policies import SimulationResult


def result_to_record(result: SimulationResult, repeat: int) -> dict[str, float | int | str]:
    record = asdict(result)
    record["repeat"] = repeat
    return record


def summarize(records: list[dict[str, float | int | str]]) -> pd.DataFrame:
    raw = pd.DataFrame(records)
    group_columns = [
        "policy",
        "load_factor",
        "proximity",
        "decision_budget",
        "inference_error",
    ]
    metric_columns = [
        "critical_value_loss",
        "direct_loss",
        "cascade_loss",
        "dropped_services",
        "decision_steps",
        "unresolved_overload",
    ]
    summary = raw.groupby(group_columns)[metric_columns].agg(["mean", "std"]).reset_index()
    summary.columns = group_columns + [
        f"{metric}_{statistic}"
        for metric in metric_columns
        for statistic in ("mean", "std")
    ]
    return summary
