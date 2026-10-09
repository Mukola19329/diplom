from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CONFIG, OUT_DIR, POLICIES
from data import generate_system
from metrics import result_to_record, summarize
from policies import simulate_policy


def _run_seed(
    repeat: int,
    policy_index: int,
    load_index: int,
    proximity_index: int,
    budget_index: int,
    error_index: int,
) -> int:
    values = (
        CONFIG.seed,
        repeat,
        policy_index,
        load_index,
        proximity_index,
        budget_index,
        error_index,
    )
    seed = 0
    for value in values:
        seed = (seed * 1_000_003 + value + 17) % (2**63 - 1)
    return seed


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, float | int | str]] = []

    for repeat in range(CONFIG.n_repeats):
        system_rng = np.random.default_rng(CONFIG.seed + repeat)
        system = generate_system(system_rng, CONFIG)
        for load_index, load_factor in enumerate(CONFIG.load_levels):
            for proximity_index, proximity in enumerate(CONFIG.cascade_proximities):
                for budget_index, decision_budget in enumerate(CONFIG.decision_budgets):
                    for error_index, inference_error in enumerate(CONFIG.inference_errors):
                        for policy_index, policy in enumerate(POLICIES):
                            run_rng = np.random.default_rng(
                                _run_seed(
                                    repeat,
                                    policy_index,
                                    load_index,
                                    proximity_index,
                                    budget_index,
                                    error_index,
                                )
                            )
                            result = simulate_policy(
                                system=system,
                                policy=policy,
                                load_factor=load_factor,
                                proximity=proximity,
                                decision_budget=decision_budget,
                                inference_error=inference_error,
                                rng=run_rng,
                                config=CONFIG,
                            )
                            records.append(result_to_record(result, repeat))

    raw = pd.DataFrame(records)
    summary = summarize(records)
    run_date = date.today().isoformat()
    raw_path = OUT_DIR / f"raw_runs__nd5__seed{CONFIG.seed}__n{CONFIG.n_repeats}__{run_date}.csv"
    summary_path = OUT_DIR / f"summary__nd5__seed{CONFIG.seed}__n{CONFIG.n_repeats}__{run_date}.csv"
    plot_path = OUT_DIR / f"critical_loss__nd5__seed{CONFIG.seed}__n{CONFIG.n_repeats}__{run_date}.png"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    _plot_summary(summary, plot_path)
    return raw, summary, run_date


def _plot_summary(summary: pd.DataFrame, output_path) -> None:
    selected_load = CONFIG.load_levels[1]
    selected_budget = CONFIG.decision_budgets[1]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for axis, proximity in zip(axes, CONFIG.cascade_proximities):
        for policy in POLICIES:
            subset = summary[
                (summary["policy"] == policy)
                & (summary["load_factor"] == selected_load)
                & (summary["decision_budget"] == selected_budget)
                & (summary["proximity"] == proximity)
            ].sort_values("inference_error")
            axis.errorbar(
                subset["inference_error"],
                subset["critical_value_loss_mean"],
                yerr=subset["critical_value_loss_std"].fillna(0.0),
                marker="o",
                capsize=3,
                label=policy,
            )
        axis.set_title(f"Каскадна близькість = {proximity:.2f}")
        axis.set_xlabel("Похибка інференції")
        axis.grid(alpha=0.3)

    axes[0].set_ylabel("Втрата критичної цінності")
    axes[-1].legend(fontsize=8)
    figure.suptitle(
        f"ND-5: втрата критичної цінності, навантаження={selected_load:.2f}, "
        f"бюджет рішень={selected_budget}"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    raw, summary, run_date = run_experiment()
    print(f"Прогон завершено: {len(raw)} рядків, дата {run_date}")
    print(summary[["policy", "critical_value_loss_mean", "critical_value_loss_std"]].head(12).to_string(index=False))
