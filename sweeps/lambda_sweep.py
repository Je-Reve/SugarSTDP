"""
Lambda sweep runner. Loop infrastructure only; the penalty terms
themselves live in regularization.py (spike_count_penalty for System A,
apply_reward_shaping for System B) and config.lambda_sweep_values.

Sweep design (locked): INDEPENDENT of the k-sweep, not crossed - runs at
k=1 only. A full cross (5 lambda x 4 k x 2 systems x 3 seeds = 120 runs)
exceeds the Colab compute budget. Adjust the upper bound of
config.lambda_sweep_values after seeing unregularized (lambda=0) spike
magnitudes from the first training run.

Output feeds plotting.plot_pareto_frontier: Task Reward vs. estimated
joules-emulated (energy.py), both systems, across lambda values.
"""

from typing import Callable, Dict, List

from config import Config
from sugarstdp.episode_loop import run_training
from sugarstdp.logger import ExperimentLogger
from sugarstdp.system_a import SystemA
from sugarstdp.system_b import SystemB

SYSTEM_BUILDERS: Dict[str, Callable] = {
    "A": lambda cfg: SystemA(cfg),
    "B": lambda cfg: SystemB(cfg),
}


def run_lambda_sweep(config: Config, n_episodes: int, log_dir: str = "logs") -> List[dict]:
    """
    Loop over config.lambda_sweep_values x {"A", "B"} x config.seeds at
    k=1, calling episode_loop.run_training for each combination. Collects
    mean episode reward and an energy estimate (via energy.py) per
    (system, lambda), for the Pareto frontier plot.

    Requires energy.py and both systems' update() to be implemented first
    - this only orchestrates calls into them, it contains no conceptual
    learning-rule or energy-estimation logic itself.
    """
    results = []
    for system_name, builder in SYSTEM_BUILDERS.items():
        for lam in config.lambda_sweep_values:
            seed_rewards = []
            for seed in config.seeds:
                run_name = f"lambda_sweep_system{system_name}_lam{lam}_seed{seed}"
                logger = ExperimentLogger(run_name, config, log_dir=log_dir)

                rewards = run_training(
                    system_builder=builder,
                    config=config,
                    seed=seed,
                    n_episodes=n_episodes,
                    k=1,
                    lam=lam,
                    logger=logger,
                )
                logger.close()
                seed_rewards.append(rewards)

                # TODO once energy.py is implemented: read total_spikes
                # back out of the CSV logger.csv_path just wrote (or have
                # run_training return it directly), then call
                # energy.compute_avg_fan_out / energy.estimate_energy_joules
                # here to populate seed_energy_estimates for the Pareto plot.

            results.append(
                {
                    "system": system_name,
                    "lambda": lam,
                    "seed_rewards": seed_rewards,
                }
            )
    return results
