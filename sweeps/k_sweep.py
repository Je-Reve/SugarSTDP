"""
Reward-delay k-sweep - THE "MONEY PLOT".

MANDATORY ordering (locked): only run this AFTER tau_e_sweep.py has
produced a winning tau_e, and set config.tau_e to that value first. Do
not sweep tau_e and k together/crossed.

Spec: with best tau_e fixed, sweep config.k over config.k_sweep_values =
{1, 5, 10, 20} environment timesteps, both systems, config.seeds (3 seeds)
each, at best lambda from lambda_sweep.py if already run, else lambda=0.
Prediction to check: System B degrades with increasing k (eligibility
trace decays too far before reward arrives), System A stays robust (BPTT
does not care about reward delay). This is the empirical validation of
the distal-reward theoretical argument - interpreting whether a given
result is a genuine credit-assignment failure or a tuning problem is a
judgment call, not something to resolve automatically.
"""

from config import Config


def run_k_sweep(config: Config, n_episodes: int) -> dict:
    """
    Loops over config.k_sweep_values, for each k runs both System A and
    System B across config.seeds via episode_loop.run_training (with
    config.tau_e already fixed to the tau_e_sweep.py winner), and records
    mean/std episode reward per (k, system). Returns whatever structure
    plotting.plot_k_sweep expects.
    """
    raise NotImplementedError(
        "TODO: implement the k-sweep loop (runs AFTER tau_e_sweep.py; this is the money plot)."
    )
