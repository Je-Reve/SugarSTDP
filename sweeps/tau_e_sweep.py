"""
tau_e sweep.

MANDATORY ordering (locked): this sweep MUST run and be resolved BEFORE
k_sweep.py. Reversing the order confounds k-sweep results with the tau_e
choice and invalidates the "money plot."

Spec: sweep config.tau_e over config.tau_e_sweep_values = {200ms, 500ms,
1s, 2s, 5s}, at k=1, both systems, config.seeds (3 seeds) each. Winning
tau_e = value maximizing mean episode reward for System B across seeds.
Lock that value into config.tau_e before running k_sweep.py.
"""

from config import Config


def run_tau_e_sweep(config: Config, n_episodes: int) -> dict:
    """
    Loops over config.tau_e_sweep_values, for each value runs both
    System A and System B (System A is tau_e-invariant but included for a
    sanity-check baseline) across config.seeds via episode_loop.run_training,
    and records mean/std episode reward per (tau_e, system). Returns
    whatever structure plotting.plot_tau_e_sweep expects, and prints/saves
    the winning tau_e for System B.
    """
    raise NotImplementedError("TODO: implement the tau_e sweep loop (runs BEFORE k_sweep.py).")
