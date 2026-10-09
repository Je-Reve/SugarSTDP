"""
Plotting code.

Left as STUBS: real plot design depends on what the actual data looks
like, which doesn't exist yet. To be filled in once a given sweep has real
CSVs to plot.

Expected inputs are pandas DataFrames built from the CSV files logger.py
writes to `logs/` (one row per episode; columns: episode, episode_reward,
total_spikes, eligibility_trace_mean, eligibility_trace_max, weight_norm.*).
"""

from typing import Optional

import pandas as pd


def plot_reward_curves(runs: dict, save_path: Optional[str] = None):
    """
    Reward-per-episode curves, mean +/- std across seeds, one line per
    system (A vs B). `runs` maps a label (e.g. "System A") to a list of
    per-seed episode-reward DataFrames/arrays.
    """
    raise NotImplementedError("TODO: reward curve plot once real training CSVs exist.")


def plot_eligibility_trace_raster(trace_log: pd.DataFrame, save_path: Optional[str] = None):
    """
    Visualize e_ij(t) dynamics for a single System B run: trace growth on
    spike pairs, exponential decay between events, collapse at large k.
    """
    raise NotImplementedError("TODO: eligibility trace raster plot once System B logs traces.")


def plot_k_sweep(results_by_system_and_k: dict, save_path: Optional[str] = None):
    """
    The "money plot": mean episode reward (or reward-degradation) vs. k,
    one line per system, error bars from seeds.
    """
    raise NotImplementedError("TODO: k-sweep plot once tau_e is fixed and k-sweep CSVs exist.")


def plot_tau_e_sweep(results_by_tau_e: dict, save_path: Optional[str] = None):
    """Mean System B reward vs. tau_e at k=1, to pick the winning tau_e before the k-sweep."""
    raise NotImplementedError("TODO: tau_e sweep plot once CSVs exist.")


def plot_pareto_frontier(results_by_system_and_lambda: dict, save_path: Optional[str] = None):
    """
    Task Reward vs. estimated joules-emulated (energy.py), across lambda
    values, both systems.
    """
    raise NotImplementedError("TODO: Pareto frontier plot once lambda-sweep CSVs and energy estimates exist.")
