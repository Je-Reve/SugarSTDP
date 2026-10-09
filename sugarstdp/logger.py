"""
Logging / checkpointing infrastructure.

Wired in from the start, not retrofitted later. Logging spec:
  - Episode reward per episode, per seed
  - Spike counts per layer per episode (energy estimation + regularization monitoring)
  - Eligibility trace mean and max per episode (System B only)
  - Weight norm per layer per episode (exploding/vanishing sanity check)

Storage: flat CSV per (system, k, lambda, seed) run under `logs/`, plus a
JSON run-config sidecar so plotting.py / sweep aggregation can reconstruct
which run produced which file without parsing filenames.

Also provides `find_solved_episode` - System A's performance gate (mean
episode reward >= 195 over 100 consecutive episodes, checkable in code via
a sliding window over the reward log) made concrete. Checked and reported,
but training never stops early because of it (locked stopping criterion).
"""

import csv
import json
import os
from dataclasses import asdict
from typing import Dict, List, Optional

import torch

from config import Config


class ExperimentLogger:
    """One instance per training run (system, k, lambda, seed)."""

    def __init__(self, run_name: str, config: Config, log_dir: str = "logs"):
        self.run_name = run_name
        self.config = config
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.csv_path = os.path.join(log_dir, f"{run_name}.csv")
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer: Optional[csv.DictWriter] = None

        self._step_spike_counts: List[float] = []
        self._step_metrics: List[Dict[str, float]] = []

        with open(os.path.join(log_dir, f"{run_name}.config.json"), "w") as f:
            json.dump(asdict(config), f, indent=2)

    def log_step(self, episode_idx: int, spike_counts_per_layer: list, metrics: Dict[str, float]) -> None:
        """Buffer per-timestep spike counts and system.update() metrics;
        flushed into the per-episode row by log_episode()."""
        total_spikes = sum(float(c.detach().item()) if isinstance(c, torch.Tensor) else float(c)
                            for c in spike_counts_per_layer)
        self._step_spike_counts.append(total_spikes)
        self._step_metrics.append(metrics)

    def log_episode(self, episode_idx: int, episode_reward: float, weight_norms: Dict[str, float]) -> None:
        """Aggregate this episode's buffered step data into one CSV row."""
        total_spikes = sum(self._step_spike_counts)

        trace_means = [m["trace_mean"] for m in self._step_metrics if "trace_mean" in m]
        trace_maxes = [m["trace_max"] for m in self._step_metrics if "trace_max" in m]

        row = {
            "episode": episode_idx,
            "episode_reward": episode_reward,
            "total_spikes": total_spikes,
            "eligibility_trace_mean": sum(trace_means) / len(trace_means) if trace_means else None,
            "eligibility_trace_max": max(trace_maxes) if trace_maxes else None,
        }
        row.update({f"weight_norm.{name}": val for name, val in weight_norms.items()})

        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        self._step_spike_counts = []
        self._step_metrics = []

    def close(self) -> None:
        self._csv_file.close()

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def find_solved_episode(episode_rewards: List[float], threshold: float, window: int) -> Optional[int]:
    """
    First episode index (0-based) at which the trailing `window`-episode
    mean reward reaches `threshold`, or None if it never does.

    Locked: threshold=195.0, window=100 for System A's "solved" gate -
    pass config.solve_reward_threshold / config.solve_window. This only
    REPORTS the gate; it must never be used to cut a run short (locked
    stopping criterion: full episode budget always, no early stopping for
    either system).
    """
    for i in range(window - 1, len(episode_rewards)):
        window_mean = sum(episode_rewards[i - window + 1 : i + 1]) / window
        if window_mean >= threshold:
            return i
    return None
