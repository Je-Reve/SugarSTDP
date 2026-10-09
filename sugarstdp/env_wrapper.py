"""
Gymnasium environment wrapper implementing the reward-delay k-sweep
variant, plus seeding.

PRECISE k-delay definition (implementation-critical): k is measured in
ENVIRONMENT TIMESTEPS, not episodes. Reward accumulates at +1 per timestep
survived, is held, and delivered as a lump sum at the k-th, 2k-th, 3k-th
... environment timestep within each episode. The within-episode timestep
counter resets at every episode boundary.

DEFAULT for what constitutes a "fair" reward-delay variant design: it is
not fully specified what happens to a trailing partial window when an
episode ends before its next k-th-timestep boundary (e.g. k=20, episode
ends at timestep 47 -> 7 timesteps of accumulated reward would never hit
a multiple of 20). This wrapper's default is to flush any pending reward
on episode end (terminated or truncated) so total episode reward always
equals the undelayed CartPole total (only the TIMING of delivery changes,
not the total) - this keeps the "solved" reward threshold (>=195)
comparable across k values. To make un-flushed trailing reward simply be
lost instead, change the `done` branch below.
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class DelayedRewardCartPole(gym.Wrapper):
    """CartPole-v1 with reward held and delivered every k environment timesteps."""

    def __init__(self, k: int, render_mode: Optional[str] = None):
        env = gym.make("CartPole-v1", render_mode=render_mode)
        super().__init__(env)
        if k < 1:
            raise ValueError(f"k must be >= 1 environment timesteps, got {k}")
        self.k = k
        self._pending_reward = 0.0
        self._timestep_in_episode = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._pending_reward = 0.0
        self._timestep_in_episode = 0
        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, raw_reward, terminated, truncated, info = self.env.step(action)
        self._timestep_in_episode += 1
        self._pending_reward += raw_reward
        done = terminated or truncated

        at_k_boundary = (self._timestep_in_episode % self.k) == 0
        if at_k_boundary or done:
            delivered_reward = self._pending_reward
            self._pending_reward = 0.0
        else:
            delivered_reward = 0.0

        return obs, delivered_reward, terminated, truncated, info


def make_env(k: int, seed: int, render_mode: Optional[str] = None) -> DelayedRewardCartPole:
    """
    Construct a seeded DelayedRewardCartPole. Seed must be set at
    construction/first-reset time (torch.manual_seed + np.random.seed
    happen elsewhere, in episode_loop.py - this only handles the env's
    own seed).
    """
    env = DelayedRewardCartPole(k=k, render_mode=render_mode)
    env.reset(seed=seed)
    return env
