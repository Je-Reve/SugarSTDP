"""
Shared episode loop, parameterized by `system` so System A and System B
reuse the exact same loop. Learning rule is a swappable module - the
episode loop is parameterized by system ('A' or 'B'), not duplicated.

This module only calls the ActorCriticSystem interface (systems_base.py) -
it never branches on which concrete system it was given. It will raise
NotImplementedError as soon as it calls into system.select_action /
system.update until SystemA/SystemB are implemented, and as soon as it
calls encoding.encode_state until encoding.py is implemented.

Seed strategy (mandatory from the start, not retrofitted):
torch.manual_seed(seed) + np.random.seed(seed) + env.reset(seed=seed) are
all set together by `run_training`, once per seed, before any episodes run.

Stopping criterion (locked): full episode budget always, NO early stopping
for either system, even if the performance gate is hit early - early
termination would give System A an unfair compute advantage over System B
and invalidate the comparison. The performance gate (mean reward >= 195
over 100 consecutive episodes) is checked and logged, but training does
not stop because of it.
"""

import random
from typing import Optional

import numpy as np
import torch

from config import Config
from sugarstdp.env_wrapper import make_env
from sugarstdp.logger import ExperimentLogger
from sugarstdp.systems_base import ActorCriticSystem, Transition


def set_seed(seed: int) -> None:
    """torch + numpy + python random, per the module's seed strategy."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_episode(
    env,
    system: ActorCriticSystem,
    config: Config,
    logger: Optional[ExperimentLogger] = None,
    episode_idx: int = 0,
) -> float:
    """
    Run one full episode with `system` (either SystemA or SystemB -
    identical call sites either way) acting in `env`, updating online at
    every environment timestep. Returns total (undelayed-equivalent)
    episode reward.
    """
    system.reset_state()
    obs, _ = env.reset()
    done = False
    episode_reward = 0.0

    while not done:
        state = torch.as_tensor(obs, dtype=torch.float32)
        # Encoding (encoding.encode_state) happens INSIDE system.select_action,
        # not here - keeps episode_loop.py encoding-scheme-agnostic.
        action, policy_probs, value, spike_counts_per_layer = system.select_action(state)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = torch.as_tensor(next_obs, dtype=torch.float32)

        # V(s_next) is intentionally NOT computed here - system.update()
        # computes it from transition.next_state itself (see systems_base.py:
        # Transition docstring for why: done-masking is a system-level detail).
        transition = Transition(
            state=state,
            action=action,
            policy_probs=policy_probs,
            reward=reward,
            next_state=next_state,
            done=done,
            value=value,
            spike_counts_per_layer=spike_counts_per_layer,
        )
        metrics = system.update(transition)

        episode_reward += reward
        obs = next_obs

        if logger is not None:
            logger.log_step(episode_idx, spike_counts_per_layer, metrics)

    if logger is not None:
        logger.log_episode(episode_idx, episode_reward, system.get_weight_norms())

    return episode_reward


def run_training(
    system_builder,
    config: Config,
    seed: int,
    n_episodes: int,
    k: int,
    lam: float,
    logger: Optional[ExperimentLogger] = None,
) -> list:
    """
    Full training run for one (system, seed, k, lambda) configuration.

    Args:
        system_builder: callable(config) -> ActorCriticSystem, i.e.
            `lambda cfg: SystemA(cfg)` or `lambda cfg: SystemB(cfg)`. Passed
            in rather than imported directly so this module never branches
            on 'A' vs 'B'.
        config: base config; k and lam are applied onto a copy so callers
            can sweep without mutating shared state.
        seed: applied via set_seed() and env.reset(seed=seed) together.
        n_episodes: full budget - always run to completion, no early stop.
        k, lam: current sweep values, always passed from config, never
            hardcoded anywhere in the loop.

    Returns:
        List of per-episode rewards, length n_episodes.
    """
    set_seed(seed)
    run_config = config
    run_config.k = k
    run_config.lam = lam

    env = make_env(k=k, seed=seed)
    system = system_builder(run_config)

    rewards = []
    for episode_idx in range(n_episodes):
        episode_reward = run_episode(env, system, run_config, logger=logger, episode_idx=episode_idx)
        rewards.append(episode_reward)

    return rewards
