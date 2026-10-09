"""
Common interface shared by System A and System B so that episode_loop.py
can drive either one without duplicating the loop. Learning rule is a
swappable module - the episode loop is parameterized by system ('A' or
'B'), not duplicated.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class Transition:
    """One environment timestep's worth of data, passed to system.update()."""

    state: torch.Tensor          # raw observation, shape [obs_dim]
    action: int
    policy_probs: torch.Tensor   # shape [n_actions], from select_action
    reward: float                # raw env reward for this timestep (before any k-delay holding or lambda shaping)
    next_state: torch.Tensor
    done: bool
    value: torch.Tensor          # V(s_t), scalar - from select_action's critic call
    spike_counts_per_layer: list  # list[torch.Tensor], from this timestep's forward pass

    # No next_value field on purpose: V(s_{t+1}) must be computed by
    # system.update() itself (via self.critic(next_state)) rather than
    # precomputed here, since done-masking (V(s_next)=0 when done) is a
    # system-level correctness detail, not episode_loop's to get right.


class ActorCriticSystem(ABC):
    """
    Common interface implemented by SystemA and SystemB. episode_loop.py
    only talks to this interface - it never branches on 'A' vs 'B'.
    """

    @abstractmethod
    def reset_state(self) -> None:
        """Reset LIF membrane/synaptic state (and, for System B, eligibility
        traces) at the start of an episode."""
        raise NotImplementedError

    @abstractmethod
    def select_action(self, state: torch.Tensor) -> tuple:
        """
        Args:
            state: raw observation, shape [obs_dim].
        Returns:
            (action: int, policy_probs: torch.Tensor, value: torch.Tensor,
             spike_counts_per_layer: list[torch.Tensor])
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, transition: Transition) -> Dict[str, float]:
        """
        Perform one learning update from a single Transition (System A:
        accumulate/step through BPTT + surrogate gradient; System B:
        update eligibility traces and apply the R-STDP weight update).

        Returns:
            metrics dict for logger.py, e.g. {"td_error": ..., "loss": ...}
            (System A) or {"td_error": ..., "trace_mean": ..., "trace_max": ...}
            (System B).
        """
        raise NotImplementedError

    @abstractmethod
    def get_weight_norms(self) -> Dict[str, float]:
        """Per-layer weight norms, for the exploding/vanishing sanity check
        logged every episode."""
        raise NotImplementedError
