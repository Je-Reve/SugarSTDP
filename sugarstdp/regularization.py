"""
Activity-based regularization.

Locked decision: the regularizer targets ACTUAL EMITTED SPIKES, not weight
magnitude (no weight-L1 anywhere). Spike count maps directly onto real
hardware energy cost (see energy.py), which is the point of the whole
lambda axis.

The two systems apply the same physical penalty through two different
mechanisms - this asymmetry is intentional ("the regularization
wildcard"), not a bug to reconcile:

  System A (gradient descent on energy, differentiable):
      L_total = L_task + lambda * sum_{i,t} s_i(t)
  gradient of the spike-count term flows back through BPTT via the
  surrogate derivative, same as everything else in System A's loss.

  System B (reward shaping, biologically motivated, no backprop):
      r_effective = r_t - lambda * total_spikes_this_step
      delta_t     = r_effective + gamma * V(s_next) - V(s_current)
      dW_actor    = eta * delta_t * (action - pi) * e_ij   # inherits sparsity pressure
  i.e. lambda modifies the TD error's reward term BEFORE it multiplies the
  eligibility trace in system_b.py - it does not touch the eligibility
  trace computation itself (that stays in eligibility.py, untouched by lambda).
"""

import torch

from config import Config


def spike_count_penalty(spike_counts_per_layer: list, lam: float) -> torch.Tensor:
    """
    System A's regularization term: lambda * sum_{i,t} s_i(t), added
    directly into the total loss before calling .backward().

    Args:
        spike_counts_per_layer: list[torch.Tensor], each a scalar total
            spike count for one layer over the current episode/rollout
            (as returned by SpikingTrunk.forward).
        lam: current lambda value from config.lam.

    Returns:
        Scalar tensor to add to L_total. Must remain differentiable
        (do not .detach() or .item() the spike counts) so the surrogate
        gradient can flow back through it.
    """
    raise NotImplementedError("TODO: implement lambda * sum of spike counts.")


def apply_reward_shaping(r_t: float, total_spikes_this_step: float, lam: float) -> float:
    """
    System B's regularization mechanism: shape the raw reward before it
    enters the TD-error computation, per r_effective = r_t - lambda *
    total_spikes_this_step.

    Args:
        r_t: raw environment reward for this timestep (post k-delay
            handling, i.e. whatever env_wrapper.py already produced).
        total_spikes_this_step: total spikes emitted across all layers
            at this timestep.
        lam: current lambda value from config.lam.

    Returns:
        r_effective, to be used in place of r_t everywhere downstream in
        System B's TD-error computation (system_b.py). Does not touch the
        eligibility trace itself.
    """
    raise NotImplementedError("TODO: implement r_effective = r_t - lambda * total_spikes_this_step.")
