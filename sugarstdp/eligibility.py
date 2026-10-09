"""
Event-gated eligibility trace - Izhikevich (2007) R-STDP formulation.

This is System B's actor-only mechanism (System A has no eligibility trace
at all). Explicitly NOT Bellec e-prop - no pseudo-derivative(V_post) term
anywhere in here. The Bellec connection is a citation for the writeup,
not something to implement.

The trace must be EVENT-GATED: e_ij only changes on genuine causal
pre/post spike pairs. It must NOT accumulate on every timestep regardless
of spiking - that would silently turn this into noise-accumulating
Hebbian learning and break the "tag not memory" property that makes
R-STDP interesting. Per-timestep behavior:

    e_ij *= exp(-dt / tau_e)                                   # decay, every timestep

    if pre fired at i and post fired at j with t_post > t_pre:  # causal order
        e_ij += A_plus * exp(-(t_post - t_pre) / tau_e)         # potentiate

    if post fired at j and pre fired at i with t_pre > t_post:  # anti-causal order
        e_ij -= A_minus * exp(-(t_pre - t_post) / tau_e)        # depress

Before wiring this into System B's training loop, unit-test it in
tests/test_eligibility_trace.py with synthetic spike pairs:
pre-then-post -> potentiation by the exact expected amount, post-then-pre
-> depression by the exact expected amount, unpaired spikes -> decay only,
no update.
"""

from typing import Optional

import torch
import torch.nn as nn

from config import Config


class EligibilityTrace(nn.Module):
    """
    Maintains e_ij for one weight matrix (pre_dim x post_dim) between an
    actor layer's presynaptic and postsynaptic populations.

    `decay`, `update`, and `reset` implement the event-gated STDP kernel
    described in the module docstring. tau_e is read from config
    (config.tau_e) - never hardcode it, it is swept before the k-sweep.
    """

    def __init__(self, pre_dim: int, post_dim: int, config: Config):
        super().__init__()
        self.pre_dim = pre_dim
        self.post_dim = post_dim
        self.config = config
        self.register_buffer("e", torch.zeros(pre_dim, post_dim))
        # Bookkeeping for "did i/j spike this step, and how long ago did the
        # other side last spike" - shape and dtype are an open choice; needed
        # to evaluate the event-gate conditions above.

    def reset(self) -> None:
        """Zero the trace (and any spike-timing bookkeeping) at episode start."""
        raise NotImplementedError("TODO: reset e_ij and timing bookkeeping to zero.")

    def update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, t: float) -> None:
        """
        Advance the trace by one environment timestep.

        Args:
            pre_spikes: shape [pre_dim], values in {0, 1} - presynaptic
                spikes at this timestep (z_pre in this module's notation).
            post_spikes: shape [post_dim], values in {0, 1} - postsynaptic
                spikes at this timestep (z_post).
            t: current simulation time in seconds (for computing
                t_post - t_pre spike-timing gaps).

        Decays e (every call), then event-gates the potentiation/depression
        terms exactly as in the module docstring.
        """
        raise NotImplementedError("TODO: implement the decay + event-gated STDP kernel.")

    def value(self) -> torch.Tensor:
        """Current e_ij tensor, shape [pre_dim, post_dim]."""
        return self.e
