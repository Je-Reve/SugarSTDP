"""
Custom Leaky-Integrate-and-Fire neuron with TWO independent decay constants.

snnTorch's built-in `snn.Leaky` only exposes a single membrane decay ("beta").
This project's physics (Neftci et al. 2019) requires modeling the synaptic
current and the membrane potential as separately-decaying state variables:

    alpha = exp(-dt / tau_syn)   # synaptic current decay
    beta  = exp(-dt / tau_mem)   # membrane potential decay (snnTorch's "beta")

    I[t+1] = alpha * I[t] + sum_j W_ij * S_j[t]              # synaptic current
    V[t+1] = beta * V[t] + (1 - beta) * R * I[t] - S[t]      # membrane, soft reset via -S[t]
    S[t]   = spike_fn(V[t] - V_threshold)                     # spike (see note below - NOT a plain Heaviside call)

reset_mechanism is locked to 'subtract' (soft reset) for both System A and
System B - see config.reset_mechanism.

SURROGATE GRADIENT (important): a literal Heaviside step (e.g.
`(V - threshold >= 0).float()`) has zero gradient almost everywhere and
would silently break BPTT in System A - `.backward()` would produce
all-zero gradients for the whole network. Since we dropped snn.Leaky (it
can't do the two-decay-constant model above), we don't get snnTorch's
usual surrogate-gradient wiring for free, so `self.spike_fn` below supplies
it explicitly: `snntorch.surrogate.fast_sigmoid()` outputs real 0/1 spikes
on the forward pass (numerically identical to a true Heaviside) but
substitutes a smooth surrogate derivative on the backward pass. Call
`self.spike_fn(V - self.v_threshold)` in the forward implementation in
place of writing a hand-rolled comparison - this is standard snnTorch API
usage, not something to hand-roll.

This module is used identically by both System A (through BPTT + surrogate
gradient on the backward pass) and System B (forward pass only, no
backward through this module at all - R-STDP updates weights directly from
spike timing, not from autograd). Do NOT special-case system A/B in here -
`self.spike_fn` behaves correctly either way: under System B's
torch.no_grad() (see system_b.py select_action), no autograd graph gets
built at all, so the surrogate backward is simply never invoked; the
forward-pass spike values are identical regardless.
"""

from typing import Tuple

import torch
import torch.nn as nn
from snntorch import surrogate

from config import Config


class CustomLIFNeuron(nn.Module):
    """
    A layer of LIF neurons with independent synaptic (alpha) and membrane
    (beta) decay constants and a soft ('subtract') reset.

    `forward` implements the discretized update equations above. Verified
    against the closed-form solution for constant input current in
    tests/test_lif_physics.py BEFORE this is used anywhere in an RL loop.
    """

    def __init__(self, input_dim: int, n_neurons: int, config: Config, v_threshold: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.n_neurons = n_neurons
        self.config = config
        self.v_threshold = v_threshold
        # The synaptic weight matrix W_ij. Standard nn.Linear instantiation -
        # used inside the sum_j W_ij*S_j[t] term below.
        self.linear = nn.Linear(input_dim, n_neurons, bias=False)
        # Surrogate-gradient spike function (see module docstring for why this
        # exists), standard snnTorch API usage. Called as
        # self.spike_fn(V - self.v_threshold) at the right point in
        # forward(), in place of a hand-rolled Heaviside.
        self.spike_fn = surrogate.fast_sigmoid(slope=config.surrogate_slope)
        # alpha/beta come from config.alpha() / config.beta().
        # Stored as buffers (not learnable parameters - these are fixed time constants).
        self.register_buffer("alpha", torch.tensor(config.alpha))
        self.register_buffer("beta", torch.tensor(config.beta))


    def forward(
        self, input_spikes: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Advance the neuron layer by one SNN timestep.

        Args:
            input_spikes: presynaptic spikes, shape [batch, input_dim],
                values in {0, 1}. Apply self.linear(input_spikes) to get
                the sum_j W_ij * S_j[t] term - do not bypass self.linear.
            state: (I, V) from the previous timestep, each shape
                [batch, n_neurons].

        Returns:
            spikes: shape [batch, n_neurons], values in {0, 1}.
            new_state: updated (I, V) tuple.

        Implements the two-decay-constant update + soft reset described
        in the module docstring.
        """
        
        # synapse currents and membrane voltages from previous synapse spikes
        I_prev, V_prev = state
        
        # update V 
        V_new = self.beta * V_prev + (1-self.beta) *  I_prev
        
        # spike computation
        spikes = self.spike_fn(V_new - self.v_threshold)
        
        # pre-synapse V
        V_new -= spikes * self.v_threshold
        
        # update I
        I_new = self.alpha * I_prev + self.linear(input_spikes)
        
        return spikes, (I_new, V_new)

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero-initialize (I, V) at the start of an episode."""
        I_init = torch.zeros(batch_size, self.n_neurons).to(device)
        V_init = torch.zeros(batch_size, self.n_neurons).to(device)
        return (I_init, V_init)
        raise NotImplementedError("TODO: return zeroed (I, V) tensors of shape [batch, n_neurons].")
