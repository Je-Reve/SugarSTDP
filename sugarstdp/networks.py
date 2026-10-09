"""
Shared network wiring + actor/critic heads.

Architecture (exact sizes TBD pending a joint sizing decision): a stack of
`config.n_lif_layers` CustomLIFNeuron layers, topped by a plain nn.Linear
head. Identical neuron counts, identical layer structure, identical T -
System A and System B both instantiate this same class for both their
actor and critic; only their *learning rules* (how gradients/weight
updates are computed and applied, in system_a.py / system_b.py) differ.

The critic is a SEPARATE, PLAIN (non-spiking) MLP - not a second
SpikingTrunk. The experimental question is entirely about the ACTOR's
learning rule; the critic is just a shared value-function baseline used to
compute the TD error both systems' actors consume. Giving it its own
LIF/encoding machinery would add complexity that has nothing to do with
the hypothesis under test, and would force System B's critic (backprop)
and actor (manual R-STDP, no autograd at all) to look like they're the
same kind of thing when they're actually two fully decoupled learning
processes that happen to both read the same environment. The `Critic`
class below is instantiated identically by both systems so the critic
code path is genuinely shared, not just conceptually equivalent - this is
the concrete enforcement of "critic is identical in both systems."

The per-timestep loop below only calls into `CustomLIFNeuron.forward`
(lif.py) and does not itself implement any neuron dynamics - it is
infrastructure wiring, not conceptual content. Nothing here will run
until lif.py is filled in.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from config import Config
from sugarstdp.lif import CustomLIFNeuron


class SpikingTrunk(nn.Module):
    """Stack of LIF layers shared by the actor and critic heads."""

    def __init__(self, config: Config):
        super().__init__()
        if config.hidden_size is None or config.n_lif_layers is None:
            raise ValueError(
                "config.hidden_size / config.n_lif_layers are still TBD - "
                "pin these before instantiating a network."
            )
        self.config = config
        self.hidden_size: int = config.hidden_size
        dims = [config.obs_dim] + [config.hidden_size] * config.n_lif_layers
        self.layers = nn.ModuleList(
            [CustomLIFNeuron(dims[i], dims[i + 1], config) for i in range(config.n_lif_layers)]
        )

    def init_state(self, batch_size: int, device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [layer.init_state(batch_size, device) for layer in self.layers]

    def forward(
        self, spike_input: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run the full T-timestep spike train through all LIF layers.

        Args:
            spike_input: shape [T, batch, obs_dim] - output of
                `encoding.encode_state`, batched.

        Returns:
            last_layer_spikes: shape [T, batch, hidden_size] - spikes
                emitted by the final LIF layer at every timestep (actor and
                critic heads read from this).
            spike_counts_per_layer: list of length n_lif_layers, each a
                scalar tensor = total spikes emitted by that layer summed
                over T and batch (needed for regularization.py and
                energy.py - do not recompute spike counting elsewhere).
        """
        T, batch_size, _ = spike_input.shape
        device = spike_input.device
        state = self.init_state(batch_size, device)
        spike_counts_per_layer = [torch.zeros((), device=device) for _ in self.layers]

        outputs = []
        layer_input = spike_input
        all_layer_spikes = [[] for _ in self.layers]
        # SNN SIMULATION TIME STEP
        for t in range(T):
            # multiple layer input
            x = layer_input[t]
            for i, layer in enumerate(self.layers):
                spikes, state[i] = layer(x, state[i])
                spike_counts_per_layer[i] = spike_counts_per_layer[i] + spikes.sum()
                all_layer_spikes[i].append(spikes)
                x = spikes
            outputs.append(x)

        last_layer_spikes = torch.stack(outputs, dim=0)  # [T, batch, hidden_size]
        return last_layer_spikes, spike_counts_per_layer


class ActorHead(nn.Module):
    """Linear readout -> action logits. Standard nn.Linear."""

    def __init__(self, hidden_size: int, n_actions: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, n_actions)

    def forward(self, trunk_spikes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trunk_spikes: [T, batch, hidden_size] from SpikingTrunk.
        Returns:
            action logits, [batch, n_actions], read out from the
            time-averaged trunk activity (rate-coded readout).
        """
        rate = trunk_spikes.mean(dim=0)  # [batch, hidden_size]
        return self.linear(rate)


class ActorNetwork(nn.Module):
    """Own SpikingTrunk + ActorHead. Instantiated by both System A and
    System B with identical architecture; only how it gets trained differs."""

    def __init__(self, config: Config):
        super().__init__()
        self.trunk = SpikingTrunk(config)
        self.head = ActorHead(self.trunk.hidden_size, config.n_actions)

    def forward(self, spike_input: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            spike_input: [T, batch, obs_dim].
        Returns:
            action_logits: [batch, n_actions].
            spike_counts_per_layer: list[torch.Tensor], for regularization.py / energy.py.
        """
        trunk_spikes, spike_counts_per_layer = self.trunk(spike_input)
        return self.head(trunk_spikes), spike_counts_per_layer


class Critic(nn.Module):
    """
    Plain (non-spiking) MLP baseline: V(s) directly from the raw
    observation, no spike encoding, no T-timestep loop. Standard ANN work -
    actor/critic head linear layers (nn.Linear).

    Trained via ordinary backprop MSE on TD targets in BOTH systems
    (locked) - instantiate this exact class in system_a.py and system_b.py
    rather than duplicating it, so "critic is identical" holds at the code
    level, not just conceptually.
    """

    def __init__(self, config: Config, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: raw observation, shape [batch, obs_dim] (or [obs_dim],
                will be unsqueezed by the caller as needed).
        Returns:
            V(s), shape [batch, 1].
        """
        return self.net(state)
