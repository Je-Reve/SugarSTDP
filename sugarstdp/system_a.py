"""
System A - Surrogate-Gradient Actor-Critic (baseline / control).

The loss construction inside `update()` - actor log-prob term, critic MSE
term, combining them, and adding the lambda*spike-count penalty - is
standard backprop, MSE loss on TD targets, combined actor+critic loss,
plus the regularization line.

Uses snnTorch's surrogate-gradient machinery under the hood: real spikes
(Heaviside) on the forward pass through SpikingTrunk/CustomLIFNeuron,
smooth surrogate derivative on the backward pass. gamma_pd (config.gamma_pd)
is the pseudo-derivative scale - System A only, never used in System B.

The critic is a plain non-spiking MLP over the raw state (see networks.py
docstring for why - the experimental question is entirely about the
actor's learning rule). Both actor and critic are trained end-to-end via
backprop here, so nothing stops a single combined loss/optimizer step
across both - unlike System B, where the critic's backprop and the
actor's R-STDP update must stay decoupled.
"""

from typing import Dict

import torch
import torch.optim as optim

from config import Config
from sugarstdp.encoding import encode_state
from sugarstdp.networks import ActorNetwork, Critic
from sugarstdp.regularization import spike_count_penalty
from sugarstdp.systems_base import ActorCriticSystem, Transition


class SystemA(ActorCriticSystem):
    def __init__(self, config: Config):
        self.config = config
        self.actor = ActorNetwork(config)
        self.critic = Critic(config)
        if config.eta_A is None:
            raise ValueError("config.eta_A is still TBD - tune with equal budget to eta_B.")
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = optim.Adam(params, lr=config.eta_A)

    def reset_state(self) -> None:
        """No persistent state to reset between episodes beyond what
        SpikingTrunk.forward already re-initializes each call (fresh
        (I, V) state per forward pass)."""
        pass

    def select_action(self, state: torch.Tensor):
        """
        Encode state -> spike train -> self.actor(spikes) -> sample an
        action from the softmax policy; self.critic(state) reads the RAW
        (unencoded) state directly - no spike train involved.

        This is standard infrastructure wiring - it will raise
        NotImplementedError from inside encode_state / the LIF forward pass
        until encoding.py and lif.py are implemented. That is expected.
        """
        spike_train = encode_state(state, self.config.T, self.config.encoding_type, self.config)
        spike_train = spike_train.unsqueeze(1)  # [T, obs_dim] -> [T, batch=1, obs_dim]

        action_logits, spike_counts_per_layer = self.actor(spike_train)
        policy_probs = torch.softmax(action_logits, dim=-1).squeeze(0)  # [n_actions]
        action = torch.distributions.Categorical(probs=policy_probs).sample()

        value = self.critic(state.unsqueeze(0)).squeeze()  # V(s), scalar

        return int(action.item()), policy_probs, value, spike_counts_per_layer

    def update(self, transition: Transition) -> Dict[str, float]:
        """
        Constructs L_total for this transition/rollout:

            L_actor  = -log(policy_probs[action]) * delta_t.detach()   # or full-episode return
            L_critic = (V(s_t) - (r_t + gamma * V(s_{t+1})))**2        # self.critic(state) / self.critic(next_state), raw states
            L_total  = L_actor + L_critic + spike_count_penalty(spike_counts_per_layer, config.lam)

        then self.optimizer.zero_grad(); L_total.backward(); self.optimizer.step().

        Return a metrics dict (loss components, td_error) for logger.py.
        """
        raise NotImplementedError("TODO: construct the combined actor+critic loss and backprop.")

    def get_weight_norms(self) -> Dict[str, float]:
        norms = {}
        for i, layer in enumerate(self.actor.trunk.layers):
            norms[f"actor_trunk_layer_{i}"] = layer.linear.weight.norm().item()
        norms["actor_head"] = self.actor.head.linear.weight.norm().item()
        for i, layer in enumerate(self.critic.net):
            if isinstance(layer, torch.nn.Linear):
                norms[f"critic_layer_{i}"] = layer.weight.norm().item()
        return norms
