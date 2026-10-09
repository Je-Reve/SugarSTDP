"""
System B - Reward-Modulated STDP Actor-Critic (experimental).

The eligibility-trace integration and the R-STDP weight update itself:
no backprop, no BPTT in the actor - fully online, local update rule only.

Critic is `networks.Critic` - the exact same plain non-spiking MLP class
System A uses, reading the raw state directly (no encoding, no spikes) and
trained identically via backprop MSE on TD targets (locked). It is a
fully separate, decoupled learning process from the actor's R-STDP update
below - they share nothing but the environment.

The actor runs its forward pass under torch.no_grad() - there is no
autograd graph to build for R-STDP, weight updates come directly from
eligibility traces and TD error, not from .backward().

OPEN DESIGN QUESTION (joint, do not resolve unilaterally): the
discrete-action R-STDP formula -

    dW_actor = eta * delta_t * (action_onehot - policy_probs) * e_ji

- is written for a single actor weight matrix feeding the action
readout. If config.n_lif_layers > 1, how the (action_onehot - policy_probs)
learning signal reaches eligibility traces on EARLIER hidden layers is not
yet specified and needs a decision before it's relied on - e.g.
restricting R-STDP to the final trunk-layer -> action-head weights and
leaving earlier layers fixed, vs. some broadcast/feedback scheme. Do not
silently invent an answer here.
"""

from typing import Dict, List

import torch
import torch.optim as optim

from config import Config
from sugarstdp.eligibility import EligibilityTrace
from sugarstdp.encoding import encode_state
from sugarstdp.networks import ActorNetwork, Critic
from sugarstdp.regularization import apply_reward_shaping
from sugarstdp.systems_base import ActorCriticSystem, Transition


class SystemB(ActorCriticSystem):
    def __init__(self, config: Config):
        self.config = config
        self.actor = ActorNetwork(config)
        self.critic = Critic(config)
        if config.eta_B is None:
            raise ValueError("config.eta_B is still TBD - tune with equal budget to eta_A.")
        # The hyperparameter table only defines eta_A (System A actor) and
        # eta_B (System B actor) - no separate critic LR. Since the critic is
        # identical in both systems and System A already trains its critic
        # jointly with its actor at eta_A, System B's standalone critic
        # reuses eta_A here for consistency. Revisit if a dedicated critic LR
        # turns out to matter during tuning.
        if config.eta_A is None:
            raise ValueError("config.eta_A is still TBD - also used as System B's critic LR (see comment above).")
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.eta_A)
        self.eligibility_traces: List[EligibilityTrace] = self._build_eligibility_traces(config)

    def _build_eligibility_traces(self, config: Config) -> List[EligibilityTrace]:
        """One EligibilityTrace per actor weight matrix (each trunk LIF
        layer's linear weight, plus the actor head's linear weight). See
        the open design question in the module docstring re: multi-layer
        credit assignment."""
        traces = []
        for layer in self.actor.trunk.layers:
            traces.append(EligibilityTrace(layer.linear.in_features, layer.linear.out_features, config))
        traces.append(EligibilityTrace(self.actor.head.linear.in_features, self.actor.head.linear.out_features, config))
        return traces

    def reset_state(self) -> None:
        """Reset actor LIF state (handled fresh per SpikingTrunk.forward call)
        AND eligibility traces - traces do NOT persist across episodes."""
        for trace in self.eligibility_traces:
            trace.reset()

    def select_action(self, state: torch.Tensor):
        """
        Encode state -> spike train -> self.actor (no_grad) -> sample
        action; self.critic(state) -> V(s) (raw state, grad-tracked, for
        backprop training).

        This is standard infrastructure wiring - it will raise
        NotImplementedError from inside encode_state / the LIF forward pass
        until encoding.py and lif.py are implemented. That is expected.
        """
        with torch.no_grad():
            spike_train = encode_state(state, self.config.T, self.config.encoding_type, self.config)
            spike_train = spike_train.unsqueeze(1)  # [T, obs_dim] -> [T, batch=1, obs_dim]

            action_logits, spike_counts_per_layer = self.actor(spike_train)
            policy_probs = torch.softmax(action_logits, dim=-1).squeeze(0)  # [n_actions]
            action = torch.distributions.Categorical(probs=policy_probs).sample()

        value = self.critic(state.unsqueeze(0)).squeeze()  # V(s), scalar - grad-tracked, NOT under no_grad

        return int(action.item()), policy_probs, value, spike_counts_per_layer

    def update(self, transition: Transition) -> Dict[str, float]:
        """
        Per the R-STDP weight update:

            r_effective = apply_reward_shaping(transition.reward, total_spikes_this_step, config.lam)
            delta_t     = r_effective + gamma * V(s_next) - V(s_current)   # critic values, .detach()'d for the actor side
            L_j         = action_onehot - policy_probs                     # REINFORCE score function, NOT optional
            dW_actor    = eta_B * delta_t * L_j * e_ji                     # per eligibility_traces[i]

        Also: update self.eligibility_traces (decay + event-gated STDP kernel,
        via trace.update(pre_spikes, post_spikes, t)) BEFORE computing dW_actor
        for this step - e_ji at the moment of the update must reflect this
        step's spikes.

        Critic update is standard backprop MSE (identical to SystemA's critic):
        self.critic_optimizer.zero_grad(); L_critic.backward(); self.critic_optimizer.step().

        Return a metrics dict (td_error, trace_mean, trace_max per layer) for logger.py.
        """
        raise NotImplementedError(
            "TODO: integrate the eligibility trace update and the R-STDP weight update."
        )

    def get_weight_norms(self) -> Dict[str, float]:
        norms = {}
        for i, layer in enumerate(self.actor.trunk.layers):
            norms[f"actor_trunk_layer_{i}"] = layer.linear.weight.norm().item()
        norms["actor_head"] = self.actor.head.linear.weight.norm().item()
        for i, layer in enumerate(self.critic.net):
            if isinstance(layer, torch.nn.Linear):
                norms[f"critic_layer_{i}"] = layer.weight.norm().item()
        return norms
