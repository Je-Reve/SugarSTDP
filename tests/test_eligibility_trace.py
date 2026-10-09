"""
Unit tests for the eligibility trace.

Unit-test the R-STDP eligibility trace in isolation, with synthetic spike
pairs, BEFORE it is ever entangled with the RL loop. Getting this wrong -
e.g. accumulating on every timestep - produces incorrect results that are
hard to debug later. Four scenarios to cover:

  1. Pure decay: no spikes at all -> e_ij decays by exactly
     exp(-dt/tau_e) each step, no other change.
  2. Pre-then-post (causal order): feed a synthetic pre->post pair ->
     confirm e_ij increments by exactly A_plus * exp(-(t_post - t_pre)/tau_e).
  3. Post-then-pre (anti-causal order): feed a synthetic post->pre pair ->
     confirm e_ij decrements by exactly A_minus * exp(-(t_pre - t_post)/tau_e).
  4. Unpaired/random spikes: feed spikes that don't form a genuine causal
     pair -> confirm e_ij does NOT update from the STDP kernel (decay only).

Also verify the weight update dW = eta * delta_t * (action_onehot -
policy_probs) * e scales correctly and independently with each of its
three factors (eta, delta_t, the score-function term, and e).

Intentionally NOT scaffolded further than signatures - writing these
tests IS how the event-gate logic gets internalized.
"""

import torch

from config import Config
from sugarstdp.eligibility import EligibilityTrace


def test_trace_decays_with_no_spikes():
    raise NotImplementedError("TODO: verify pure exponential decay with no spikes.")


def test_trace_potentiates_on_causal_pre_then_post_pair():
    raise NotImplementedError("TODO: verify e_ij += A_plus * exp(-(t_post-t_pre)/tau_e).")


def test_trace_depresses_on_anticausal_post_then_pre_pair():
    raise NotImplementedError("TODO: verify e_ij -= A_minus * exp(-(t_pre-t_post)/tau_e).")


def test_trace_does_not_update_on_unpaired_spikes():
    raise NotImplementedError("TODO: verify unpaired spikes cause decay only, no STDP-kernel update.")


def test_weight_update_scales_with_each_factor_independently():
    """dW = eta * delta_t * (action_onehot - policy_probs) * e - vary each
    factor independently and confirm dW scales linearly with it."""
    raise NotImplementedError("TODO: verify dW scales correctly with eta, delta_t, the score-function term, and e.")
