"""
LIF physics verification.

No RL here at all - this is the "does my LIF match the physics" gate that
must pass before anything else. Two independent closed-form solutions to
verify (Neftci 2019's two decay constants), not one:

  1. Constant input current I_0 into the synaptic-current state: with
     alpha = exp(-dt/tau_syn), the discrete update I[t+1] = alpha*I[t] + I_0
     should match the known closed-form solution for a leaky integrator
     driven by a constant input (compare against the analytical steady
     state / trajectory, not just eyeballing a plot).
  2. Constant synaptic current into the membrane: with
     beta = exp(-dt/tau_mem), V[t+1] = beta*V[t] + (1-beta)*R*I[t] - S[t]
     (soft reset) should match the closed-form subthreshold membrane
     trajectory for a constant input current, before it ever reaches the
     spiking threshold.

Also verify here: deterministic rate encoding (encoding.py) maps
CartPole's observation range to a sane spike-rate range (e.g. spike
counts over T that make sense given the observation's actual bounds).

Intentionally NOT scaffolded further than signatures - deriving and
coding the closed-form check IS the exercise.
"""

import torch
import torch.nn.functional as F

from config import Config
from sugarstdp.lif import CustomLIFNeuron
from sugarstdp.encoding import encode_state



def test_synaptic_current_matches_closed_form():
    """Feed a constant input current; compare I[t] trajectory against the
    analytical solution for alpha = exp(-dt/tau_syn)."""

    # initialize input
    config = Config()
    
    
    a = config.alpha()
    I_const = 1.0
    I_curr = 0.0    
    for t in range(1, config.T+1):
        I_curr = a * I_curr + I_const 
        
        # Analytical derivation        
        I_analytical = I_const * (1 - a ** t) / (1-a)
    
        assert torch.isclose(
            torch.tensor(I_curr), 
            torch.tensor(I_analytical), 
            atol=1e-5
        )
def test_membrane_potential_matches_closed_form():
    """Feed a constant synaptic current (below threshold); compare V[t]
    trajectory against the analytical solution for beta = exp(-dt/tau_mem)."""
    
    
    config = Config()
    b = config.beta()
    
    
    
    # assume R is 1 and constant 
    I_const = 2.0               # arbitrary
    V_curr = 1.0       # arbitrary
    V_init = V_curr
    const_current = (1 - config.beta()) * I_const    # constant, R = 1 in assumptions
    
    for t in range(1, config.T+1):
        
        # beta * V[t] + (1-b) * I_const; the last term is constant
        V_curr = V_curr * b + const_current
        
        V_analytical  = b**t * V_init + I_const * (1-b**t)
        
        assert torch.isclose(
            torch.tensor(V_curr),
            torch.tensor(V_analytical),
            atol=1e-5
        )
        
def test_deterministic_rate_encoding_matches_observation_range():
    config = Config()
    T = config.T  # 20
    
    # Test 1: SHAPE 
    state = torch.zeros(4)
    spikes = encode_state(state, T, "rate", config)
    assert spikes.shape == (T, 1, 4), f"Expected [T,1,obs_dim], got {spikes.shape}"
    
    # Test 2: ZERO rate gives zero spikes
    # Feed min_obs exactly -> rate = 0.0 -> no spikes at all
    state_min = torch.Tensor(config.min_obs).clone()
    spikes_min = encode_state(state_min, T, "rate", config)
    assert spikes_min.sum() == 0, "min obs should produce zero spikes"

    # Test 3: MAX rate gives all spikes
    # Feed max_obs exactly -> rate = 1.0 -> spike every timestep
    state_max = torch.Tensor(config.max_obs).clone()
    spikes_max = encode_state(state_max, T, "rate", config)
    assert spikes_max.sum() == T * 4, "max obs should produce T spikes per feature"

    # Test 4: SPIKE COUNT matches expected rate
    # Mid-range input -> rate = 0.5 -> should produce exactly T//2 spikes per feature
    state_mid = (torch.tensor(config.min_obs) + torch.tensor(config.max_obs)) / 2.0
    spikes_mid = encode_state(state_mid, T, "rate", config)
    expected_count = torch.floor(torch.tensor(0.5) * T).long()
    for feat in range(4):
        count = spikes_mid[:, 0, feat].sum().long()
        assert count == expected_count, f"feat {feat}: got {count} spikes, expected {expected_count}"
    
    # Test 5: Confirm deterministic 
    state_rand = torch.tensor([0.5, -0.1, 0.2, -0.3])
    out1 = encode_state(state_rand, T, "rate", config)
    out2 = encode_state(state_rand, T, "rate", config)
    assert torch.equal(out1, out2), "encoding must be deterministic"
    
    # Test 6: Confirm binary 
    assert torch.all((spikes_mid == 0) | (spikes_mid == 1)), "spikes must be 0 or 1 only"
    
    
    
