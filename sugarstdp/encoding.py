"""
Spike encoding: maps a continuous CartPole observation to a spike train.

Locked decision: encoding must be DETERMINISTIC rate coding, not Poisson.
Poisson introduces per-step stochasticity that interferes with seed
control and makes eligibility-trace behavior noisier (harder to attribute
System B's results cleanly to the learning rule).

Must stay swappable via `encoding_type` so rate/Bernoulli/temporal
encodings can be added later without touching the episode loop or either
system's code.
"""

import torch

from config import Config


def encode_state(state: torch.Tensor, T: int, encoding_type: str, config: Config) -> torch.Tensor:
    """
    Encode a single CartPole observation into a spike train.

    Args:
        state: raw observation, shape [obs_dim] (cart position, cart velocity,
            pole angle, pole angular velocity).
        T: number of SNN timesteps to encode over (config.T).
        encoding_type: currently only "rate" is locked-in and required;
            keep the branch structure open for future encodings.
        config: full run config, in case bounds/scaling need other fields.

    Returns:
        Spike tensor of shape [T, batch, obs_dim], dtype matching what the LIF
        layer expects (0/1 spikes).

    For "rate" encoding, maps each of the 4 unbounded/bounded CartPole
    floats to a firing rate in [0, 1], then generates a DETERMINISTIC
    (not Poisson-sampled) spike train at that rate over T steps - i.e.
    evenly-spaced spikes rather than randomly-sampled ones. Verified
    against known CartPole observation ranges before it ever touches RL.
    """
    
    # deterministic 
    if encoding_type == "rate":

        # normalize
        min_obs = torch.Tensor(config.min_obs)
        max_obs = torch.Tensor(config.max_obs)
        
        rates = (state - min_obs) / (max_obs - min_obs + 1e-8)  # shape [obs_dim], ∈ [0,1]
        rates = rates.clamp(0.0, 1.0)

        # Deterministic evenly-spaced spike placement using floor difference:
        # spike[t, feat] = floor((t+1) * rate[feat]) - floor(t * rate[feat])
        # This places exactly floor(rate * T) spikes, evenly distributed over T steps.
        timesteps = torch.arange(T, dtype=torch.float32)          # shape [T]
        spikes = (
            torch.floor((timesteps[:, None] + 1) * rates[None, :])   # [T, obs_dim]
            - torch.floor(timesteps[:, None]      * rates[None, :])   # [T, obs_dim]
        )                                                              # values are 0.0 or 1.0

        return spikes.unsqueeze(1)  # [T, 1, obs_dim]
    
    raise NotImplementedError(f"Encoding {encoding_type} not yet supported.")
