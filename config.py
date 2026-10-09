"""
Central configuration for the SugarSTDP project.

Single source for every hyperparameter. Lambda and k must always
be read from here (or from a Config instance built from CLI args) - never
hardcoded in loop bodies.

Fields marked "TBD" are placeholders for now.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import math


@dataclass
class Config:
    # ---- Environment ----
    env_name: str = "CartPole-v1"
    obs_dim: int = 4
    n_actions: int = 2

    # ---- Architecture (locked 2026-07-12) ----
    hidden_size: int = 256                  # within the N<=256/layer BPTT-memory guideline; training happens on Colab
    n_lif_layers: int = 1                   # single spiking layer - keeps the actor's only weight matrices at
                                             # input->hidden (genuine pre/post spikes) and hidden->action-head (see
                                             # system_b.py's open design question re: the head not spiking - n_lif_layers=1
                                             # avoids the upstream multi-layer credit-assignment problem but does NOT
                                             # resolve that one, it's a separate question)
    T: int = 20                             # SNN timesteps / env step. Starting point: Bellec 2020 (dt=1ms, 50Hz env)

    # ---- LIF neuron time constants (Neftci 2019 - two independent decay constants) ----
    dt: float = 1e-3        # seconds, simulation timestep (Delta t)
    tau_mem: float = 20e-3  # seconds, membrane decay -> beta = exp(-dt/tau_mem)
    tau_syn: float = 5e-3   # seconds, synaptic current decay -> alpha = exp(-dt/tau_syn)
    reset_mechanism: str = "subtract"  # soft reset (Neftci 2019); identical in both systems - locked

    # ---- Encoding ----
    encoding_type: str = "rate"  # deterministic rate coding - locked (NOT Poisson)

    # ---- Eligibility trace (System B, Izhikevich 2007) ----
    tau_e: float = 1.0  # seconds - current value used for training; set from tau_e sweep winner before k-sweep
    tau_e_sweep_values: List[float] = field(default_factory=lambda: [0.2, 0.5, 1.0, 2.0, 5.0])
    A_plus: float = 0.01   # STDP potentiation amplitude, starting point
    A_minus: float = 0.01  # STDP depression amplitude, starting point

    # ---- Learning rates (TBD - tune both with equal grid-search budget) ----
    eta_A: Optional[float] = None  # System A actor LR
    eta_B: Optional[float] = None  # System B R-STDP LR

    # ---- Surrogate gradient (System A only - not used anywhere in System B) ----
    gamma_pd: float = 0.3  # pseudo-derivative scale (Bellec 2020 / Neftci 2019) - CITATION VALUE, see note below
    surrogate_slope: int = 25  # snnTorch's snntorch.surrogate.fast_sigmoid(slope=...) parameter, used in lif.py.
    # NOTE: gamma_pd and surrogate_slope are NOT the same parameterization. gamma_pd is
    # Bellec/Neftci's pseudo-derivative dampening factor for their specific triangular
    # surrogate shape; snnTorch's fast_sigmoid uses a differently-scaled "slope" (higher =
    # steeper, closer to a true Heaviside; snnTorch's own tutorials default to 25). Surrogate
    # *choice* has minimal impact on final performance and snnTorch's default (fast sigmoid)
    # is fine to use - so lif.py uses fast_sigmoid with surrogate_slope, and gamma_pd is kept
    # only as a citation/reference value, not plugged numerically into fast_sigmoid's slope.

    # ---- RL discount (standard actor-critic; not yet pinned in the hyperparameter table) ----
    gamma: float = 0.99

    # ---- Activity regularization (spike-count based, NOT weight-L1; both systems, different mechanism per system) ----
    lam: float = 0.0  # current lambda in use for a given run
    lambda_sweep_values: List[float] = field(default_factory=lambda: [0.0, 1e-4, 1e-3, 1e-2, 1e-1])

    # ---- Reward-delay sweep (locked) ----
    k: int = 1  # current k in use for a given run; measured in ENVIRONMENT TIMESTEPS, resets every episode
    k_sweep_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20])

    # ---- Seeds (locked - minimum 3, report mean +/- std) ----
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])

    # ---- Episode budget (locked 2026-07-12; can still be overridden via CLI) ----
    n_episodes: int = 1000

    # ---- Performance gate (System A, locked) ----
    solve_reward_threshold: float = 195.0
    solve_window: int = 100

    # ---- Energy estimation (Loihi 2 published figure, locked) ----
    joules_per_synaptic_op: float = 23e-12

    
    # ---- BOUNDED Max and Min OBS
    min_obs = [-4.8, -4.0, -0.418, -4.0]
    max_obs = [ 4.8,  4.0,  0.418,  4.0]
    
    def alpha(self) -> float:
        """exp(-dt/tau_syn) - synaptic current decay constant."""
        return math.exp(-self.dt/self.tau_syn)
    def beta(self) -> float:
        """exp(-dt/tau_mem) - membrane potential decay constant (snnTorch's 'beta')."""
        return math.exp(-self.dt / self.tau_mem)