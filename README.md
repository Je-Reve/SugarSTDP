# SugarSTDP - SUrrogate Gradient And Reward STDP Benchmarking

### 🚧 WORK IN PROGRESS 

Author: Jeanard Sinfuego
Date: 2026-10-09

Spiking actor-critic on CartPole-v1: a controlled benchmark comparison between surrogate-gradient training (System A) and reward-modulated STDP (System B). Both systems run under strict experimental control (same architecture, same compute budget, actor learning rule as the sole independent variable) to produce a mechanistic characterization of the distal credit-assignment problem. Spike counts are logged per layer and converted to estimated synaptic operations on Loihi 2 (~23 pJ/op), producing a Pareto frontier of task reward versus estimated joules across activity-regularization sweeps.

---

## Two systems benchmarked

### System A: Surrogate-Gradient Actor-Critic

- Trained via BPTT with surrogate gradients (fast sigmoid, snnTorch)
- Actor and critic share a spiking LIF trunk; critic uses MSE on TD targets with standard backprop
- Establishes what is achievable when credit assignment is computed offline through the full computational graph

### System B: Reward-Modulated STDP Actor-Critic

- Identical architecture to System A; the actor learning rule is the sole independent variable
- Actor update: event-gated Izhikevich R-STDP eligibility traces multiplied by TD error at reward delivery. No backprop through the actor.
- Traces accumulate exclusively on genuine causal pre-before-post spike pairs and decay exponentially with time constant tau\_e, carrying per-synapse credit forward until the reward signal arrives
- Critic: identical to System A (standard backprop, MSE on TD targets)

The reward-delay sweep (reward delivered every k environment timesteps, k in {1, 5, 10, 20}) characterizes the degradation onset as traces decay before reward arrives. System A is robust across k because BPTT is indifferent to reward timing; System B's performance is predicted to fall as tau\_e becomes insufficient to bridge the delay.

---

## Custom LIF neuron

snnTorch's built-in neurons are insufficient for this project. The Neftci et al. (2019) biophysical LIF model requires two independent coupled ODEs:

```
I[t+1] = alpha * I[t] + sum_j W_ij * S_j[t]          # synaptic current; alpha = exp(-dt/tau_syn)
V[t+1] = beta * V[t] + (1 - beta) * R * I[t] - S[t]   # membrane, soft reset; beta = exp(-dt/tau_mem)
```

`snn.Leaky` tracks only membrane potential under a single decay constant, equivalent to assuming tau\_syn approaches zero. `snn.Synaptic` has two state variables but omits the `(1-beta)` normalization factor, producing a steady-state amplitude error of 1/(1-beta) (approximately 20x for tau\_mem = 20ms), and applies an implicit rather than explicit Euler update ordering inconsistent with Neftci's discretization. `CustomLIFNeuron` in `sugarstdp/lif.py` implements the correct two-ODE explicit Euler formulation, verified against closed-form analytical solutions before any RL code runs.

---

## Key variables (currently)

| Variable | Values | Description |
|---|---|---|
| `tau_mem` | 20 ms | Membrane time constant; beta = exp(-dt/tau\_mem) |
| `tau_syn` | 5 ms | Synaptic current decay; alpha = exp(-dt/tau\_syn) |
| `T` | 20 steps | SNN timesteps per environment step (dt = 1 ms) |
| `tau_e` | {200 ms, 500 ms, 1 s, 2 s, 5 s} | Eligibility trace decay; swept before k-sweep |
| `k` | {1, 5, 10, 20} | Reward delivered every k environment timesteps |
| `lambda` | {0, 1e-4, 1e-3, 1e-2, 1e-1} | Activity regularization coefficient |
| `A_plus` | 0.01 | STDP potentiation amplitude (System B) |
| `A_minus` | 0.01 | STDP depression amplitude (System B) |
| Seeds | {0, 1, 2} | Results reported as mean +/- std across 3 seeds |

---

## Progress

- [x] Full project skeleton: all modules, signatures, config.py, episode loop, sweep runners, logging infrastructure
- [x] `CustomLIFNeuron.forward()` implemented
- [ ] System A: physics verification passing, surrogate-gradient actor-critic training on CartPole-v1
- [ ] System B: eligibility trace unit tests, R-STDP actor-critic training
- [ ] Reward-delay sweep, tau\_e sweep, activity regularization sweep, energy-efficiency Pareto analysis

---

## Repo layout

```
config.py                  central hyperparameter config
sugarstdp/
  lif.py                   custom two-decay LIF neuron
  encoding.py              deterministic rate coding
  eligibility.py           Izhikevich event-gated R-STDP trace
  regularization.py        activity penalty for System A and B
  energy.py                Loihi energy estimate from spike counts
  networks.py              SpikingTrunk / ActorNetwork / Critic wiring
  systems_base.py          shared ActorCriticSystem interface
  system_a.py              surrogate-gradient actor-critic
  system_b.py              R-STDP actor-critic
  env_wrapper.py           Gymnasium wrapper, reward-delay logic, seeding
  episode_loop.py          shared training loop
  logger.py                episode/step logging
  plotting.py              reward curves, trace rasters, Pareto frontier (stubs)
sweeps/
  tau_e_sweep.py           eligibility trace time constant sweep (runs first)
  k_sweep.py               reward-delay sweep
  lambda_sweep.py          activity regularization sweep
tests/
  test_lif_physics.py      closed-form LIF physics verification
  test_eligibility_trace.py  synthetic spike-pair unit tests
train.py                   CLI entry point
```

---

## References

Izhikevich, E. M. (2007). Solving the distal reward problem through linkage of STDP and dopamine signaling. *Cerebral Cortex*, 17(10), 2443-2452.

Neftci, E. O., Mostafa, H., and Zenke, F. (2019). Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks. *IEEE Signal Processing Magazine*, 36(6), 51-63.