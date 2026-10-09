"""
Energy estimation.

Locked decision: estimated_joules = total_spikes * avg_fan_out * 23e-12
(Loihi 2 published per-synaptic-op figure, clearly an ESTIMATE from
published hardware characterization, not measured silicon power).
avg_fan_out is NOT a constant and must NOT be hardcoded - for a
fully-connected layer of size H, fan_out = H (every spike fans out to all H
post-synaptic neurons), and the true average must be a spike-count-weighted
average across all layers in the actual locked architecture (derived from
config, not a placeholder).
"""

from typing import List

from config import Config


def compute_avg_fan_out(spike_counts_per_layer: List[float], config: Config) -> float:
    """
    Spike-count-weighted average fan-out across all layers of the actual
    architecture (trunk layers + head), NOT a hardcoded constant.

    Args:
        spike_counts_per_layer: total spike count per layer (same ordering
            as the network's layers - trunk layers then head) for the
            period being estimated (e.g. one episode).
        config: run config - layer sizes (config.hidden_size,
            config.n_lif_layers, config.n_actions) define each layer's
            fan_out for a fully-connected topology.

    Returns:
        Spike-count-weighted average fan_out (float).

    For each layer, fan_out_i = size of the NEXT layer it projects to
    (H for hidden->hidden, n_actions for the final layer to the action
    head). Weight each layer's fan_out by its share of total spikes, then
    average. Derive layer sizes from config, never hardcode.
    """
    raise NotImplementedError("TODO: derive fan_out per layer from config and weight by spike share.")


def estimate_energy_joules(total_spikes: float, avg_fan_out: float, config: Config) -> float:
    """
    estimated_joules = total_spikes * avg_fan_out * config.joules_per_synaptic_op

    This is an ESTIMATE based on published Loihi 2 hardware figures
    (~23pJ/synaptic-op), not measured silicon power - label it as such
    anywhere it's reported (plots, README).
    """
    raise NotImplementedError("TODO: implement the energy estimate formula.")
