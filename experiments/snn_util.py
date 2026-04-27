import snntorch as snn
from QLIF import QLIF
import torch
import numpy as np

def spike_encode(
    batch_sequence_embeddings,
    n_steps,
    input_mode="spatial",
    encoding_method="poisson"
):
    """
    batch_sequence_embeddings: [B, seq_len, emb_dim]
    returns:
      spatial: [T, B, seq_len * emb_dim]
      temporal: [T * seq_len, B, emb_dim]

        Encoding is generated first in a shared representation [T, B, seq_len, emb_dim],
        then reshaped for the selected input_mode. This keeps encoding_method and
        input_mode as separate choices.
    """
    max_abs = batch_sequence_embeddings.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    base_prob = (batch_sequence_embeddings.abs() / max_abs).clamp(0.0, 1.0)

    # Keep explicit zero-padding vectors silent.
    pad_mask = batch_sequence_embeddings.abs().sum(dim=2, keepdim=True).eq(0)
    base_prob = base_prob.masked_fill(pad_mask, 0.0)

    batch_size, seq_len, emb_dim = base_prob.shape

    if encoding_method == "poisson":
        spike_prob = (base_prob).clamp(0.0, 1.0)
        # Independent Bernoulli sampling at each timestep for each input feature.
        rand = torch.rand(
            (n_steps, batch_size, seq_len, emb_dim),
            device=base_prob.device,
            dtype=base_prob.dtype,
        )
        spikes_4d = (rand < spike_prob.unsqueeze(0)).to(base_prob.dtype)
    elif encoding_method == "latency":
        spike_prob_flat = base_prob.reshape(batch_size, seq_len * emb_dim)
        latency_spikes = snn.spikegen.latency(
            spike_prob_flat,
            num_steps=n_steps,
            threshold=0.01,
            tau=1,
            first_spike_time=0,
            clip=True,
            normalize=True,
            linear=True,
        )
        spikes_4d = latency_spikes.reshape(n_steps, batch_size, seq_len, emb_dim)
    else:
        raise ValueError("encoding_method must be either 'poisson' or 'latency'")

    if input_mode == "spatial":
        return spikes_4d.reshape(n_steps, batch_size, seq_len * emb_dim)

    if input_mode == "temporal":
        # Present words sequentially in time using the same encoded spikes.
        return spikes_4d.permute(2, 0, 1, 3).reshape(seq_len * n_steps, batch_size, emb_dim)

    raise ValueError("input_mode must be either 'spatial' or 'temporal'")


def build_neuron_layer(model_name, beta = np.random.rand(), alpha = np.random.rand(), threshold = np.random.rand(), learn_beta=False, learn_threshold=False, threshold_layer_scalar=1.0):
    model_name = model_name.lower()
    if model_name == "lif":
        return snn.Leaky(beta=beta or np.random.rand(), threshold=(threshold or np.random.rand()) * threshold_layer_scalar, init_hidden=False, learn_beta=learn_beta, learn_threshold=learn_threshold)
    if model_name == "synaptic":
        return snn.Synaptic(alpha=alpha or np.random.rand(), beta=beta or np.random.rand(), threshold=(threshold or np.random.rand()) * threshold_layer_scalar, init_hidden=False, learn_alpha=True, learn_beta=learn_beta, learn_threshold=learn_threshold)
    if model_name == "qlif":
        return QLIF(alpha=alpha or np.random.rand(), beta=beta or np.random.rand(), threshold=(threshold or np.random.rand()) * threshold_layer_scalar, init_hidden=False, learn_alpha=True, learn_beta=learn_beta, learn_threshold=learn_threshold)
    raise ValueError("--neuron_model must be one of: lif, synaptic, qlif")

def get_neuron_beta_values_by_layer(model, layer_names=("lif1", "lif2", "lif3")):
    beta_values = {}
    for layer_name in layer_names:
        layer = getattr(model, layer_name, None)
        if layer is None or not hasattr(layer, "beta"):
            continue

        beta_value = layer.beta
        if torch.is_tensor(beta_value):
            beta_value = beta_value.detach().cpu().reshape(-1).tolist()
            if len(beta_value) == 1:
                beta_value = beta_value[0]
        elif hasattr(beta_value, "item"):
            beta_value = beta_value.item()

        beta_values[layer_name] = beta_value

    return beta_values