import snntorch as snn
from QLIF import QLIF
import torch
import numpy as np
from typing import Literal
from snntorch.spikegen import latency

BetaValues = list[float] | list[list[float]] | list[list[list[float]]]
LayerBetaStats = dict[str, BetaValues | float]

def spike_encode(
    batch_sequence_embeddings: torch.Tensor,
    n_steps: int,
    input_mode: Literal["spatial", "temporal"] = "spatial",
    encoding_method: Literal["poisson", "latency"] = "poisson",
) -> torch.Tensor:
    """
    batch_sequence_embeddings: [B, seq_len, emb_dim]
    returns:
      spatial: [T, B, seq_len * emb_dim]
      temporal: [T * seq_len, B, emb_dim]

        Encoding is generated first in a shared representation [T, B, seq_len, emb_dim],
        then reshaped for the selected input_mode. This keeps encoding_method and
        input_mode as separate choices.
    """

    batch_size, seq_len, emb_dim = batch_sequence_embeddings.shape

    if encoding_method == "poisson":
        # Independent Bernoulli sampling at each timestep for each input feature.
        rand = torch.rand(
            (n_steps, batch_size, seq_len, emb_dim),
            device=batch_sequence_embeddings.device,
            dtype=batch_sequence_embeddings.dtype,
        )
        spikes_4d = (rand < batch_sequence_embeddings.unsqueeze(0)).to(batch_sequence_embeddings.dtype)
    elif encoding_method == "latency":
        spike_prob_flat = batch_sequence_embeddings.reshape(batch_size, seq_len * emb_dim)
        latency_spikes = latency(
            spike_prob_flat,
            num_steps=n_steps,
            first_spike_time=0,
            # threshold=0.01,
            # tau=1,
            normalize=True,
            clip=True,
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


def build_neuron_layer(
    model_name: str,
    beta: float = np.random.rand(),
    alpha: float = np.random.rand(),
    threshold: float = np.random.rand(),
    learn_beta: bool = False,
    learn_alpha: bool = False,
    learn_threshold: bool = False,
    threshold_layer_scalar: float = 1.0,
    per_neuron_params: bool = False,
    num_neurons: int = 1,
) -> torch.nn.Module:
    """Construct a configured spiking neuron layer by model name."""
    model_name = model_name.lower()
    
    if per_neuron_params:
        betas = torch.rand(num_neurons) if beta is None else torch.full((num_neurons,), beta)
    else:
        betas = beta or np.random.rand() #fallback to random scalar if beta was None and per_neuron_params is False

    if per_neuron_params:
        alphas = torch.rand(num_neurons) if alpha is None else torch.full((num_neurons,), alpha)
    else:
        alphas = alpha or np.random.rand() #fallback to random scalar if alpha was None and per_neuron_params is False

    threshold = (threshold or np.random.rand()) * threshold_layer_scalar

    if model_name == "lif":
        return snn.Leaky(beta=betas, threshold=threshold, init_hidden=False, learn_beta=learn_beta, learn_threshold=learn_threshold)
    if model_name == "synaptic":
        return snn.Synaptic(alpha=alphas, beta=betas, threshold=threshold, init_hidden=False, learn_alpha=learn_alpha, learn_beta=learn_beta, learn_threshold=learn_threshold, reset_mechanism="zero")
    if model_name == "qlif":
        return QLIF(alpha=alphas, beta=betas, threshold=threshold, init_hidden=False, learn_alpha=learn_alpha, learn_beta=learn_beta, learn_threshold=learn_threshold, reset_mechanism="zero")
    raise ValueError("--neuron_model must be one of: lif, synaptic, qlif")

def get_neuron_beta_values_by_layer(
    model: torch.nn.Module,
    layer_names: tuple[str, ...] = ("lif1", "lif2", "lif3"),
) -> dict[str, LayerBetaStats]:
    """Extract per-layer beta values and their mean for available neuron layers."""
    beta_values: dict[str, LayerBetaStats] = {}
    for layer_name in layer_names:
        layer = getattr(model, layer_name, None)
        if layer is None or not hasattr(layer, "beta"):
            continue

        beta_value = layer.beta
        beta_tensor = beta_value if torch.is_tensor(beta_value) else torch.as_tensor(beta_value)
        beta_tensor = beta_tensor.detach().cpu()
        beta_list = beta_tensor.reshape(-1).tolist()
        beta_values[layer_name] = {
            "values": beta_list,
            "mean": float(torch.tensor(beta_list, dtype=torch.float32).mean().item()),
        }

    return beta_values

def parse_threshold_layer_scalars(raw_value):
	if raw_value is None:
		return []
	if isinstance(raw_value, (list, tuple)):
		values = [float(v) for v in raw_value]
	elif isinstance(raw_value, str):
		stripped = raw_value.strip().strip("[]")
		values = [float(v.strip()) for v in stripped.split(",") if v.strip()]
	else:
		raise ValueError("threshold_layer_scalars must be list/tuple/str")

	if len(values) != 3:
		raise ValueError(f"threshold_layer_scalars must have exactly 3 values. Got: {values}")
	return values

def main() -> None:
    """Run a tiny spike encoding demo when this file is executed directly."""
    n_steps = 10
    input_mode = "spatial"
    encoding_methods= ("poisson", "latency")
    scalar_values = (0.0, 0.5, 1.0)

    for encoding_method in encoding_methods:
        print(f"\nencoding_method={encoding_method}, input_mode={input_mode}")
        for scalar in scalar_values:
            sample = torch.tensor([[[scalar]]], dtype=torch.float32)
            spikes = spike_encode(
                batch_sequence_embeddings=sample,
                n_steps=n_steps,
                input_mode=input_mode,
                encoding_method=encoding_method,
            )
            # Spatial mode output is [T, B, N], so flatten to a single 10-step train here.
            spike_train = spikes[:, 0, 0].to(torch.int).tolist()
            print(f"  scalar={scalar} ", spike_train)


if __name__ == "__main__":
    main()