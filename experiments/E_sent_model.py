import torch
import torch.nn as nn

from snn_util import build_neuron_layer


class SequenceSentimentSNN(nn.Module):
    """Predict binary sentiment from sequence embeddings."""

    def __init__(
        self,
        input_size,
        hidden_size_1,
        hidden_size_2,
        output_size,
        neuron_model_name,
        beta=None,
        learn_beta=False,
        alpha=None,
        learn_alpha=False,
        threshold=None,
        threshold_layer_scalars=None,
        per_neuron_params=False,
    ):
        super().__init__()
        scalars = threshold_layer_scalars or [1.0, 0.8, 0.7]
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        # the beta and alpha params are either a scalar for all neurons sharing the same value or a tensor of shape (hidden_size,) for per-neuron values, and are passed to build_neuron_layer which handles both cases
        self.lif1 = build_neuron_layer(neuron_model_name, beta=beta, alpha=alpha, threshold=threshold, threshold_layer_scalar=scalars[0], learn_beta=learn_beta, learn_alpha=learn_alpha, per_neuron_params=per_neuron_params, num_neurons=hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.lif2 = build_neuron_layer(neuron_model_name, beta=beta, alpha=alpha, threshold=threshold, threshold_layer_scalar=scalars[1], learn_beta=learn_beta, learn_alpha=learn_alpha, per_neuron_params=per_neuron_params, num_neurons=hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, output_size)
        self.lif3 = build_neuron_layer(neuron_model_name, beta=beta, alpha=alpha, threshold=threshold, threshold_layer_scalar=scalars[2], learn_beta=learn_beta, learn_alpha=learn_alpha, per_neuron_params=per_neuron_params, num_neurons=output_size)

    def forward(self, spike_seq, track_ttfs=False):
        num_steps = spike_seq.shape[0]
        batch_size = spike_seq.shape[1]
        output_size = self.fc3.out_features

        spk3_sum = torch.zeros(batch_size, output_size, device=spike_seq.device, dtype=spike_seq.dtype)
        first_spike_idx = None
        has_fired = None
        ttfs_spk_rec = None
        if track_ttfs:
            first_spike_idx = torch.full(
                (batch_size, output_size),
                float(num_steps + 1),
                device=spike_seq.device,
                dtype=spike_seq.dtype,
            )
            has_fired = torch.zeros((batch_size, output_size), device=spike_seq.device, dtype=torch.bool)
            ttfs_spk_rec = []

        final_mem = torch.zeros(batch_size, output_size, device=spike_seq.device, dtype=spike_seq.dtype)

        neuron_class1 = self.lif1.__class__.__name__
        neuron_class2 = self.lif2.__class__.__name__
        neuron_class3 = self.lif3.__class__.__name__

        mem1 = torch.zeros(batch_size, self.fc1.out_features, device=spike_seq.device, dtype=spike_seq.dtype)
        mem2 = torch.zeros(batch_size, self.fc2.out_features, device=spike_seq.device, dtype=spike_seq.dtype)
        mem3 = torch.zeros(batch_size, output_size, device=spike_seq.device, dtype=spike_seq.dtype)

        syn1 = None
        syn2 = None
        syn3 = None
        if neuron_class1 in ("Synaptic", "QLIF"):
            syn1 = torch.zeros(batch_size, self.fc1.out_features, device=spike_seq.device, dtype=spike_seq.dtype)
        if neuron_class2 in ("Synaptic", "QLIF"):
            syn2 = torch.zeros(batch_size, self.fc2.out_features, device=spike_seq.device, dtype=spike_seq.dtype)
        if neuron_class3 in ("Synaptic", "QLIF"):
            syn3 = torch.zeros(batch_size, output_size, device=spike_seq.device, dtype=spike_seq.dtype)

        for step in range(num_steps):
            cur1 = self.fc1(spike_seq[step])
            if neuron_class1 in ("Synaptic", "QLIF"):
                spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            else:
                spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            if neuron_class2 in ("Synaptic", "QLIF"):
                spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
            else:
                spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            if neuron_class3 in ("Synaptic", "QLIF"):
                spk3, syn3, mem3 = self.lif3(cur3, syn3, mem3)
            else:
                spk3, mem3 = self.lif3(cur3, mem3)

            spk3_sum += spk3
            final_mem = mem3

            if track_ttfs:
                ttfs_spk_rec.append(spk3)
                spk3_fired = spk3 > 0
                new_fired = spk3_fired & (~has_fired)
                first_spike_idx[new_fired] = float(step)
                has_fired = has_fired | spk3_fired
                if torch.all(has_fired):
                    break

        if track_ttfs:
            sample_has_spike = has_fired.any(dim=1)
            ttfs_spk_rec = torch.stack(ttfs_spk_rec, dim=0)
            return spk3_sum, first_spike_idx, sample_has_spike, final_mem, ttfs_spk_rec

        return spk3_sum
