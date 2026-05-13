import torch
import torch.nn as nn
import snntorch as snn


class SequencePOS_SNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size_1,
        hidden_size_2,
        output_size,
        beta=None,
        alpha=None,
        learn_alpha=False,
        learn_beta=False,
        threshold=None,
        threshold_layer_scalars=None,
        per_neuron_params=False,
        learn_threshold=False,
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, output_size)

        if threshold is None and not learn_threshold:
            threshold = 1.0

        if threshold_layer_scalars is None:
            threshold_layer_scalars = [1.0, 1.0, 1.0]

        def make_param(value, size):
            if value is None:
                return None
            if per_neuron_params:
                return torch.full((size,), float(value))
            return float(value)

        alpha_1 = make_param(alpha, hidden_size_1)
        beta_1 = make_param(beta, hidden_size_1)
        alpha_2 = make_param(alpha, hidden_size_2)
        beta_2 = make_param(beta, hidden_size_2)
        alpha_3 = make_param(alpha, output_size)
        beta_3 = make_param(beta, output_size)

        thr1 = torch.rand(hidden_size_1) if threshold is None else float(threshold) * threshold_layer_scalars[0]
        thr2 = torch.rand(hidden_size_2) if threshold is None else float(threshold) * threshold_layer_scalars[1]
        thr3 = torch.rand(output_size) if threshold is None else float(threshold) * threshold_layer_scalars[2]

        self.lif1 = snn.Synaptic(
            alpha=alpha_1,
            beta=beta_1,
            threshold=thr1,
            learn_alpha=learn_alpha,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            reset_mechanism="zero",
        )
        self.lif2 = snn.Synaptic(
            alpha=alpha_2,
            beta=beta_2,
            threshold=thr2,
            learn_alpha=learn_alpha,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            reset_mechanism="zero",
        )
        self.lif3 = snn.Synaptic(
            alpha=alpha_3,
            beta=beta_3,
            threshold=thr3,
            learn_alpha=learn_alpha,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            reset_mechanism="zero",
        )

        self.seq_linear = None

    def init_seq_classifier(self, hidden_size, num_tags):
        self.seq_linear = nn.Linear(hidden_size, num_tags)

    def forward(self, spike_seq, track_ttfs: bool = False):
        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()
        syn3, mem3 = self.lif3.init_synaptic()

        per_step_outputs = []

        for step in range(spike_seq.size(0)):
            x = spike_seq[step]
            cur1 = self.fc1(x)
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)

            cur2 = self.fc2(spk1)
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)

            cur3 = self.fc3(spk2)
            spk3, syn3, mem3 = self.lif3(cur3, syn3, mem3)

            per_step_outputs.append(spk3)

        return torch.stack(per_step_outputs, dim=0)
