import torch
import torch.nn as nn
import snntorch as snn
from TorchCRF import CRF
from snn_util import spike_encode


class SequencePOS_SNN(nn.Module):
    """
    SNN + CRF for UPOS sequence labeling.
    
    Architecture:
      Input: (batch, seq_len, emb_dim) — token embeddings
      Process each token sequentially with state persistence:
        → fc1/lif1: 256 neurons
        → fc2/lif2: 128 neurons
                → linear_out: nn.Linear(128, num_tags) — raw real-valued emissions per token
      → CRF: log-likelihood loss or Viterbi decoding
    
    State (syn, mem) persists across token positions, giving each token context
    from all prior tokens. Emission scores are real-valued for CRF.
    """
    def __init__(
        self,
        emb_dim,
        hidden_size_1,
        hidden_size_2,
        num_tags,
        n_steps=20,
        input_mode="spatial",
        encoding_method="latency",
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

        if alpha is None:
            alpha = 0.95
        if beta is None:
            beta = 0.9

        self.n_steps = int(n_steps)
        self.input_mode = input_mode
        self.encoding_method = encoding_method

        self.fc1 = nn.Linear(emb_dim, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        
        # Plain linear layer for emissions (no spiking)
        self.linear_out = nn.Linear(hidden_size_2, num_tags)
        
        # CRF for sequence decoding
        self.crf = CRF(num_tags)

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

        thr1 = torch.rand(hidden_size_1) if threshold is None else float(threshold) * threshold_layer_scalars[0]
        thr2 = torch.rand(hidden_size_2) if threshold is None else float(threshold) * threshold_layer_scalars[1]

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

    def forward(self, x_emb, tags=None, mask=None):
        """
        Forward pass through SNN + CRF for sequence labeling.
        
        Parameters
        ----------
        x_emb : (batch, seq_len, emb_dim)
            Token embeddings for the sequence
        tags : (batch, seq_len), optional
            Gold tag indices. If provided, returns CRF loss.
        mask : (batch, seq_len), optional
            Boolean mask for valid tokens (True = real token, False = pad)
        
        Returns
        -------
        loss : scalar            if tags provided (negated CRF log-likelihood)
        predictions : list[list]  if tags is None (Viterbi decoded sequences)
        """
        batch_size, seq_len, emb_dim = x_emb.shape
        
        # Initialize SNN state (no reset between tokens—state carries forward)
        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()
        
        # Process each token sequentially, collecting emissions
        emissions_list = []
        
        for t in range(seq_len):
            # Token embeddings are already in [0, 1], so pass them through directly.
            x_t = x_emb[:, t, :].unsqueeze(1)  # (batch, 1, emb_dim)
            x_spikes = spike_encode(
                x_t,
                n_steps=self.n_steps,
                input_mode=self.input_mode,
                encoding_method=self.encoding_method,
            )
            
            token_emissions = []
            for step in range(x_spikes.shape[0]):
                x_step = x_spikes[step]

                # Pass through SNN layers, state persists across both time and token positions.
                cur1 = self.fc1(x_step)
                spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)

                cur2 = self.fc2(spk1)
                spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)

                # Project to real-valued emissions and accumulate across the spike train.
                token_emissions.append(self.linear_out(spk2))

            emissions_t = torch.stack(token_emissions, dim=0).mean(dim=0)
            emissions_list.append(emissions_t)
        
        # Stack emissions: (batch, seq_len, num_tags)
        emissions = torch.stack(emissions_list, dim=1)

        if tags is not None:
            # Return scalar CRF loss (negative mean log-likelihood)
            return -self.crf(emissions, tags, mask=mask).mean()
        else:
            # Return Viterbi decoded sequences
            return self.crf.viterbi_decode(emissions, mask=mask)
