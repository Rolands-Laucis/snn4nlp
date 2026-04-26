import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import snntorch as snn
import snntorch.functional as SF
import torch
import torch.nn as nn
from snntorch import spikegen
from snntorch import utils
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from QLIF import QLIF
from readers import ReadSENTInputFile
from snn_diagnostics import collect_forward_diagnostics
from snn_diagnostics import plot_layer_membrane_traces
from snn_diagnostics import plot_layer_spike_trains

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAST_INPUT_DIR = PROJECT_ROOT / "input_data" / "cast_sent"


def parse_args():
    parser = argparse.ArgumentParser(description="Train an SNN for token-level binary sentiment analysis")
    parser.add_argument("--input_mode", type=str, default="spatial", choices=["spatial", "temporal"], help="Input mode for the SNN [spatial|temporal]")
    parser.add_argument("--limit", type=int, default=None, help="Limit sample count after dataset preparation (applied separately to train and test)")
    parser.add_argument("--input_file_prefix", type=str, default="sent_d50", help="Prefix for input files")
    parser.add_argument("--output_file_prefix", type=str, default="", help="Prefix for output files")
    parser.add_argument("--num_hidden_1", type=int, default=256, help="Number of neurons in first hidden layer")
    parser.add_argument("--num_hidden_2", type=int, default=128, help="Number of neurons in second hidden layer")
    parser.add_argument("--sim_steps", type=int, default=20, help="Poisson/SNN simulation steps")
    parser.add_argument("--encoding_method", type=str, default="poisson", choices=["poisson", "latency"], help="Spike encoding method [poisson|latency]")
    parser.add_argument("--decoding_method", type=str, default="spike_count", choices=["spike_count", "ttfs"], help="Output decoding method [spike_count|ttfs]")
    parser.add_argument("--ttfs_temporal_loss", type=str, default="ce_temporal_loss", choices=["ce_temporal_loss", "mse_temporal_loss"], help="Temporal loss used when decoding_method=ttfs")
    parser.add_argument("--neuron_model", type=str, default="lif", choices=["lif", "synaptic", "qlif"], help="Neuron model to use [lif|synaptic|qlif]")
    parser.add_argument("--alpha", type=float, default=None, help="Synaptic decay factor for second-order neurons; defaults to beta when omitted")
    parser.add_argument("--beta", type=float, default=None, help="Leaky neuron decay factor. None for learning or random init (0..1 recommended)")
    parser.add_argument("--learn_beta", type=bool, default=False, help="Whether to learn the beta parameter")
    parser.add_argument("--threshold", type=float, default=None, help="Leaky neuron threshold factor. None for learning or random init (0..1 recommended)")
    parser.add_argument("--learn_threshold", type=bool, default=False, help="Whether to learn the threshold parameter")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--save", action="store_true", help="Whether to save the model checkpoint")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "output_results" / "E2"), help="Output directory for checkpoint and metadata")
    parser.add_argument("--diagnose", action="store_true", help="Run first-batch SNN diagnostics and generate plots")
    parser.add_argument(
        "--diagnose_dir",
        type=str,
        default=str(PROJECT_ROOT / "output_results" / "E2"),
        help="Directory for exported diagnostics figures",
    )
    return parser.parse_args()


def build_sentiment_samples(samples, embedding_dim):
    """
    Convert sentiment samples into tensors.

    Expected sample structure: [padded_sequence_embeddings, binary_label]
      - padded_sequence_embeddings: list of token embeddings
      - binary_label: 0/1, bool, or sentiment string
    """
    x_list = []
    y_list = []

    for sample_idx, sample in enumerate(samples):
        if not isinstance(sample, (list, tuple)) or len(sample) < 2:
            raise ValueError(f"Invalid sample format at index {sample_idx}: expected [sequence_embeddings, binary_label]")

        token_embeddings = sample[0]
        label_value = sample[1]
        # print(sample_idx, label_value)

        if not token_embeddings:
            raise ValueError(f"Empty token embedding sequence at sample index {sample_idx}")

        seq_embeddings = []
        for token_idx, token_embedding in enumerate(token_embeddings):
            embedding = torch.as_tensor(token_embedding, dtype=torch.float32)
            if embedding.ndim != 1 or embedding.numel() != embedding_dim:
                raise ValueError(
                    "Embedding dimension mismatch "
                    f"at sample {sample_idx}, token {token_idx}: expected {embedding_dim}, got {embedding.numel()}"
                )
            seq_embeddings.append(embedding)

        if label_value not in (0, 1):
            raise ValueError(f"Label must be exactly 0 or 1. Got: {label_value} at sample index {sample_idx}")

        x_list.append(torch.stack(seq_embeddings, dim=0))
        y_list.append(int(label_value))

    if not x_list:
        raise ValueError("No valid samples were produced for sentiment training.")

    X = torch.stack(x_list, dim=0)  # [num_samples, seq_len, embedding_dim]
    y = torch.tensor(y_list, dtype=torch.long)
    print(f"Built sentiment samples | X shape: {X.shape} | y shape: {y.shape}")
    return X, y


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
        latency_spikes = spikegen.latency(
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


def build_neuron_layer(model_name, layer_size, beta_value = np.random.rand(), alpha = np.random.rand(), threshold = np.random.rand()):
    model_name = model_name.lower()
    if model_name == "lif":
        return snn.Leaky(beta=beta_value if beta_value is not None else np.random.rand(), threshold=threshold if threshold is not None else np.random.rand(), init_hidden=False, learn_beta=args.learn_beta, learn_threshold=args.learn_threshold)
    if model_name == "synaptic":
        return snn.Synaptic(alpha=alpha if alpha is not None else np.random.rand(), beta=beta_value if beta_value is not None else np.random.rand(), threshold=threshold if threshold is not None else np.random.rand(), init_hidden=False, learn_alpha=True, learn_beta=args.learn_beta, learn_threshold=args.learn_threshold)
    if model_name == "qlif":
        return QLIF(alpha=alpha, beta=beta_value if beta_value is not None else np.random.rand(), threshold=threshold if threshold is not None else np.random.rand(), init_hidden=False, learn_alpha=True, learn_beta=args.learn_beta, learn_threshold=args.learn_threshold)
    raise ValueError("--neuron_model must be one of: lif, synaptic, qlif")

class SequenceSentimentSNN(nn.Module):
    """Predict binary sentiment from sequence embeddings."""

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, beta_val, neuron_model_name):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.lif1 = build_neuron_layer(neuron_model_name, layer_size=hidden_size_1, beta_value=args.beta, alpha=args.alpha, threshold=args.threshold)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.lif2 = build_neuron_layer(neuron_model_name, layer_size=hidden_size_2, beta_value=beta_val, alpha=args.alpha, threshold=args.threshold*0.7)
        self.fc3 = nn.Linear(hidden_size_2, output_size)
        self.lif3 = build_neuron_layer(neuron_model_name, layer_size=output_size, beta_value=beta_val, alpha=args.alpha, threshold=args.threshold)

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


def decode_predictions(spike_counts, decoding_method="spike_count", first_spike_idx=None, sample_has_spike=None, final_mem=None):
    if decoding_method == "spike_count":
        preds = torch.argmax(spike_counts, dim=1)
        return preds, 0

    if decoding_method == "ttfs":
        if first_spike_idx is None or sample_has_spike is None:
            raise ValueError("first_spike_idx and sample_has_spike are required for TTFS decoding")

        preds = torch.argmin(first_spike_idx, dim=1)

        all_silent = ~sample_has_spike
        fallback_count = int(all_silent.sum().item())
        if fallback_count:
            if final_mem is None:
                raise ValueError("final_mem is required when TTFS fallback is needed")
            mem_fallback = torch.argmax(final_mem, dim=1)
            preds = torch.where(all_silent, mem_fallback, preds)

        return preds, fallback_count

    raise ValueError("decoding_method must be either 'spike_count' or 'ttfs'")


def compute_classification_loss(
    spike_count_loss_function,
    ttfs_loss_function,
    targets,
    decoding_method="spike_count",
    spike_counts=None,
    ttfs_spk_rec=None,
):
    if decoding_method == "ttfs":
        if ttfs_spk_rec is None:
            raise ValueError("ttfs_spk_rec is required for TTFS loss")
        return ttfs_loss_function(ttfs_spk_rec, targets)

    if spike_counts is None:
        raise ValueError("spike_counts is required for spike_count loss")
    return spike_count_loss_function(spike_counts, targets)


def evaluate_model(model, features, labels, batch_size, device, n_steps):
    eval_ds = TensorDataset(features, labels)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0
    running_fallback = 0
    running_first_spike_time_sum = 0.0
    running_first_spike_time_count = 0

    with torch.no_grad():
        for xb, yb in eval_loader:
            # model reset is handled by init_hidden=False in neuron layers, so no explicit reset needed here.
            xb = xb.to(device)
            yb = yb.to(device)

            spike_seq = spike_encode(
                xb,
                n_steps,
                input_mode=input_mode,
                encoding_method=encoding_method
            ).to(device)
            need_ttfs_state = decoding_method == "ttfs"
            model_output = model(spike_seq, track_ttfs=need_ttfs_state)
            if need_ttfs_state:
                spike_counts, first_spike_idx, sample_has_spike, final_mem, ttfs_spk_rec = model_output
            else:
                spike_counts = model_output
                first_spike_idx = None
                sample_has_spike = None
                final_mem = None
                ttfs_spk_rec = None

            loss = compute_classification_loss(
                loss_fn,
                ttfs_loss_fn,
                yb,
                decoding_method=decoding_method,
                spike_counts=spike_counts,
                ttfs_spk_rec=ttfs_spk_rec,
            )

            running_loss += loss.item() * xb.size(0)
            preds, fallback_count = decode_predictions(
                spike_counts,
                decoding_method=decoding_method,
                first_spike_idx=first_spike_idx,
                sample_has_spike=sample_has_spike,
                final_mem=final_mem,
            )
            running_correct += (preds == yb).sum().item()
            running_total += xb.size(0)
            running_fallback += fallback_count

            if need_ttfs_state and first_spike_idx is not None:
                fired_mask = first_spike_idx <= float(spike_seq.shape[0])
                running_first_spike_time_sum += float(first_spike_idx[fired_mask].sum().item())
                running_first_spike_time_count += int(fired_mask.sum().item())

            utils.reset(model)

    avg_loss = running_loss / max(1, running_total)
    avg_acc = running_correct / max(1, running_total)
    fallback_rate = running_fallback / max(1, running_total)
    mean_first_spike_time = running_first_spike_time_sum / max(1, running_first_spike_time_count)
    return avg_loss, avg_acc, fallback_rate, mean_first_spike_time


args = parse_args()
input_mode = args.input_mode.lower()
if input_mode not in {"spatial", "temporal"}:
    raise ValueError("--input_mode must be either 'spatial' or 'temporal'")
if args.limit is not None and args.limit <= 0:
    raise ValueError("--limit must be a positive integer when provided")

encoding_method = args.encoding_method.lower()
decoding_method = args.decoding_method.lower()
ttfs_temporal_loss_name = args.ttfs_temporal_loss.lower()
neuron_model = args.neuron_model.lower()
alpha = args.alpha if args.alpha is not None else args.beta

sent_train_data, embedding_dim = ReadSENTInputFile(CAST_INPUT_DIR / f"{args.input_file_prefix}_train.pkl", limit=args.limit)
sent_test_data, _ = ReadSENTInputFile(CAST_INPUT_DIR / f"{args.input_file_prefix}_test.pkl", limit=args.limit)
print(len(sent_train_data), len(sent_test_data), embedding_dim)
# print(set(s[1] for s in sent_test_data))
# exit(0)

print('building training samples...')
X_train, y_train = build_sentiment_samples(sent_train_data, embedding_dim)
print('building test samples...')
X_test, y_test = build_sentiment_samples(sent_test_data, embedding_dim)

if args.limit is not None:
    train_limit = min(args.limit, X_train.shape[0])
    test_limit = min(args.limit, X_test.shape[0])
    X_train = X_train[:train_limit]
    y_train = y_train[:train_limit]
    X_test = X_test[:test_limit]
    y_test = y_test[:test_limit]
    print(f"Applied sample limit: train={train_limit}, test={test_limit}")

sequence_length = X_train.shape[1]
num_labels = 2
label_to_idx = {"negative": 0, "positive": 1}
idx_to_label = {0: "negative", 1: "positive"}

train_positive = int((y_train == 1).sum().item())
train_negative = int((y_train == 0).sum().item())
test_positive = int((y_test == 1).sum().item())
test_negative = int((y_test == 0).sum().item())

print("Task: token-level binary sentiment analysis")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"X_train shape: {X_train.shape}  # [samples, seq_len, embed_dim]")
print(f"y_train shape: {y_train.shape}  # [samples]")
print(f"Class balance (train): neg={train_negative}, pos={train_positive}")
print(f"Class balance (test): neg={test_negative}, pos={test_positive}")

input_size = sequence_length * embedding_dim if input_mode == "spatial" else embedding_dim
net = SequenceSentimentSNN(input_size, args.num_hidden_1, args.num_hidden_2, num_labels, args.beta, neuron_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
if ttfs_temporal_loss_name == "ce_temporal_loss":
    ttfs_loss_fn = SF.ce_temporal_loss()
elif ttfs_temporal_loss_name == "mse_temporal_loss":
    ttfs_loss_fn = SF.mse_temporal_loss()
else:
    raise ValueError("--ttfs_temporal_loss must be one of: ce_temporal_loss, mse_temporal_loss")
optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

output_dir = Path(args.output_dir) or PROJECT_ROOT / "output_results" / "E2"
output_dir.mkdir(parents=True, exist_ok=True)

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_filename_base = "_".join(
    [
        args.output_file_prefix or "sent",
        now,
        f"e-{args.epochs}",
        f"s-{args.sim_steps}",
        input_mode,
    ]
)

diagnose_dir = Path(args.diagnose_dir)
diagnostic_paths = []
diagnostics_ran = False
if args.diagnose:
    diagnose_dir.mkdir(parents=True, exist_ok=True)
    print(f"Diagnostics enabled. Export directory: {diagnose_dir}")

print("\nTraining config:")
print(f"  Device: {device}")
print(f"  Input mode: {input_mode}")
print(f"  Encoding method: {encoding_method}")
print(f"  Decoding method: {decoding_method}")
print(f"  Loss: {ttfs_loss_fn.__class__.__name__ if decoding_method == 'ttfs' else loss_fn.__class__.__name__}")
print(f"  Neuron model: {neuron_model}")
print(f"  Beta: {args.beta}")
print(f"  Learn Beta: {args.learn_beta}")
print(f"  Learn Threshold: {args.learn_threshold}")
print(f"  Alpha: {alpha}")
print(f"  Sequence length: {sequence_length}")
print(f"  Embedding dim: {embedding_dim}")
print(f"  Input size: {input_size}")
print(f"  Hidden size 1: {args.num_hidden_1}")
print(f"  Hidden size 2: {args.num_hidden_2}")
print(f"  Output classes: {num_labels}")
print(f"  Num steps: {args.sim_steps}")
print(f"  Batch size: {args.batch_size}")
print(f"  Epochs: {args.epochs}")
print(f"  Learning rate: {args.learning_rate}")
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"  Total learnable parameters: {total_params}")


# Train (samples are shuffled by DataLoader each epoch)
epoch_losses = []
epoch_accuracies = []
epoch_ttfs_fallback_rates = []
epoch_ttfs_mean_first_spike_times = []
progress_print_every_samples = 10_000
net.train()
training_start_time = time.perf_counter()

for epoch in range(args.epochs):
    epoch_start_time = time.perf_counter()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    running_fallback = 0
    running_first_spike_time_sum = 0.0
    running_first_spike_time_count = 0
    next_progress_print_at = progress_print_every_samples
    epoch_total_samples = len(train_loader.dataset)
    diagnostics_ran = False

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        spike_seq = spike_encode(
            xb,
            args.sim_steps,
            input_mode=input_mode,
            encoding_method=encoding_method
        ).to(device)

        # wait 1 epoch to run diagnostics to allow for any initial setup overhead to be excluded from timing and to ensure diagnostics are run on a fully initialized model and data pipeline
        if args.diagnose and not diagnostics_ran:
            with torch.no_grad():
                diagnostics = collect_forward_diagnostics(net, spike_seq)

            sample_index = 0
            diag_filename = "_".join(
                [
                    args.output_file_prefix or "sent",
                    f"e-{epoch}",
                    f's-{sample_index}'
                ]
            )

            spike_fig, _ = plot_layer_spike_trains(
                diagnostics,
                sample_index=sample_index,
                input_spikes=spike_seq,
            )
            spike_path = diagnose_dir / f"{diag_filename}_layer_spike_trains.png"
            spike_fig.savefig(spike_path, dpi=200, bbox_inches="tight")
            diagnostic_paths.append(str(spike_path))

            output_layer_name = list(diagnostics.keys())[-1]
            output_mem_fig, _ = plot_layer_membrane_traces(
                diagnostics,
                layer_name=output_layer_name,
                sample_index=sample_index,
            )
            output_mem_path = diagnose_dir / f"{diag_filename}_{output_layer_name}_membranes.png"
            output_mem_fig.savefig(output_mem_path, dpi=200, bbox_inches="tight")
            diagnostic_paths.append(str(output_mem_path))

            plt.close("all")

            print(f"Diagnostics completed for first training batch. Output layer: {output_layer_name}")
            diagnostics_ran = True
            # for p in diagnostic_paths:
            #     print(f"  saved: {p}")

            if epoch == args.epochs - 1:
                # always exit after diagnostics to avoid long training runs when only diagnostics are desired
                exit(0)

        need_ttfs_state = decoding_method == "ttfs"
        model_output = net(spike_seq, track_ttfs=need_ttfs_state)
        if need_ttfs_state:
            spike_counts, first_spike_idx, sample_has_spike, final_mem, ttfs_spk_rec = model_output
        else:
            spike_counts = model_output
            first_spike_idx = None
            sample_has_spike = None
            final_mem = None
            ttfs_spk_rec = None

        loss = compute_classification_loss(
            loss_fn,
            ttfs_loss_fn,
            yb,
            decoding_method=decoding_method,
            spike_counts=spike_counts,
            ttfs_spk_rec=ttfs_spk_rec,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        preds, fallback_count = decode_predictions(
            spike_counts,
            decoding_method=decoding_method,
            first_spike_idx=first_spike_idx,
            sample_has_spike=sample_has_spike,
            final_mem=final_mem,
        )
        running_correct += (preds == yb).sum().item()
        running_total += xb.size(0)
        running_fallback += fallback_count

        if need_ttfs_state and first_spike_idx is not None:
            fired_mask = first_spike_idx <= float(spike_seq.shape[0])
            running_first_spike_time_sum += float(first_spike_idx[fired_mask].sum().item())
            running_first_spike_time_count += int(fired_mask.sum().item())

        while running_total >= next_progress_print_at:
            progress_pct = (running_total / max(1, epoch_total_samples)) * 100.0
            epoch_elapsed_s = time.perf_counter() - epoch_start_time
            print(
                f"Epoch {epoch + 1}/{args.epochs} progress | "
                f"samples: {running_total}/{epoch_total_samples} ({progress_pct:.2f}%) | "
                f"elapsed_s: {epoch_elapsed_s:.2f}"
            )
            next_progress_print_at += progress_print_every_samples

    epoch_loss = running_loss / max(1, running_total)
    epoch_acc = running_correct / max(1, running_total)
    epoch_fallback_rate = running_fallback / max(1, running_total)
    epoch_mean_first_spike_time = running_first_spike_time_sum / max(1, running_first_spike_time_count)
    epoch_losses.append(float(epoch_loss))
    epoch_accuracies.append(float(epoch_acc))
    epoch_ttfs_fallback_rates.append(float(epoch_fallback_rate))
    epoch_ttfs_mean_first_spike_times.append(float(epoch_mean_first_spike_time))

    epoch_duration_s = time.perf_counter() - epoch_start_time
    elapsed_s = time.perf_counter() - training_start_time
    avg_epoch_s = elapsed_s / float(epoch + 1)
    remaining_epochs = max(0, args.epochs - (epoch + 1))
    eta_minutes = (avg_epoch_s * remaining_epochs) / 60.0
    print(
        f"Epoch {epoch + 1}/{args.epochs} | loss: {epoch_loss:.4f} | acc: {epoch_acc:.4f} "
        f"| epoch_time_s: {epoch_duration_s:.2f} | eta_min: {eta_minutes:.2f}"
    )
    if decoding_method == "ttfs":
        print(f"TTFS fallback rate: {epoch_fallback_rate:.4f}")
        print(f"TTFS mean first spike time (fired output neurons): {epoch_mean_first_spike_time:.4f}")

print("Training finished.")
if args.diagnose:
    exit(0)

test_loss, test_acc, test_ttfs_fallback_rate, test_ttfs_mean_first_spike_time = evaluate_model(
    net,
    X_test,
    y_test,
    args.batch_size,
    device,
    args.sim_steps,
)
print(f"Test evaluation | loss: {test_loss:.4f} | acc: {test_acc:.4f}")
if decoding_method == "ttfs":
    print(f"Test TTFS fallback rate: {test_ttfs_fallback_rate:.4f}")
    print(f"Test TTFS mean first spike time (fired output neurons): {test_ttfs_mean_first_spike_time:.4f}")

if args.save:
    model_output_path = output_dir / f"{run_filename_base}.pt"
    checkpoint = {
        "model_state_dict": net.state_dict(),
        "model_class": "SequenceSentimentSNN",
        "model_config": {
            "input_size": input_size,
            "hidden_size_1": args.num_hidden_1,
            "hidden_size_2": args.num_hidden_2,
            "output_size": num_labels,
            "beta": args.beta,
            "alpha": alpha,
            "input_mode": input_mode,
            "encoding_method": encoding_method,
            "decoding_method": decoding_method,
            "neuron_model": neuron_model,
            "sequence_length": int(sequence_length),
            "embedding_dim": int(embedding_dim),
            "sim_steps": args.sim_steps,
        },
        "label_maps": {
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
        },
        "metrics": {
            "epoch_train_loss": epoch_losses,
            "epoch_train_accuracy": epoch_accuracies,
            "epoch_ttfs_fallback_rate": epoch_ttfs_fallback_rates,
            "epoch_ttfs_mean_first_spike_time": epoch_ttfs_mean_first_spike_times,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "test_ttfs_fallback_rate": float(test_ttfs_fallback_rate),
            "test_ttfs_mean_first_spike_time": float(test_ttfs_mean_first_spike_time),
        },
        "cli_args": vars(args),
    }
    torch.save(checkpoint, model_output_path)
    print(f"Model checkpoint saved to {model_output_path}")

training_metadata = {
    "training_config": {
        "date": now,
        "task": "token_level_binary_sentiment",
        "embedding_dim": int(embedding_dim),
        "sequence_length": int(sequence_length),
        "input_size": int(input_size),
        "num_labels": num_labels,
        "num_training_samples": int(X_train.shape[0]),
        "num_test_samples": int(X_test.shape[0]),
        "device": str(device),
        "total_params": total_params,
        "train_neg": train_negative,
        "train_pos": train_positive,
        "test_neg": test_negative,
        "test_pos": test_positive,
    }
    | {k: str(v) for k, v in vars(args).items()},
    "results": {
        "epoch_train_loss": epoch_losses,
        "epoch_train_accuracy": epoch_accuracies,
        "epoch_ttfs_fallback_rate": epoch_ttfs_fallback_rates,
        "epoch_ttfs_mean_first_spike_time": epoch_ttfs_mean_first_spike_times,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_ttfs_fallback_rate": float(test_ttfs_fallback_rate),
        "test_ttfs_mean_first_spike_time": float(test_ttfs_mean_first_spike_time),
        "diagnostics_enabled": bool(args.diagnose),
        "diagnostic_files": diagnostic_paths,
    },
}

metadata_file = output_dir / f"{run_filename_base}.json"
with open(metadata_file, "w", encoding="utf-8") as f:
    json.dump(training_metadata, f, indent=2)

print(f"\nTraining metadata exported to {metadata_file}")