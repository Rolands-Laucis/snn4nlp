import snntorch as snn
from snntorch import spikegen
from snntorch import utils
import snntorch.functional as SF
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from QLIF import QLIF

from readers import ReadUPOSInputFile
import argparse
from pathlib import Path
import json
from datetime import datetime
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAST_POS_DIR = PROJECT_ROOT / 'input_data' / 'cast_pos'

parser = argparse.ArgumentParser(description='Train an SNN for POS tagging')
parser.add_argument('--input_mode', type=str, default='spatial', choices=['spatial', 'temporal'], help='Input mode for the SNN [spatial|temporal]')
parser.add_argument('--label_feature', type=str, default='upos', choices=['upos', 'xpos'], help='Target label feature to predict [upos|xpos]')
parser.add_argument('--limit', type=int, default=None, help='Limit sample count after dataset preparation (applied separately to train and test)')
parser.add_argument('--input_file_prefix', type=str, default='pos_d50', help='Prefix for input files')
parser.add_argument('--output_file_prefix', type=str, default='', help='Prefix for output files')
parser.add_argument('--context_size', type=int, default=5, help='N context words; predict POS of the last token in the window')
parser.add_argument('--num_hidden_1', type=int, default=256, help='Number of neurons in first hidden layer')
parser.add_argument('--num_hidden_2', type=int, default=128, help='Number of neurons in second hidden layer')
parser.add_argument('--sim_steps', type=int, default=20, help='Poisson/SNN simulation steps')
parser.add_argument('--encoding_method', type=str, default='poisson', choices=['poisson', 'latency'], help='Spike encoding method [poisson|latency]')
parser.add_argument('--decoding_method', type=str, default='spike_count', choices=['spike_count', 'ttfs'], help='Output decoding method [spike_count|ttfs]')
parser.add_argument('--ttfs_temporal_loss', type=str, default='ce_temporal_loss', choices=['ce_temporal_loss', 'mse_temporal_loss'], help='Temporal loss used when decoding_method=ttfs')
parser.add_argument('--neuron_model', type=str, default='lif', choices=['lif', 'synaptic', 'qlif'], help='Neuron model to use [lif|synaptic|qlif]')
parser.add_argument('--alpha', type=float, default=None, help='Synaptic decay factor for second-order neurons; defaults to beta when omitted')
parser.add_argument('--beta', type=float, default=0.95, help='Leaky neuron decay factor')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--save', type=bool, default=False, help='Whether to save the model checkpoint')
parser.add_argument('--output_dir', type=str, default=PROJECT_ROOT / 'output_results' / 'E1', help='Output directory for saved model checkpoint')
args = parser.parse_args()
input_mode = args.input_mode.lower()
if input_mode not in {'spatial', 'temporal'}:
    raise ValueError("--input_mode must be either 'spatial' or 'temporal'")
if args.context_size <= 0 or args.context_size % 2 == 0:
    raise ValueError("--context_size must be a positive odd integer")
encoding_method = args.encoding_method.lower()
decoding_method = args.decoding_method.lower()
ttfs_temporal_loss_name = args.ttfs_temporal_loss.lower()
neuron_model = args.neuron_model.lower()
alpha = args.alpha if args.alpha is not None else args.beta
label_feature = args.label_feature.lower()
label_feature_to_index = {'upos': 1, 'xpos': 2}
label_index = label_feature_to_index[label_feature]

pos_train_data, embedding_dim = ReadUPOSInputFile(CAST_POS_DIR / f'{args.input_file_prefix}_train.pkl', limit=args.limit)
pos_dev_data, _ = ReadUPOSInputFile(CAST_POS_DIR / f'{args.input_file_prefix}_dev.pkl', limit=args.limit)
pos_test_data, _ = ReadUPOSInputFile(CAST_POS_DIR / f'{args.input_file_prefix}_test.pkl', limit=args.limit)

pos_train_data += pos_dev_data  # combine train and dev for training

if not embedding_dim and pos_train_data and pos_train_data[0] and len(pos_train_data[0][0]) > 3:
    embedding_dim = len(pos_train_data[0][0][3:])

# Build target label vocabulary from training data
label_tags = {
    word_info[label_index]
    for sentence in pos_train_data + pos_test_data
    for word_info in sentence
}

label_to_idx = {tag: idx for idx, tag in enumerate(sorted(label_tags))}
idx_to_label = {idx: tag for tag, idx in label_to_idx.items()}
num_labels = len(label_to_idx)

print(f"Label feature: {label_feature}")
print(f"Label tags ({num_labels}): {label_tags}")

def build_rolling_context_samples(sentences, label_map, label_index, context_n, embedding_dim, skip_unknown_labels=False):
    """
    Build (X, y) where each sample predicts the selected label of the middle token in a context window.

    For each sentence and token position t:
      - Input: centered embeddings [t-half_window ... t+half_window],
               with sentence overhang on either side padded by an UNK vector
      - Target: label at position t (the middle token)
      
    Each word_info is [lemma, upos, xpos, embed1, embed2, ...]
    """
    if context_n <= 0 or context_n % 2 == 0:
        raise ValueError("context_n must be a positive odd integer")

    x_list = []
    y_list = []
    skipped_samples = 0

    # UNK/padding embedding used whenever the context window overhangs sentence boundaries.
    unk_embedding = torch.zeros(embedding_dim, dtype=torch.float32)
    half_window = context_n // 2

    for sentence in sentences:
        for t in range(len(sentence)):
            ctx_embeddings = []
            start = t - half_window
            end = t + half_window

            for pos in range(start, end + 1):
                if pos < 0 or pos >= len(sentence):
                    ctx_embeddings.append(unk_embedding.clone())
                else:
                    # Extract embedding from word_info[3:]
                    embedding = torch.tensor(sentence[pos][3:], dtype=torch.float32)
                    if embedding.numel() != embedding_dim:
                        raise ValueError(
                            f"Embedding dimension mismatch at sentence token: expected {embedding_dim}, got {embedding.numel()}"
                        )
                    ctx_embeddings.append(embedding)

            label = sentence[t][label_index]
            if label not in label_map:
                if skip_unknown_labels:
                    skipped_samples += 1
                    continue
                raise KeyError(f"Unknown label '{label}' not found in training vocabulary")

            x_list.append(torch.stack(ctx_embeddings, dim=0))       # [context_n, embedding_dim]
            y_list.append(label_map[label])                         # scalar class index

    if not x_list:
        raise ValueError("No valid samples were produced for sample creation.")

    X = torch.stack(x_list, dim=0)                                  # [num_samples, context_n, embedding_dim]
    y = torch.tensor(y_list, dtype=torch.long)                      # [num_samples]
    if skip_unknown_labels and skipped_samples:
        print(f"Skipped {skipped_samples} samples with unknown labels during sample creation.")
    return X, y


# Build rolling-window training samples
X_train, y_train = build_rolling_context_samples(
    pos_train_data[:args.limit] if args.limit is not None else pos_train_data,
    label_to_idx,
    label_index,
    args.context_size,
    embedding_dim,
)

X_test, y_test = build_rolling_context_samples(
    pos_test_data[:args.limit] if args.limit is not None else pos_test_data,
    label_to_idx,
    label_index,
    args.context_size,
    embedding_dim,
    skip_unknown_labels=True,
)

if args.limit is not None:
    if args.limit <= 0:
        raise ValueError("--limit must be a positive integer when provided")

    train_limit = min(args.limit, X_train.shape[0])
    test_limit = min(args.limit, X_test.shape[0])
    X_train = X_train[:train_limit]
    y_train = y_train[:train_limit]
    X_test = X_test[:test_limit]
    y_test = y_test[:test_limit]
    print(f"Applied sample limit: train={train_limit}, test={test_limit}")

print(f"Training samples: {X_train.shape[0]}")
print(f"X_train shape: {X_train.shape}  # [samples, context, embed_dim]")
print(f"y_train shape: {y_train.shape}  # [samples]")
print(f"Example X_train[0]:{X_train[0].shape} Y_train[0]: {y_train[0]}")

# Spike encoding for SNN input
def spike_encode(batch_context_embeddings, n_steps, input_mode='spatial', encoding_method='poisson'):
    """
    batch_context_embeddings: [B, context_n, emb_dim]
        returns:
            spatial: [T, B, context_n * emb_dim]
            temporal: [T * context_n, B, emb_dim]
    """
    # Convert signed embeddings to [0, 1] probabilities for rate coding.
    max_abs = batch_context_embeddings.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    spike_prob = 0.5 * ((batch_context_embeddings / max_abs) + 1.0)
    spike_prob = spike_prob.clamp(0.0, 1.0)

    # Keep explicit zero-padding vectors silent.
    pad_mask = batch_context_embeddings.abs().sum(dim=2, keepdim=True).eq(0)
    spike_prob = spike_prob.masked_fill(pad_mask, 0.0)

    B, context_n, emb_dim = spike_prob.shape

    def encode_features(prob_tensor):
        if encoding_method == 'poisson':
            return spikegen.rate(
                prob_tensor,
                num_steps=n_steps,
                gain=1,
                offset=0,
                first_spike_time=0,
                time_var_input=False,
            )

        if encoding_method == 'latency':
            return spikegen.latency(
                prob_tensor,
                num_steps=n_steps,
                threshold=0.01,
                tau=1,
                first_spike_time=0,
                clip=True,
                normalize=True,
                linear=True,
            )

        raise ValueError("encoding_method must be either 'poisson' or 'latency'")

    if input_mode == 'spatial':
        # Follow snnTorch docs: data shape [batch x input_size].
        spike_prob_flat = spike_prob.reshape(B, context_n * emb_dim)

        spikes = encode_features(spike_prob_flat)

        # Return [T, B, context_n * emb_dim]
        return spikes.reshape(n_steps, B, context_n * emb_dim)

    if input_mode == 'temporal':
        spike_segments = []
        for word_index in range(context_n):
            word_prob = spike_prob[:, word_index, :]
            word_spikes = encode_features(word_prob)
            spike_segments.append(word_spikes)

        return torch.cat(spike_segments, dim=0)

    raise ValueError("input_mode must be either 'spatial' or 'temporal'")


def build_neuron_layer(model_name, beta_value, layer_size):
    model_name = model_name.lower()
    if model_name == 'lif':
        return snn.Leaky(beta=beta_value, init_hidden=False)
    if model_name == 'synaptic':
        return snn.Synaptic(alpha=alpha, beta=beta_value, init_hidden=False)
    if model_name == 'qlif':
        return QLIF(alpha=alpha, beta=beta_value, init_hidden=False)
    raise ValueError("--neuron_model must be one of: lif, synaptic, qlif")


def reset_model_state(model):
    # With init_hidden=False, states are managed locally in forward pass
    # No need to reset instance states
    pass


class ContextSNN(nn.Module):
    """Predict POS of last token in context window."""

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, beta_val, neuron_model_name):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.lif1 = build_neuron_layer(neuron_model_name, beta_value=beta_val, layer_size=hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.lif2 = build_neuron_layer(neuron_model_name, beta_value=beta_val, layer_size=hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, output_size)
        self.lif3 = build_neuron_layer(neuron_model_name, beta_value=beta_val, layer_size=output_size)

    def forward(self, spike_seq, track_ttfs=False):
        """
        spike_seq: [T, B, input_size]
        Returns output spike counts over all timesteps [B, output_size].
        When track_ttfs=True, also returns:
          - first_spike_idx: [B, output_size] first spike time per output neuron
          - sample_has_spike: [B] whether any output neuron fired
          - final_mem: [B, output_size] output membrane at the final simulated step
        """
        num_steps = spike_seq.shape[0]
        batch_size = spike_seq.shape[1]
        output_size = self.fc3.out_features

        spk3_sum = torch.zeros(
            batch_size,
            output_size,
            device=spike_seq.device,
            dtype=spike_seq.dtype,
        )
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
            has_fired = torch.zeros(
                (batch_size, output_size),
                device=spike_seq.device,
                dtype=torch.bool,
            )
            ttfs_spk_rec = []

        final_mem = torch.zeros(
            batch_size,
            output_size,
            device=spike_seq.device,
            dtype=spike_seq.dtype,
        )

        # Determine neuron types and initialize appropriate hidden states
        neuron_class1 = self.lif1.__class__.__name__
        neuron_class2 = self.lif2.__class__.__name__
        neuron_class3 = self.lif3.__class__.__name__
        
        mem1 = torch.zeros(batch_size, self.fc1.out_features, device=spike_seq.device, dtype=spike_seq.dtype)
        mem2 = torch.zeros(batch_size, self.fc2.out_features, device=spike_seq.device, dtype=spike_seq.dtype)
        mem3 = torch.zeros(batch_size, output_size, device=spike_seq.device, dtype=spike_seq.dtype)
        
        syn1 = None
        syn2 = None
        syn3 = None
        
        if neuron_class1 in ('Synaptic', 'QLIF'):
            syn1 = torch.zeros(batch_size, self.fc1.out_features, device=spike_seq.device, dtype=spike_seq.dtype)
        
        if neuron_class2 in ('Synaptic', 'QLIF'):
            syn2 = torch.zeros(batch_size, self.fc2.out_features, device=spike_seq.device, dtype=spike_seq.dtype)
        
        if neuron_class3 in ('Synaptic', 'QLIF'):
            syn3 = torch.zeros(batch_size, output_size, device=spike_seq.device, dtype=spike_seq.dtype)

        for step in range(num_steps):
            cur1 = self.fc1(spike_seq[step])
            
            # Layer 1: Call with appropriate hidden state arguments
            if neuron_class1 in ('Synaptic', 'QLIF'):
                spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            else:  # Leaky
                spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            
            # Layer 2: Call with appropriate hidden state arguments
            if neuron_class2 in ('Synaptic', 'QLIF'):
                spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
            else:  # Leaky
                spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            
            # Layer 3: Call with appropriate hidden state arguments
            if neuron_class3 in ('Synaptic', 'QLIF'):
                spk3, syn3, mem3 = self.lif3(cur3, syn3, mem3)
            else:  # Leaky
                spk3, mem3 = self.lif3(cur3, mem3)

            spk3_sum += spk3
            final_mem = mem3

            if track_ttfs:
                ttfs_spk_rec.append(spk3)

                spk3_fired = spk3 > 0
                new_fired = spk3_fired & (~has_fired)
                first_spike_idx[new_fired] = float(step)
                has_fired = has_fired | spk3_fired

                # Once all output neurons in the batch have fired once, TTFS timings are complete.
                if torch.all(has_fired):
                    break

        if track_ttfs:
            sample_has_spike = has_fired.any(dim=1)
            ttfs_spk_rec = torch.stack(ttfs_spk_rec, dim=0)
            return spk3_sum, first_spike_idx, sample_has_spike, final_mem, ttfs_spk_rec

        return spk3_sum


def decode_predictions(spike_counts, decoding_method='spike_count', first_spike_idx=None, sample_has_spike=None, final_mem=None):
    if decoding_method == 'spike_count':
        return torch.argmax(spike_counts, dim=1), 0

    if decoding_method == 'ttfs':
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
    decoding_method='spike_count',
    spike_counts=None,
    ttfs_spk_rec=None,
):
    if decoding_method == 'ttfs':
        if ttfs_spk_rec is None:
            raise ValueError("ttfs_spk_rec is required for TTFS loss")
        return ttfs_loss_function(ttfs_spk_rec, targets)

    if spike_counts is None:
        raise ValueError("spike_counts is required for spike_count loss")
    return spike_count_loss_function(spike_counts, targets)


input_size = args.context_size * embedding_dim if input_mode == 'spatial' else embedding_dim
net = ContextSNN(input_size, args.num_hidden_1, args.num_hidden_2, num_labels, args.beta, neuron_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
if ttfs_temporal_loss_name == 'ce_temporal_loss':
    ttfs_loss_fn = SF.ce_temporal_loss()
elif ttfs_temporal_loss_name == 'mse_temporal_loss':
    ttfs_loss_fn = SF.mse_temporal_loss()
else:
    raise ValueError("--ttfs_temporal_loss must be one of: ce_temporal_loss, mse_temporal_loss")
optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

print(f"\nTraining config:")
print(f"  Device: {device}")
print(f"  Input mode: {input_mode}")
print(f"  Encoding method: {encoding_method}")
print(f"  Decoding method: {decoding_method}")
print(f"  Loss: {ttfs_loss_fn.__class__.__name__ if decoding_method == 'ttfs' else loss_fn.__class__.__name__}")
print(f"  Neuron model: {neuron_model}")
print(f"  Beta: {args.beta}")
print(f"  Alpha: {alpha}")
print(f"  Context size: {args.context_size}")
print(f"  Input size: {input_size}")
print(f"  Hidden size 1: {args.num_hidden_1}")
print(f"  Hidden size 2: {args.num_hidden_2}")
print(f"  Output classes: {num_labels}")
print(f"  Num steps: {args.sim_steps}")
print(f"  Batch size: {args.batch_size}")
print(f"  Epochs: {args.epochs}")
print(f"  Learning rate: {args.learning_rate}")
# print total learnable parameters
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"  Total learnable parameters: {total_params}")


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
            reset_model_state(model)
            xb = xb.to(device)
            yb = yb.to(device)

            spike_seq = spike_encode(xb, n_steps, input_mode=input_mode, encoding_method=encoding_method).to(device)
            need_ttfs_state = decoding_method == 'ttfs'
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


# Train (samples are shuffled by DataLoader each epoch)
epoch_losses = []
epoch_accuracies = []
epoch_ttfs_fallback_rates = []
epoch_ttfs_mean_first_spike_times = []
progress_print_every_samples = 10_000
net.train()
reset_model_state(net)
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

    for xb, yb in train_loader:
        reset_model_state(net)
        xb = xb.to(device)                                          # [B, context_n, emb_dim]
        yb = yb.to(device)                                          # [B]

        spike_seq = spike_encode(xb, args.sim_steps, input_mode=input_mode, encoding_method=encoding_method).to(device)        # [T, B, input_size]
        need_ttfs_state = decoding_method == 'ttfs'
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

        reset_model_state(net)

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
    if decoding_method == 'ttfs':
        print(f"TTFS fallback rate: {epoch_fallback_rate:.4f}")
        print(f"TTFS mean first spike time (fired output neurons): {epoch_mean_first_spike_time:.4f}")

print("Training finished.")

test_loss, test_acc, test_ttfs_fallback_rate, test_ttfs_mean_first_spike_time = evaluate_model(net, X_test, y_test, args.batch_size, device, args.sim_steps)
print(f"Test evaluation | loss: {test_loss:.4f} | acc: {test_acc:.4f}")
if decoding_method == 'ttfs':
    print(f"Test TTFS fallback rate: {test_ttfs_fallback_rate:.4f}")
    print(f"Test TTFS mean first spike time (fired output neurons): {test_ttfs_mean_first_spike_time:.4f}")

# Save trained model checkpoint for easy reload.
output_dir = Path(args.output_dir) or PROJECT_ROOT / 'output_results' / 'E1'
output_dir.mkdir(parents=True, exist_ok=True)

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_filename_base = "_".join([args.output_file_prefix or args.label_feature, now, f'e-{args.epochs}', f'ctx-{args.context_size}', f's-{args.sim_steps}'])
if args.save:
    model_output_path = output_dir / f'{run_filename_base}.pt'

    checkpoint = {
        "model_state_dict": net.state_dict(),
        "model_class": "ContextSNN",
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
            "label_feature": label_feature,
            "context_size": args.context_size,
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

# Export training metadata and results to JSON
# del args['output_dir']
training_metadata = {
    "training_config": {
        "date":now,
        "embedding_dim": int(embedding_dim),
        "input_size": input_size,
        "label_feature": label_feature,
        "num_labels": num_labels,
        "num_training_samples": int(X_train.shape[0]),
        "device": str(device),
        "total_params":total_params
    } | {k: str(v) for k, v in vars(args).items()},  # include all CLI args in metadata #{k: str(v) for k, v in vars(args).items()}
    "results": {
        "epoch_train_loss": epoch_losses,
        "epoch_train_accuracy": epoch_accuracies,
        "epoch_ttfs_fallback_rate": epoch_ttfs_fallback_rates,
        "epoch_ttfs_mean_first_spike_time": epoch_ttfs_mean_first_spike_times,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_ttfs_fallback_rate": float(test_ttfs_fallback_rate),
        "test_ttfs_mean_first_spike_time": float(test_ttfs_mean_first_spike_time),
    },
}


metadata_file = output_dir / f'{run_filename_base}.json'
with open(metadata_file, 'w', encoding='utf-8') as f:
    json.dump(training_metadata, f, indent=2)

print(f"\nTraining metadata exported to {metadata_file}")