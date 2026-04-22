import snntorch as snn
from snntorch import spikegen
from snntorch import utils
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from QLIF import QLIF

from readers import ReadUPOSInputFile
import argparse
from pathlib import Path
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAST_POS_DIR = PROJECT_ROOT / 'input_data' / 'cast_pos'

parser = argparse.ArgumentParser(description='Train an SNN for POS tagging')
parser.add_argument('--input_mode', type=str, default='spatial', help='Input mode for the SNN [spatial|temporal]')
parser.add_argument('--label_feature', type=str, default='upos', choices=['upos', 'xpos'], help='Target label feature to predict [upos|xpos]')
parser.add_argument('--limit', type=int, default=None, help='Limit the number of sentences for testing')
parser.add_argument('--input_file_prefix', type=str, default='pos_d50', help='Prefix for input files')
parser.add_argument('--output_file_prefix', type=str, default='', help='Prefix for output files')
parser.add_argument('--context_size', type=int, default=5, help='N context words; predict POS of the last token in the window')
parser.add_argument('--num_hidden', type=int, default=64, help='Number of hidden units')
parser.add_argument('--sim_steps', type=int, default=20, help='Poisson/SNN simulation steps')
parser.add_argument('--encoding_method', type=str, default='poisson', choices=['poisson', 'latency'], help='Spike encoding method [poisson|latency]')
parser.add_argument('--decoding_method', type=str, default='spike_count', choices=['spike_count', 'ttfs'], help='Output decoding method [spike_count|ttfs]')
parser.add_argument('--neuron_model', type=str, default='lif', choices=['lif', 'rleaky', 'synaptic', 'rsynaptic', 'lapicque', 'alpha', 'qlif'], help='Neuron model to use [lif|rleaky|synaptic|rsynaptic|lapicque|alpha|qlif]')
parser.add_argument('--alpha', type=float, default=None, help='Synaptic decay factor for second-order neurons; defaults to beta when omitted')
parser.add_argument('--beta', type=float, default=0.95, help='Leaky neuron decay factor')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--save', type=bool, default=False, help='Whether to save the model checkpoint')
parser.add_argument('--output_dir', type=str, default=PROJECT_ROOT / 'output_results' / 'E1', help='Output directory for saved model checkpoint')
args = parser.parse_args()
input_mode = args.input_mode.lower()
if input_mode not in {'spatial', 'temporal'}:
    raise ValueError("--input_mode must be either 'spatial' or 'temporal'")
encoding_method = args.encoding_method.lower()
decoding_method = args.decoding_method.lower()
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
    Build (X, y) where each sample predicts the selected label of the last token in a context window.

    For each sentence and token position t:
      - Input: embeddings [t-context_n+1 ... t], with left overhang padded by zero vector
            - Target: label at position t
      
    Each word_info is [lemma, upos, xpos, embed1, embed2, ...]
    """
    x_list = []
    y_list = []
    skipped_samples = 0
    
    zero_embedding = torch.zeros(embedding_dim, dtype=torch.float32)

    for sentence in sentences:
        for t in range(len(sentence)):
            ctx_embeddings = []
            start = t - context_n + 1

            for pos in range(start, t + 1):
                if pos < 0:
                    ctx_embeddings.append(zero_embedding.clone())
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
    pos_train_data[:args.limit] if args.limit else pos_train_data,  # limit to first 100 sentences for quick testing; remove slice for full data
    label_to_idx,
    label_index,
    args.context_size,
    embedding_dim,
)

X_test, y_test = build_rolling_context_samples(
    pos_test_data[:args.limit] if args.limit else pos_test_data,
    label_to_idx,
    label_index,
    args.context_size,
    embedding_dim,
    skip_unknown_labels=True,
)

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
        return snn.Leaky(beta=beta_value, init_hidden=True)
    if model_name == 'rleaky':
        return snn.RLeaky(beta=beta_value, linear_features=layer_size, init_hidden=True)
    if model_name == 'synaptic':
        return snn.Synaptic(alpha=alpha, beta=beta_value, init_hidden=True)
    if model_name == 'rsynaptic':
        return snn.RSynaptic(alpha=alpha, beta=beta_value, linear_features=layer_size, init_hidden=True)
    if model_name == 'lapicque':
        return snn.Lapicque(beta=beta_value, init_hidden=True)
    if model_name == 'alpha':
        return snn.Alpha(alpha=alpha, beta=beta_value, init_hidden=True)
    if model_name == 'qlif':
        return QLIF(alpha=alpha, beta=beta_value, init_hidden=True)
    raise ValueError("--neuron_model must be one of: lif, rleaky, synaptic, rsynaptic, lapicque, alpha, qlif")


def reset_model_state(model):
    utils.reset(model)


class ContextSNN(nn.Module):
    """Predict POS of last token in context window."""

    def __init__(self, input_size, hidden_size, output_size, beta_val, neuron_model_name):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = build_neuron_layer(neuron_model_name, beta_value=beta_val, layer_size=hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = build_neuron_layer(neuron_model_name, beta_value=beta_val, layer_size=output_size)

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
        output_size = self.fc2.out_features

        spk2_sum = torch.zeros(
            batch_size,
            output_size,
            device=spike_seq.device,
            dtype=spike_seq.dtype,
        )
        first_spike_idx = None
        has_fired = None
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

        final_mem = torch.zeros(
            batch_size,
            output_size,
            device=spike_seq.device,
            dtype=spike_seq.dtype,
        )

        for step in range(spike_seq.shape[0]):
            cur1 = self.fc1(spike_seq[step])
            spk1 = self.lif1(cur1)
            cur2 = self.fc2(spk1)
            spk2 = self.lif2(cur2)
            spk2_sum += spk2
            final_mem = self.lif2.mem if hasattr(self.lif2, 'mem') else cur2

            if track_ttfs:
                spk2_fired = spk2 > 0
                new_fired = spk2_fired & (~has_fired)
                first_spike_idx[new_fired] = float(step)
                has_fired = has_fired | spk2_fired

                # Once all output neurons in the batch have fired once, TTFS timings are complete.
                if torch.all(has_fired):
                    break

        if track_ttfs:
            sample_has_spike = has_fired.any(dim=1)
            return spk2_sum, first_spike_idx, sample_has_spike, final_mem

        return spk2_sum


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


input_size = args.context_size * embedding_dim if input_mode == 'spatial' else embedding_dim
net = ContextSNN(input_size, args.num_hidden, num_labels, args.beta, neuron_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

print(f"\nTraining config:")
print(f"  Device: {device}")
print(f"  Input mode: {input_mode}")
print(f"  Encoding method: {encoding_method}")
print(f"  Decoding method: {decoding_method}")
print(f"  Neuron model: {neuron_model}")
print(f"  Alpha: {alpha}")
print(f"  Context size: {args.context_size}")
print(f"  Input size: {input_size}")
print(f"  Hidden size: {args.num_hidden}")
print(f"  Output classes: {num_labels}")
print(f"  Num steps: {args.sim_steps}")
print(f"  Batch size: {args.batch_size}")
print(f"  Epochs: {args.epochs}")


def evaluate_model(model, features, labels, batch_size, device, n_steps):
    eval_ds = TensorDataset(features, labels)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0
    running_fallback = 0

    with torch.no_grad():
        for xb, yb in eval_loader:
            reset_model_state(model)
            xb = xb.to(device)
            yb = yb.to(device)

            spike_seq = spike_encode(xb, n_steps, input_mode=input_mode, encoding_method=encoding_method).to(device)
            need_ttfs_state = decoding_method == 'ttfs'
            model_output = model(spike_seq, track_ttfs=need_ttfs_state)
            if need_ttfs_state:
                spike_counts, first_spike_idx, sample_has_spike, final_mem = model_output
            else:
                spike_counts = model_output
                first_spike_idx = None
                sample_has_spike = None
                final_mem = None
            loss = loss_fn(spike_counts, yb)

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

            utils.reset(model)

    avg_loss = running_loss / max(1, running_total)
    avg_acc = running_correct / max(1, running_total)
    fallback_rate = running_fallback / max(1, running_total)
    return avg_loss, avg_acc, fallback_rate


# Train (samples are shuffled by DataLoader each epoch)
epoch_losses = []
epoch_accuracies = []
epoch_ttfs_fallback_rates = []
net.train()
reset_model_state(net)
for epoch in range(args.epochs):
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    running_fallback = 0

    for xb, yb in train_loader:
        reset_model_state(net)
        xb = xb.to(device)                                          # [B, context_n, emb_dim]
        yb = yb.to(device)                                          # [B]

        spike_seq = spike_encode(xb, args.sim_steps, input_mode=input_mode, encoding_method=encoding_method).to(device)        # [T, B, input_size]
        need_ttfs_state = decoding_method == 'ttfs'
        model_output = net(spike_seq, track_ttfs=need_ttfs_state)
        if need_ttfs_state:
            spike_counts, first_spike_idx, sample_has_spike, final_mem = model_output
        else:
            spike_counts = model_output
            first_spike_idx = None
            sample_has_spike = None
            final_mem = None
        loss = loss_fn(spike_counts, yb)

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

        reset_model_state(net)

    epoch_loss = running_loss / max(1, running_total)
    epoch_acc = running_correct / max(1, running_total)
    epoch_fallback_rate = running_fallback / max(1, running_total)
    epoch_losses.append(float(epoch_loss))
    epoch_accuracies.append(float(epoch_acc))
    epoch_ttfs_fallback_rates.append(float(epoch_fallback_rate))
    if decoding_method == 'ttfs':
        print(f"Epoch {epoch + 1}/{args.epochs} | loss: {epoch_loss:.4f} | acc: {epoch_acc:.4f} | ttfs_fallback_rate: {epoch_fallback_rate:.4f}")
    else:
        print(f"Epoch {epoch + 1}/{args.epochs} | loss: {epoch_loss:.4f} | acc: {epoch_acc:.4f}")

print("Training finished.")

test_loss, test_acc, test_ttfs_fallback_rate = evaluate_model(net, X_test, y_test, args.batch_size, device, args.sim_steps)
print(f"Test evaluation | loss: {test_loss:.4f} | acc: {test_acc:.4f}")
if decoding_method == 'ttfs':
    print(f"Test TTFS fallback rate: {test_ttfs_fallback_rate:.4f}")

# Save trained model checkpoint for easy reload.
output_dir = Path(args.output_dir) or PROJECT_ROOT / 'output_results' / 'E1'
output_dir.mkdir(parents=True, exist_ok=True)

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_filename_base = "_".join([args.output_file_prefix or args.label_feature, now, f'e-{args.epochs}', f'ctx-{args.context_size}', str(round(test_acc * 100, 2))])
if args.save:
    model_output_path = output_dir / f'{run_filename_base}.pt'

    checkpoint = {
        "model_state_dict": net.state_dict(),
        "model_class": "ContextSNN",
        "model_config": {
            "input_size": input_size,
            "hidden_size": args.num_hidden,
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
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "test_ttfs_fallback_rate": float(test_ttfs_fallback_rate),
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
    } | {k: str(v) for k, v in vars(args).items()},  # include all CLI args in metadata #{k: str(v) for k, v in vars(args).items()}
    "results": {
        "epoch_train_loss": epoch_losses,
        "epoch_train_accuracy": epoch_accuracies,
        "epoch_ttfs_fallback_rate": epoch_ttfs_fallback_rates,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_ttfs_fallback_rate": float(test_ttfs_fallback_rate),
    },
}


metadata_file = output_dir / f'{run_filename_base}.json'
with open(metadata_file, 'w', encoding='utf-8') as f:
    json.dump(training_metadata, f, indent=2)

print(f"\nTraining metadata exported to {metadata_file}")