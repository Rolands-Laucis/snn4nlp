import snntorch as snn
from snntorch import spikegen
from snntorch import utils
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from readers import ReadUPOSInputFile
import argparse
from pathlib import Path
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAST_POS_DIR = PROJECT_ROOT / 'input_data' / 'cast_pos'

parser = argparse.ArgumentParser(description='Train an SNN for POS tagging')
parser.add_argument('--input_mode', type=str, default='spatial', help='Input mode for the SNN [spatial|temporal] (default: spatial)')
parser.add_argument('--label_feature', type=str, default='upos', choices=['upos', 'xpos'], help='Target label feature to predict [upos|xpos] (default: upos)')
parser.add_argument('--limit', type=int, default=None, help='Limit the number of sentences for testing (default: 100)')
parser.add_argument('--input_file_prefix', type=str, default='pos_d50', help='Prefix for input files (default: pos)')
parser.add_argument('--output_file_prefix', type=str, default='', help='Prefix for output files')
parser.add_argument('--context_size', type=int, default=5, help='N context words; predict POS of the last token in the window (default: 5)')
parser.add_argument('--num_hidden', type=int, default=64, help='Number of hidden units (default: 64)')
parser.add_argument('--sim_steps', type=int, default=20, help='Poisson/SNN simulation steps (default: 10)')
parser.add_argument('--beta', type=float, default=0.95, help='Leaky neuron decay factor (default: 0.95)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs (default: 5)')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate (default: 0.001)')
parser.add_argument('--save', type=bool, default=False, help='Whether to save the model checkpoint')
parser.add_argument('--model_output_dir', type=str, default=PROJECT_ROOT / 'output_results' / 'upos', help='Output directory for saved model checkpoint')
args = parser.parse_args()
input_mode = args.input_mode.lower()
if input_mode not in {'spatial', 'temporal'}:
    raise ValueError("--input_mode must be either 'spatial' or 'temporal'")
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

# Poisson encoding for SNN input
def poisson_encode(batch_context_embeddings, n_steps, input_mode='spatial'):
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
    if input_mode == 'spatial':
        # Follow snnTorch docs: data shape [batch x input_size].
        spike_prob_flat = spike_prob.reshape(B, context_n * emb_dim)

        spikes = spikegen.rate(
            spike_prob_flat,
            num_steps=n_steps,
            gain=1,
            offset=0,
            first_spike_time=0,
            time_var_input=False,
        )

        # Return [T, B, context_n * emb_dim]
        return spikes.reshape(n_steps, B, context_n * emb_dim)

    if input_mode == 'temporal':
        spike_segments = []
        for word_index in range(context_n):
            word_prob = spike_prob[:, word_index, :]
            word_spikes = spikegen.rate(
                word_prob,
                num_steps=n_steps,
                gain=1,
                offset=0,
                first_spike_time=0,
                time_var_input=False,
            )
            spike_segments.append(word_spikes)

        return torch.cat(spike_segments, dim=0)

    raise ValueError("input_mode must be either 'spatial' or 'temporal'")

class ContextSNN(nn.Module):
    """Predict POS of last token in context window."""

    def __init__(self, input_size, hidden_size, output_size, beta_val):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta_val, init_hidden=True)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=beta_val, init_hidden=True)

    def forward(self, spike_seq):
        """
        spike_seq: [T, B, input_size]
        Returns output spike counts over all timesteps [B, output_size].
        """
        spk2_sum = torch.zeros(
            spike_seq.shape[1],
            self.fc2.out_features,
            device=spike_seq.device,
            dtype=spike_seq.dtype,
        )

        for step in range(spike_seq.shape[0]):
            cur1 = self.fc1(spike_seq[step])
            spk1 = self.lif1(cur1)
            cur2 = self.fc2(spk1)
            spk2 = self.lif2(cur2)
            spk2_sum += spk2

        return spk2_sum


input_size = args.context_size * embedding_dim if input_mode == 'spatial' else embedding_dim
net = ContextSNN(input_size, args.num_hidden, num_labels, args.beta)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

print(f"\nTraining config:")
print(f"  Device: {device}")
print(f"  Input mode: {input_mode}")
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

    with torch.no_grad():
        for xb, yb in eval_loader:
            utils.reset(model)
            xb = xb.to(device)
            yb = yb.to(device)

            spike_seq = poisson_encode(xb, n_steps, input_mode=input_mode).to(device)
            spike_counts = model(spike_seq)
            loss = loss_fn(spike_counts, yb)

            running_loss += loss.item() * xb.size(0)
            preds = torch.argmax(spike_counts, dim=1)
            running_correct += (preds == yb).sum().item()
            running_total += xb.size(0)

            utils.reset(model)

    avg_loss = running_loss / max(1, running_total)
    avg_acc = running_correct / max(1, running_total)
    return avg_loss, avg_acc


# Train (samples are shuffled by DataLoader each epoch)
epoch_losses = []
epoch_accuracies = []
net.train()
utils.reset(net)
for epoch in range(args.epochs):
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for xb, yb in train_loader:
        utils.reset(net)
        xb = xb.to(device)                                          # [B, context_n, emb_dim]
        yb = yb.to(device)                                          # [B]

        spike_seq = poisson_encode(xb, args.sim_steps, input_mode=input_mode).to(device)        # [T, B, input_size]
        spike_counts = net(spike_seq)                               # [B, num_labels]
        loss = loss_fn(spike_counts, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        preds = torch.argmax(spike_counts, dim=1)
        running_correct += (preds == yb).sum().item()
        running_total += xb.size(0)

        utils.reset(net)

    epoch_loss = running_loss / max(1, running_total)
    epoch_acc = running_correct / max(1, running_total)
    epoch_losses.append(float(epoch_loss))
    epoch_accuracies.append(float(epoch_acc))
    print(f"Epoch {epoch + 1}/{args.epochs} | loss: {epoch_loss:.4f} | acc: {epoch_acc:.4f}")

print("Training finished.")

test_loss, test_acc = evaluate_model(net, X_test, y_test, args.batch_size, device, args.sim_steps)
print(f"Test evaluation | loss: {test_loss:.4f} | acc: {test_acc:.4f}")

# Save trained model checkpoint for easy reload.
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_filename_base = "_".join([args.output_file_prefix or args.label_feature, now, f'e-{args.epochs}', f'ctx-{args.context_size}', str(round(test_acc * 100, 2))])
if args.save:
    model_output_path = Path(args.model_output_dir) / f'{run_filename_base}.pt'
    if not model_output_path.is_absolute():
        model_output_path = PROJECT_ROOT / model_output_path
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": net.state_dict(),
        "model_class": "ContextSNN",
        "model_config": {
            "input_size": input_size,
            "hidden_size": args.num_hidden,
            "output_size": num_labels,
            "beta": args.beta,
            "input_mode": input_mode,
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
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
        },
        "cli_args": vars(args),
    }
    torch.save(checkpoint, model_output_path)
    print(f"Model checkpoint saved to {model_output_path}")

# Export training metadata and results to JSON
training_metadata = {
    "training_config": {
        "date":now,
        "embedding_dim": int(embedding_dim),
        "input_size": input_size,
        "label_feature": label_feature,
        "num_labels": num_labels,
        "num_training_samples": int(X_train.shape[0]),
        "device": str(device),
    } | vars(args),  # include all CLI args in metadata
    "results": {
        "epoch_train_loss": epoch_losses,
        "epoch_train_accuracy": epoch_accuracies,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
    },
}

output_dir = PROJECT_ROOT / 'output_results' / 'upos'
output_dir.mkdir(parents=True, exist_ok=True)
final_acc = epoch_accuracies[-1] if epoch_accuracies else 0.0
metadata_file = output_dir / f'{run_filename_base}.json'

with open(metadata_file, 'w', encoding='utf-8') as f:
    json.dump(training_metadata, f, indent=2)

print(f"\nTraining metadata exported to {metadata_file}")