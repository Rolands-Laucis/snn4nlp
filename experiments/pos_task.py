import snntorch as snn
from snntorch import spikegen
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from readers import ReadUPOSInputFile
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Train an SNN for UPOS tagging')
parser.add_argument('--limit', type=int, default=100, help='Limit the number of sentences for testing (default: 100)')
parser.add_argument('--input_file_prefix', type=str, default='pos_d50', help='Prefix for input files (default: pos)')
parser.add_argument('--context_size', type=int, default=5, help='N context words; predict UPOS of the last token in the window (default: 5)')
parser.add_argument('--num_hidden', type=int, default=64, help='Number of hidden units (default: 64)')
parser.add_argument('--num_steps', type=int, default=20, help='Poisson/SNN simulation steps (default: 10)')
parser.add_argument('--beta', type=float, default=0.95, help='Leaky neuron decay factor (default: 0.95)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs (default: 5)')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate (default: 0.001)')
args = parser.parse_args()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAST_POS_DIR = PROJECT_ROOT / 'input_data' / 'cast_pos'

pos_train_data, embedding_dim = ReadUPOSInputFile(CAST_POS_DIR / f'{args.input_file_prefix}_train.tsv', limit=args.limit)
pos_dev_data, _ = ReadUPOSInputFile(CAST_POS_DIR / f'{args.input_file_prefix}_dev.tsv', limit=args.limit)
pos_test_data, _ = ReadUPOSInputFile(CAST_POS_DIR / f'{args.input_file_prefix}_test.tsv', limit=args.limit)

if not embedding_dim and pos_train_data and pos_train_data[0] and len(pos_train_data[0][0]) > 3:
    embedding_dim = len(pos_train_data[0][0][3:])

# Get parameters from CLI arguments
context_size = args.context_size
num_hidden = args.num_hidden
num_steps = args.num_steps
beta = args.beta
batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate

# Build UPOS tag vocabulary from training data
upos_tags = set()
for sentence in pos_train_data:
    for word_info in sentence:
        upos_tags.add(word_info[1])  # upos is at index 1

upos_to_idx = {tag: idx for idx, tag in enumerate(sorted(upos_tags))}
idx_to_upos = {idx: tag for tag, idx in upos_to_idx.items()}
num_ud_tags = len(upos_to_idx)

print(f"UPOS tags ({num_ud_tags}): {upos_tags}")

def build_rolling_context_samples(sentences, upos_map, context_n, embedding_dim):
    """
    Build (X, y) where each sample predicts the UPOS of the last token in a context window.

    For each sentence and token position t:
      - Input: embeddings [t-context_n+1 ... t], with left overhang padded by zero vector
      - Target: UPOS at position t
      
    Each word_info is [lemma, upos, xpos, embed1, embed2, ...]
    """
    x_list = []
    y_list = []
    
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

            x_list.append(torch.stack(ctx_embeddings, dim=0))       # [context_n, embedding_dim]
            y_list.append(upos_map[sentence[t][1]])                 # scalar class index (upos at index 1)

    X = torch.stack(x_list, dim=0)                                  # [num_samples, context_n, embedding_dim]
    y = torch.tensor(y_list, dtype=torch.long)                      # [num_samples]
    return X, y


# Build rolling-window training samples
X_train, y_train = build_rolling_context_samples(
    pos_train_data[:args.limit] if args.limit else pos_train_data,  # limit to first 100 sentences for quick testing; remove slice for full data
    upos_to_idx,
    context_size,
    embedding_dim,
)

print(f"Training samples: {X_train.shape[0]}")
print(f"X_train shape: {X_train.shape}  # [samples, context, embed_dim]")
print(f"y_train shape: {y_train.shape}  # [samples]")
print(f"Example X_train[0]:\n", X_train[0].shape, y_train[0])


# Poisson encoding for SNN input
def poisson_encode(batch_context_embeddings, n_steps):
    """
    batch_context_embeddings: [B, context_n, emb_dim]
    returns: [T, B, context_n * emb_dim]
    """
    # Convert signed embeddings to [0, 1] probabilities for rate coding.
    max_abs = batch_context_embeddings.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    spike_prob = 0.5 * ((batch_context_embeddings / max_abs) + 1.0)
    spike_prob = spike_prob.clamp(0.0, 1.0)

    # Keep explicit zero-padding vectors silent.
    pad_mask = batch_context_embeddings.abs().sum(dim=2, keepdim=True).eq(0)
    spike_prob = spike_prob.masked_fill(pad_mask, 0.0)

    # Follow snnTorch docs: data shape [batch x input_size].
    B, context_n, emb_dim = spike_prob.shape
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
    spikes = spikes.reshape(n_steps, B, context_n * emb_dim)
    return spikes

# sanity check; Debug once during setup: first input sample, first real token embedding (not left padding) to verify correct encoding and shapes before training
if False:
    print("\nSetup debug Poisson spike train (first sample, first word embedding):")
    poisson_debug_selected_sample = 2
    setup_debug_word_info = pos_train_data[0][poisson_debug_selected_sample]                        # [lemma, upos, xpos, embed...]
    setup_debug_word = setup_debug_word_info[0]
    setup_debug_embedding = torch.tensor(setup_debug_word_info[3:], dtype=torch.float32)
    print(f"  word: {setup_debug_word}")
    print(f"  embedding vector shape: {setup_debug_embedding.shape}")
    print("  embedding vector values:")
    print(setup_debug_embedding)
    # exit(0)

    setup_debug_input = X_train[poisson_debug_selected_sample, -1, :].unsqueeze(0).unsqueeze(0)     # [1, 1, emb_dim]
    setup_debug_spikes = poisson_encode(setup_debug_input, num_steps)    # [T, 1, emb_dim]
    print(f"  input shape: {setup_debug_input.shape}")
    print(f"  spike shape: {setup_debug_spikes.shape}")
    print(f"  mean spike rate: {setup_debug_spikes.float().mean().item():.4f}")
    print("  values:")
    print(setup_debug_spikes[:, 0, :])
    exit(0)

class ContextSNN(nn.Module):
    """Predict UPOS of last token in context window."""

    def __init__(self, input_size, hidden_size, output_size, beta_val):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta_val)  # built-in surrogate gradient
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=beta_val)

    def forward(self, spike_seq):
        """
        spike_seq: [T, B, input_size]
        Returns final-step output membrane potential [B, output_size].
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        for step in range(spike_seq.shape[0]):
            cur1 = self.fc1(spike_seq[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

        return mem2


input_size = context_size * embedding_dim
net = ContextSNN(input_size, num_hidden, num_ud_tags, beta)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

print(f"\nTraining config:")
print(f"  Device: {device}")
print(f"  Context size: {context_size}")
print(f"  Input size: {input_size}")
print(f"  Hidden size: {num_hidden}")
print(f"  Output classes: {num_ud_tags}")
print(f"  Num steps: {num_steps}")
print(f"  Batch size: {batch_size}")
print(f"  Epochs: {num_epochs}")


# Train (samples are shuffled by DataLoader each epoch)
net.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for xb, yb in train_loader:
        xb = xb.to(device)                                          # [B, context_n, emb_dim]
        yb = yb.to(device)                                          # [B]

        spike_seq = poisson_encode(xb, num_steps).to(device)        # [T, B, input_size]
        logits = net(spike_seq)                                     # [B, num_ud_tags]
        loss = loss_fn(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1)
        running_correct += (preds == yb).sum().item()
        running_total += xb.size(0)

    epoch_loss = running_loss / max(1, running_total)
    epoch_acc = running_correct / max(1, running_total)
    print(f"Epoch {epoch + 1}/{num_epochs} | loss: {epoch_loss:.4f} | acc: {epoch_acc:.4f}")

print("Training finished.")