import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from E_pos_eval import build_pos_samples, decode_predictions, compute_classification_loss, evaluate_model
from E_pos_model import SequencePOS_SNN
from snn_util import spike_encode, get_neuron_beta_values_by_layer
from readers import ReadUPOSInputFile
from snn_diagnostics import collect_forward_diagnostics
from snn_diagnostics import plot_layer_membrane_traces
from snn_diagnostics import plot_layer_spike_trains

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAST_INPUT_DIR = PROJECT_ROOT / "input_data" / "cast_pos"

CONTEXT_WINDOW_SIZE = 5
DECODING_METHOD = "spike_count"
NEURON_MODEL = "synaptic"

parser = argparse.ArgumentParser(description="Train an SNN for token-level POS tagging")
parser.add_argument("--input_mode", type=str, default="spatial", choices=["spatial", "temporal"], help="Input mode for the SNN [spatial|temporal]")
parser.add_argument("--limit", type=int, default=None, help="Limit sentence count after dataset preparation (applied separately to train and test)")
parser.add_argument("--input_file_prefix", type=str, default="pos_d100", help="Prefix for input files")
parser.add_argument("--output_file_prefix", type=str, default="", help="Prefix for output files")
parser.add_argument("--num_hidden_1", type=int, default=256, help="Number of neurons in first hidden layer")
parser.add_argument("--num_hidden_2", type=int, default=128, help="Number of neurons in second hidden layer")
parser.add_argument("--sim_steps", type=int, default=20, help="Poisson/SNN simulation steps")
parser.add_argument("--encoding_method", type=str, default="poisson", choices=["poisson", "latency"], help="Spike encoding method [poisson|latency]")
parser.add_argument("--per_neuron_params", type=bool, default=False, help="Whether to learn parameters for each neuron individually")
parser.add_argument("--alpha", type=float, default=None, help="Synaptic decay factor for second-order neurons; defaults to beta when omitted")
parser.add_argument("--learn_alpha", type=bool, default=False, help="Whether to learn the alpha parameter")
parser.add_argument("--beta", type=float, default=None, help="Leaky neuron decay factor. None for learning or random init (0..1 recommended)")
parser.add_argument("--learn_beta", type=bool, default=False, help="Whether to learn the beta parameter")
parser.add_argument("--threshold", type=float, default=None, help="Leaky neuron threshold factor. None for learning or random init (0..1 recommended)")
parser.add_argument("--threshold_layer_scalars", type=str, default="[1, 0.8, 0.7]", help="Leaky neuron threshold scale for each layer (0..1 recommended)")
parser.add_argument("--learn_threshold", type=bool, default=False, help="Whether to learn the threshold parameter")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--save", action="store_true", help="Whether to save the model checkpoint")
parser.add_argument("--eval", action="store_true", help="Whether to evaluate the model")
parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "output_results" / "E_pos"), help="Output directory for checkpoint and metadata")
parser.add_argument("--diagnose", action="store_true", help="Run first-batch SNN diagnostics and generate plots")
parser.add_argument(
    "--diagnose_dir",
    type=str,
    default=str(PROJECT_ROOT / "output_results" / "E_pos" / "diagnostics"),
    help="Directory for exported diagnostics figures",
)
args = parser.parse_args()

input_mode = args.input_mode.lower()
if input_mode not in {"spatial", "temporal"}:
    raise ValueError("--input_mode must be either 'spatial' or 'temporal'")
if args.limit is not None and args.limit <= 0:
    raise ValueError("--limit must be a positive integer when provided")

encoding_method = args.encoding_method.lower()
alpha = args.alpha if args.alpha is not None else args.beta

args.threshold_layer_scalars = list(map(float, map(str.strip, args.threshold_layer_scalars.strip("[]").split(","))))


def save_training_metadata(metadata_path, metadata):
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


sent_train_data, embedding_dim = ReadUPOSInputFile(CAST_INPUT_DIR / f"{args.input_file_prefix}_train.pkl", limit=args.limit)
sent_test_data, _ = ReadUPOSInputFile(CAST_INPUT_DIR / f"{args.input_file_prefix}_test.pkl", limit=args.limit)

def collect_pos_tags(sentences):
    tags = set()
    for sentence in sentences:
        for token in sentence:
            if len(token) > 1:
                tags.add(token[1])
    return sorted(tags)

pos_tags = collect_pos_tags(sent_train_data + sent_test_data)
label_to_idx = {tag: i for i, tag in enumerate(pos_tags)}
idx_to_label = {i: tag for tag, i in label_to_idx.items()}
num_labels = len(label_to_idx)

# print('building training samples...')
X_train, y_train = build_pos_samples(sent_train_data, embedding_dim, label_to_idx, window_size=CONTEXT_WINDOW_SIZE)
# print('building test samples...')
X_test, y_test = build_pos_samples(sent_test_data, embedding_dim, label_to_idx, window_size=CONTEXT_WINDOW_SIZE)

# sanity check
# sample_idx = 0
# sample_x = X_train[sample_idx]  # [window_size, embedding_dim]
# sample_y = y_train[sample_idx].item()
# print("Sample label:", idx_to_label[sample_y])
# print("Window shape:", sample_x.shape)
# for pos in range(sample_x.shape[0]):
#     print(f"pos {pos}: first 8 dims = {sample_x[pos, :8].tolist()}")
# exit(0)

if args.limit is not None:
    train_limit = min(args.limit, X_train.shape[0])
    test_limit = min(args.limit, X_test.shape[0])
    X_train = X_train[:train_limit]
    y_train = y_train[:train_limit]
    X_test = X_test[:test_limit]
    y_test = y_test[:test_limit]
    # print(f"Applied sample limit: train={train_limit}, test={test_limit}")

sequence_length = X_train.shape[1]

train_class_counts = torch.bincount(y_train, minlength=num_labels)
test_class_counts = torch.bincount(y_test, minlength=num_labels)

# compute samples for each label in the training and test sets (after building samples)
# train_samples_per_label = [torch.where(y_train == i)[0] for i in range(num_labels)]
# test_samples_per_label = [torch.where(y_test == i)[0] for i in range(num_labels)]

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"X_train shape: {X_train.shape}  # [samples, window_len, embed_dim]")
print(f"y_train shape: {y_train.shape}  # [samples]")
print(f"POS tag count: {num_labels}")
print(f"Class count (train): {len(train_class_counts.tolist())}")
print(f"Class count (test): {len(test_class_counts.tolist())}")
print(f"POS tags: {pos_tags}")
print(f"Train class distribution: {train_class_counts.tolist()}")
print(f"Test class distribution: {test_class_counts.tolist()}")

input_size = sequence_length * embedding_dim if input_mode == "spatial" else embedding_dim
net = SequencePOS_SNN(
    input_size,
    args.num_hidden_1,
    args.num_hidden_2,
    num_labels,
    beta=args.beta,
    alpha=alpha,
    learn_alpha=args.learn_alpha,
    learn_beta=args.learn_beta,
    threshold=args.threshold,
    threshold_layer_scalars=args.threshold_layer_scalars,
    per_neuron_params=args.per_neuron_params,
    learn_threshold=args.learn_threshold,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

class_weights = torch.zeros(num_labels)
for i in range(num_labels):
    class_weights[i] = X_train.shape[0] / (num_labels * train_class_counts[i])
loss_fn = nn.CrossEntropyLoss(weight=class_weights) # Use class weights to handle class imbalance in the training data
optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

output_dir = Path(args.output_dir) or PROJECT_ROOT / "output_results" / "E_pos"
output_dir.mkdir(parents=True, exist_ok=True)

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_filename_base = "_".join(
    [
        args.output_file_prefix or "pos",
        now,
        f"e-{args.epochs}",
        f"s-{args.sim_steps}",
        input_mode,
    ]
)

training_start_date = datetime.now()
metadata_file = output_dir / f"{run_filename_base}.json"
training_metadata = {
    "training_config": {
        "training_start_date": training_start_date.strftime("%Y-%m-%d %H:%M:%S"),
        "training_end_date": None,
        "training_duration_s": None,
        "task": "token_level_pos",
        "embedding_dim": int(embedding_dim),
        "sequence_length": int(sequence_length),
        "input_size": int(input_size),
        "num_labels": num_labels,
        "num_training_samples": int(X_train.shape[0]),
        "num_test_samples": int(X_test.shape[0]),
        "device": str(device),
        "total_params": total_params,
        "learn_beta": bool(args.learn_beta),
        "learn_threshold": bool(args.learn_threshold),
        "decoding_method": DECODING_METHOD,
        "neuron_model": NEURON_MODEL,
    } | {k: str(v) for k, v in vars(args).items()},
    "results": {
        "epoch_train_loss": [],
        "epoch_train_accuracy": [],
        "test_loss": None,
        "test_accuracy": None,
        "diagnostics_enabled": bool(args.diagnose),
        "learned_beta_values_by_layer": None,
    },
}
save_training_metadata(metadata_file, training_metadata)

diagnose_dir = Path(args.diagnose_dir)
diagnostics_ran = False
if args.diagnose:
    diagnose_dir.mkdir(parents=True, exist_ok=True)
    print(f"Diagnostics enabled. Export directory: {diagnose_dir}")

print("\nTraining config:")
print(f"  Device: {device}")
print(f"  Input mode: {input_mode}")
print(f"  Encoding method: {encoding_method}")
print(f"  Decoding method: {DECODING_METHOD}")
print(f"  Neuron model: {NEURON_MODEL}")
print(f"  Beta: {args.beta}")
print(f"  Threshold: {args.threshold}")
print(f"  Threshold layer scalars: {args.threshold_layer_scalars}")
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
print(f"  Total learnable parameters: {total_params}")
print(f"  Save: {args.save}")
print(f"  Eval: {args.eval}")

# Train
epoch_losses = []
epoch_accuracies = []
progress_print_every_samples = 10_000
net.train()
training_start_time = time.perf_counter()

for epoch in range(args.epochs):
    epoch_start_time = time.perf_counter()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
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
        # print(f"Spike sequence shape: {spike_seq.shape}  # [sim_steps, batch_size, input_size]")
        # tmp = spike_seq[:, 0, :]
        # a = tmp.cpu().numpy()[0]
        # for j,i in enumerate(tmp[1:]):
        #     print(j, a == i.cpu().numpy())
        # exit(0)

        if args.diagnose and not diagnostics_ran:
            with torch.no_grad():
                diagnostics = collect_forward_diagnostics(net, spike_seq)

            sample_index = 0
            diag_filename = "_".join(
                [
                    args.output_file_prefix or "pos",
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

            output_layer_name = list(diagnostics.keys())[-1]
            output_mem_fig, _ = plot_layer_membrane_traces(
                diagnostics,
                layer_name=output_layer_name,
                sample_index=sample_index,
            )
            output_mem_path = diagnose_dir / f"{diag_filename}_{output_layer_name}_membranes.png"
            output_mem_fig.savefig(output_mem_path, dpi=200, bbox_inches="tight")

            plt.close("all")

            print(f"Diagnostics completed for first training batch. Output layer: {output_layer_name}")
            diagnostics_ran = True

            if epoch + 1 == args.epochs:
                exit(0)

        spike_counts = net(spike_seq)
        loss = compute_classification_loss(loss_fn, yb, spike_counts=spike_counts)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        preds, _ = decode_predictions(spike_counts)
        running_correct += (preds == yb).sum().item()
        running_total += xb.size(0)

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
    epoch_losses.append(float(epoch_loss))
    epoch_accuracies.append(float(epoch_acc))

    training_metadata["results"]["epoch_train_loss"] = epoch_losses
    training_metadata["results"]["epoch_train_accuracy"] = epoch_accuracies
    save_training_metadata(metadata_file, training_metadata)

    epoch_duration_s = time.perf_counter() - epoch_start_time
    elapsed_s = time.perf_counter() - training_start_time
    avg_epoch_s = elapsed_s / float(epoch + 1)
    remaining_epochs = max(0, args.epochs - (epoch + 1))
    eta_minutes = (avg_epoch_s * remaining_epochs) / 60.0
    print(
        f"Epoch {epoch + 1}/{args.epochs} | loss: {epoch_loss:.4f} | acc: {epoch_acc:.4f} "
        f"| epoch_time_s: {epoch_duration_s:.2f} | eta_min: {eta_minutes:.2f}"
    )

print("Training finished.")
training_end_date = datetime.now()
training_metadata["training_config"]["training_end_date"] = training_end_date.strftime("%Y-%m-%d %H:%M:%S")
training_metadata["training_config"]["training_duration_s"] = (training_end_date - training_start_date).total_seconds()

if args.diagnose:
    exit(0)

learned_beta_values_by_layer = None
if args.learn_beta:
    learned_beta_values_by_layer = get_neuron_beta_values_by_layer(net) if args.learn_beta else None
    training_metadata["results"]["learned_beta_values_by_layer"] = learned_beta_values_by_layer

checkpoint = None
if args.save:
    model_output_path = output_dir / f"{run_filename_base}.pt"
    checkpoint = {
        "model_state_dict": net.state_dict(),
        "model_class": "SequencePOS_SNN",
        "model_config": {
            "input_size": input_size,
            "hidden_size_1": args.num_hidden_1,
            "hidden_size_2": args.num_hidden_2,
            "output_size": num_labels,
            "beta": args.beta,
            "alpha": alpha,
            "input_mode": input_mode,
            "encoding_method": encoding_method,
            "decoding_method": DECODING_METHOD,
            "neuron_model": NEURON_MODEL,
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
            "test_loss": training_metadata["results"]["test_loss"],
            "test_accuracy": training_metadata["results"]["test_accuracy"],
            "learned_beta_values_by_layer": learned_beta_values_by_layer,
        },
        "cli_args": vars(args),
    }
    torch.save(checkpoint, model_output_path)
    print(f"Model checkpoint saved to {model_output_path}")

if args.eval:
    args.model = net
    args.x_data = X_test
    args.y_data = y_test
    args.model_config = training_metadata["training_config"]
    args.cli_args = args
    args.checkpoint = checkpoint
    args.estimate_energy = False
    results = evaluate_model(args)

    training_metadata["results"]["test_loss"] = results["eval_loss"]
    training_metadata["results"]["test_accuracy"] = results["eval_accuracy"]

save_training_metadata(metadata_file, training_metadata)
print(f"\nTraining metadata exported to {metadata_file}")