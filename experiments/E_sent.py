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
from torch.utils.data import DataLoader, TensorDataset

from E_sent_eval import build_sentiment_samples, decode_predictions, compute_classification_loss
from E_sent_model import SequenceSentimentSNN
from snn_util import spike_encode, get_neuron_beta_values_by_layer
from readers import ReadSENTInputFile
from snn_diagnostics import collect_forward_diagnostics
from snn_diagnostics import plot_layer_membrane_traces
from snn_diagnostics import plot_layer_spike_trains

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAST_INPUT_DIR = PROJECT_ROOT / "input_data" / "cast_sent"


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
parser.add_argument("--threshold_layer_scalars", type=str, default="[1, 0.8, 0.7]", help="Leaky neuron threshold scale for each layer (0..1 recommended)")
parser.add_argument("--learn_threshold", type=bool, default=False, help="Whether to learn the threshold parameter")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--save", action="store_true", help="Whether to save the model checkpoint")
parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "output_results" / "E_sent"), help="Output directory for checkpoint and metadata")
parser.add_argument("--diagnose", action="store_true", help="Run first-batch SNN diagnostics and generate plots")
parser.add_argument(
    "--diagnose_dir",
    type=str,
    default=str(PROJECT_ROOT / "output_results" / "E_sent" / "diagnostics"),
    help="Directory for exported diagnostics figures",
)
args = parser.parse_args()

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

args.threshold_layer_scalars = list(map(float, map(str.strip, args.threshold_layer_scalars.strip("[]").split(","))))
# print(f"Parsed threshold_layer_scalars: {args.threshold_layer_scalars}")


def save_training_metadata(metadata_path, metadata):
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


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
net = SequenceSentimentSNN(
    input_size,
    args.num_hidden_1,
    args.num_hidden_2,
    num_labels,
    neuron_model,
    beta=args.beta,
    alpha=args.alpha,
    threshold=args.threshold,
    threshold_layer_scalars=args.threshold_layer_scalars,
)

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

total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

output_dir = Path(args.output_dir) or PROJECT_ROOT / "output_results" / "E_sent"
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

training_start_date = datetime.now()
metadata_file = output_dir / f"{run_filename_base}.json"
training_metadata = {
    "training_config": {
        "training_start_date": training_start_date.strftime("%Y-%m-%d %H:%M:%S"),
        "training_end_date": None,
        "training_duration_s": None,
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
        "learn_beta": bool(args.learn_beta),
        "learn_threshold": bool(args.learn_threshold),
    } | {k: str(v) for k, v in vars(args).items()},
    "results": {
        "epoch_train_loss": [],
        "epoch_train_accuracy": [],
        "epoch_ttfs_fallback_rate": [],
        "epoch_ttfs_mean_first_spike_time": [],
        "test_loss": None,
        "test_accuracy": None,
        "test_ttfs_fallback_rate": None,
        "test_ttfs_mean_first_spike_time": None,
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
print(f"  Decoding method: {decoding_method}")
print(f"  Loss: {ttfs_loss_fn.__class__.__name__ if decoding_method == 'ttfs' else loss_fn.__class__.__name__}")
print(f"  Neuron model: {neuron_model}")
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
    if decoding_method == "ttfs":
        epoch_ttfs_fallback_rates.append(float(epoch_fallback_rate))
        epoch_ttfs_mean_first_spike_times.append(float(epoch_mean_first_spike_time))

    training_metadata["results"]["epoch_train_loss"] = epoch_losses
    training_metadata["results"]["epoch_train_accuracy"] = epoch_accuracies
    if decoding_method == "ttfs":
        training_metadata["results"]["epoch_ttfs_fallback_rate"] = epoch_ttfs_fallback_rates
        training_metadata["results"]["epoch_ttfs_mean_first_spike_time"] = epoch_ttfs_mean_first_spike_times
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
    if decoding_method == "ttfs":
        print(f"TTFS fallback rate: {epoch_fallback_rate:.4f}")
        print(f"TTFS mean first spike time (fired output neurons): {epoch_mean_first_spike_time:.4f}")

print("Training finished.")
if args.diagnose:
    exit(0)

learned_beta_values_by_layer = None
if args.learn_beta:
    learned_beta_values_by_layer = get_neuron_beta_values_by_layer(net) if args.learn_beta else None
    training_metadata["results"]["learned_beta_values_by_layer"] = learned_beta_values_by_layer


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
            "test_loss": training_metadata["results"]["test_loss"],
            "test_accuracy": training_metadata["results"]["test_accuracy"],
            "test_ttfs_fallback_rate": training_metadata["results"]["test_ttfs_fallback_rate"],
            "test_ttfs_mean_first_spike_time": training_metadata["results"]["test_ttfs_mean_first_spike_time"],
            "learned_beta_values_by_layer": learned_beta_values_by_layer,
        },
        "cli_args": vars(args),
    }
    torch.save(checkpoint, model_output_path)
    print(f"Model checkpoint saved to {model_output_path}")

training_end_date = datetime.now()
training_metadata["training_config"]["training_end_date"] = training_end_date.strftime("%Y-%m-%d %H:%M:%S")
training_metadata["training_config"]["training_duration_s"] = (training_end_date - training_start_date).total_seconds()
save_training_metadata(metadata_file, training_metadata)
print(f"\nTraining metadata exported to {metadata_file}")