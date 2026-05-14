
from typing import Any
import argparse
from argparse import Namespace
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from E_pos_seq_model import SequencePOS_SNN
from E_pos_seq_shared import build_seq_samples as shared_build_seq_samples
from E_pos_seq_shared import evaluate_model as shared_evaluate_model
from readers import ReadUPOSInputFile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAST_INPUT_DIR = PROJECT_ROOT / "input_data" / "cast_pos"


def build_seq_samples(
    sentences: list[list[list[Any]]],
    embedding_dim: int,
    label_to_idx: dict[str, int],
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build sequence samples: keep all sentences and pad shorter ones
    with zero vectors. Returns (X, y, mask) where
    - X: (samples, seq_len, embedding_dim)
    - y: (samples, seq_len) long tensor with label indices (0 if padded)
    - mask: (samples, seq_len) bool tensor where True indicates real token
    """
    samples = []
    labels = []
    masks = []

    for sentence in sentences:
        seq = []
        lab = []
        m = []
        for token in sentence:
            seq.append(token[3:])
            lab.append(label_to_idx.get(token[1], 0))
            m.append(True)

        # pad
        while len(seq) < seq_len:
            seq.append([0.0] * embedding_dim)
            lab.append(0)
            m.append(False)

        samples.append(seq)
        labels.append(lab)
        masks.append(m)

    X = torch.tensor(samples, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    mask = torch.tensor(masks, dtype=torch.bool)
    return X, y, mask


def decode_predictions(spike_counts: torch.Tensor) -> tuple[torch.Tensor, int]:
    preds = torch.argmax(spike_counts, dim=1)
    return preds, 0


def compute_classification_loss(loss_fn, y_true: torch.Tensor, spike_counts: torch.Tensor) -> torch.Tensor:
    return loss_fn(spike_counts, y_true)

def estimate_batch_ac_operations(model, spike_seq):
    """
    Estimate AC operations for one batch using the model's feedforward path.

    Assumes the model has fc1/lif1/fc2/lif2/fc3/lif3 and spike_seq is [T, B, input_size].
    AC operations are estimated by counting per-sample incoming spikes to each layer
    and multiplying by the number of synapses (output features), assuming fully
    connected layers.
    """
    if spike_seq.ndim != 3:
        raise ValueError(f"spike_seq must be rank-3 [T, B, input_size], got {tuple(spike_seq.shape)}")

    if not all(hasattr(model, name) for name in ("fc1", "fc2", "fc3", "lif1", "lif2", "lif3")):
        raise ValueError("Energy estimation expects fc1/lif1/fc2/lif2/fc3/lif3 model structure")

    batch_size = spike_seq.shape[1]
    device = spike_seq.device
    dtype = torch.float32
    running_ac_ops = torch.zeros(batch_size, device=device, dtype=dtype)

    neuron_class1 = model.lif1.__class__.__name__
    neuron_class2 = model.lif2.__class__.__name__
    neuron_class3 = model.lif3.__class__.__name__

    mem1 = torch.zeros(batch_size, model.fc1.out_features, device=device, dtype=spike_seq.dtype)
    mem2 = torch.zeros(batch_size, model.fc2.out_features, device=device, dtype=spike_seq.dtype)
    mem3 = torch.zeros(batch_size, model.fc3.out_features, device=device, dtype=spike_seq.dtype)

    syn1 = None
    syn2 = None
    syn3 = None
    if neuron_class1 in ("Synaptic", "QLIF"):
        syn1 = torch.zeros(batch_size, model.fc1.out_features, device=device, dtype=spike_seq.dtype)
    if neuron_class2 in ("Synaptic", "QLIF"):
        syn2 = torch.zeros(batch_size, model.fc2.out_features, device=device, dtype=spike_seq.dtype)
    if neuron_class3 in ("Synaptic", "QLIF"):
        syn3 = torch.zeros(batch_size, model.fc3.out_features, device=device, dtype=spike_seq.dtype)

    with torch.no_grad():
        for step in range(spike_seq.shape[0]):
            input_spikes = spike_seq[step]
            running_ac_ops += input_spikes.sum(dim=1).to(dtype) * float(model.fc1.out_features)

            cur1 = model.fc1(input_spikes)
            if neuron_class1 in ("Synaptic", "QLIF"):
                spk1, syn1, mem1 = model.lif1(cur1, syn1, mem1)
            else:
                spk1, mem1 = model.lif1(cur1, mem1)

            running_ac_ops += spk1.sum(dim=1).to(dtype) * float(model.fc2.out_features)

            cur2 = model.fc2(spk1)
            if neuron_class2 in ("Synaptic", "QLIF"):
                spk2, syn2, mem2 = model.lif2(cur2, syn2, mem2)
            else:
                spk2, mem2 = model.lif2(cur2, mem2)

            running_ac_ops += spk2.sum(dim=1).to(dtype) * float(model.fc3.out_features)

            cur3 = model.fc3(spk2)
            if neuron_class3 in ("Synaptic", "QLIF"):
                spk3, syn3, mem3 = model.lif3(cur3, syn3, mem3)
            else:
                spk3, mem3 = model.lif3(cur3, mem3)

    return running_ac_ops


def estimate_batch_energy(model, spike_seq, eac_pj):
    per_sample_ac_ops = estimate_batch_ac_operations(model, spike_seq)
    per_sample_energy_pj = per_sample_ac_ops * float(eac_pj)
    return per_sample_ac_ops, per_sample_energy_pj


def evaluate_batches(
    model,
    features,
    labels,
    masks,
    batch_size,
    device,
    n_steps,
    input_mode,
    encoding_method,
    loss_fn,
    estimate_energy=False,
    eac_pj=25.63,
):
    eval_ds = TensorDataset(features, labels, masks)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    running_loss = 0.0
    running_correct_tokens = 0
    running_total_tokens = 0
    running_ac_ops = 0.0
    running_energy_pj = 0.0

    with torch.no_grad():
        for xb, yb, mb in eval_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)

            batch_size_eff, seq_len_eff, _ = xb.shape
            xb_flat = xb.reshape(batch_size_eff, -1)
            spike_seq = spike_encode(
                xb_flat.unsqueeze(1),
                n_steps,
                input_mode=input_mode,
                encoding_method=encoding_method,
            ).to(device)

            if estimate_energy:
                batch_ac_ops, batch_energy_pj = estimate_batch_energy(
                    model,
                    spike_seq,
                    eac_pj,
                )
                running_ac_ops += float(batch_ac_ops.sum().item())
                running_energy_pj += float(batch_energy_pj.sum().item())

            per_step_spikes = model(spike_seq)
            spike_sum = per_step_spikes.sum(dim=0)
            emissions = spike_sum.view(batch_size_eff, seq_len_eff, -1)

            valid_emissions = emissions[mb]
            valid_labels = yb[mb]
            loss = loss_fn(valid_emissions, valid_labels)

            running_loss += loss.item() * int(mb.sum().item())
            preds = emissions.argmax(dim=-1)
            running_correct_tokens += int(((preds == yb) & mb).sum().item())
            running_total_tokens += int(mb.sum().item())

    avg_loss = running_loss / max(1, running_total_tokens)
    avg_acc = running_correct_tokens / max(1, running_total_tokens)
    avg_ac_ops = running_ac_ops / max(1, running_total_tokens) if estimate_energy else None
    avg_energy_pj = running_energy_pj / max(1, running_total_tokens) if estimate_energy else None
    return avg_loss, avg_acc, avg_ac_ops, avg_energy_pj


def load_model_from_checkpoint(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError("Checkpoint is missing required keys: model_state_dict/model_config")

    model_config = checkpoint["model_config"]
    cli_args = checkpoint.get("cli_args", {})

    threshold_layer_scalars = parse_threshold_layer_scalars(cli_args.get("threshold_layer_scalars"))
    model = SequencePOS_SNN(
        input_size=int(model_config["input_size"]),
        hidden_size_1=int(model_config["hidden_size_1"]),
        hidden_size_2=int(model_config["hidden_size_2"]),
        output_size=int(model_config["output_size"]),
        beta=model_config.get("beta", cli_args.get("beta")),
        alpha=model_config.get("alpha", cli_args.get("alpha")),
        threshold=cli_args.get("threshold"),
        threshold_layer_scalars=threshold_layer_scalars,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint

def evaluate_model(args: Namespace) -> dict:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer when provided")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = getattr(args, "model", None)
    model_path = getattr(args, "model_path", None)
    x_data = getattr(args, "x_data", None)
    y_data = getattr(args, "y_data", None)
    limit = getattr(args, "limit", None)

    if model is None:
        if model_path is None:
            raise ValueError("Either args.model or args.model_path must be provided")
        model, _ = load_model_from_checkpoint(model_path, device)

    model = model.to(device)
    model.eval()

    batch_size = getattr(args, "batch_size", 32)
    estimate_energy = getattr(args, "estimate_energy", False)
    eac_pj = getattr(args, "energy_ac_cost_pj", 25.63)

    print(f"Evaluating model", model_path, 'Limit:', limit)

    # If x/y data are not provided, load cast POS input files and build samples
    if x_data is None or y_data is None:
        input_file_prefix = getattr(args, "input_file_prefix", "pos_d100")
        split = getattr(args, "split", "test")

        train_file = CAST_INPUT_DIR / f"{input_file_prefix}_train.pkl"
        test_file = CAST_INPUT_DIR / f"{input_file_prefix}_test.pkl"
        split_file = CAST_INPUT_DIR / f"{input_file_prefix}_{split}.pkl"
        if not train_file.exists():
            raise FileNotFoundError(f"POS input file not found: {train_file}")
        if not test_file.exists():
            raise FileNotFoundError(f"POS input file not found: {test_file}")
        if not split_file.exists():
            raise FileNotFoundError(f"POS input file not found: {split_file}")

        train_sentences, train_embedding_dim = ReadUPOSInputFile(train_file, limit=None)
        test_sentences, test_embedding_dim = ReadUPOSInputFile(test_file, limit=None)
        sentences, embedding_dim = ReadUPOSInputFile(split_file, limit=None)
        if embedding_dim != train_embedding_dim:
            raise ValueError("Embedding dimension mismatch between train and evaluation split")
        if embedding_dim != test_embedding_dim:
            raise ValueError("Embedding dimension mismatch between test and evaluation split")

        global_max_seq_len = max(
            max((len(s) for s in train_sentences), default=0),
            max((len(s) for s in test_sentences), default=0),
        )
        if global_max_seq_len <= 0:
            raise ValueError("No valid sentences found to infer sequence length")

        tags = set()
        for sent in train_sentences:
            for token in sent:
                if len(token) > 1:
                    tags.add(token[1])
        for sent in sentences:
            for token in sent:
                if len(token) > 1:
                    tags.add(token[1])
        label_to_idx = {tag: i for i, tag in enumerate(sorted(tags))}

        x_data, y_data, masks = build_seq_samples(sentences, embedding_dim, label_to_idx, seq_len=global_max_seq_len)

    eval_start = time.perf_counter()
    masks = getattr(args, "masks", None)
    if masks is None:
        masks = torch.ones_like(y_data, dtype=torch.bool)

    test_loss, test_acc, test_avg_ac_ops, test_avg_energy_pj = evaluate_batches(
        model=model,
        features=x_data,
        labels=y_data,
        masks=masks,
        batch_size=batch_size,
        device=device,
        n_steps=None,
        input_mode=None,
        encoding_method=None,
        loss_fn=None,
        estimate_energy=estimate_energy,
        eac_pj=eac_pj,
    )
    eval_time_ms = (time.perf_counter() - eval_start) * 1000.0

    results = {
        "samples": int(x_data.shape[0]),
        "batch_size": int(batch_size),
        "eval_time_ms": float(eval_time_ms),
        "eval_loss": float(test_loss),
        "eval_accuracy": float(test_acc),
    }

    if estimate_energy:
        results["energy_ac_cost_pj"] = float(eac_pj)
        results["avg_ac_operations_per_sample"] = float(test_avg_ac_ops)
        results["avg_energy_pj_per_sample"] = float(test_avg_energy_pj)
        results["avg_energy_nj_per_sample"] = float(test_avg_energy_pj / 1000.0)

    print(
        f"Evaluation | samples={results['samples']} "
        f"| loss={results['eval_loss']:.4f} | acc={results['eval_accuracy']:.4f} "
        f"| eval_time_ms={results['eval_time_ms']:.2f}"
    )
    if estimate_energy:
        print(f"Average AC operations per sample: {results['avg_ac_operations_per_sample']:.2f}")
        print(f"Average energy per sample: {results['avg_energy_pj_per_sample']:.2f} pJ")

    if getattr(args, "output_json", False):
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Saved evaluation results to {output_json}")

    return results



CONTEXT_WINDOW_SIZE = 5
DECODING_METHOD = "spike_count"
NEURON_MODEL = "synaptic"

parser = argparse.ArgumentParser(description="Train an SNN for token-level POS tagging")
parser.add_argument("--input_mode", type=str, default="spatial", choices=["spatial", "temporal"], help="Input mode for the SNN [spatial|temporal]")
parser.add_argument("--limit", type=int, default=None, help="Limit sentence count after dataset preparation (applied separately to train and test)")
parser.add_argument("--input_file_prefix", type=str, default="pos_d100", help="Prefix for input files")
parser.add_argument("--output_file_prefix", type=str, default="upos_seq", help="Prefix for output files")
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
parser.add_argument("--threshold_layer_scalars", type=str, default="[1, 1, 1]", help="Leaky neuron threshold scale for each layer (0..1 recommended)")
parser.add_argument("--learn_threshold", type=bool, default=False, help="Whether to learn the threshold parameter")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--save", action="store_true", help="Whether to save the model checkpoint")
parser.add_argument("--eval", action="store_true", help="Whether to evaluate the model")
parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "output_results" / "E_pos" / "seq"), help="Output directory for checkpoint and metadata")
parser.add_argument("--diagnose", action="store_true", help="Run first-batch SNN diagnostics and generate plots")
parser.add_argument(
    "--diagnose_dir",
    type=str,
    default=str(PROJECT_ROOT / "output_results" / "E_pos" / "diagnostics"),
    help="Directory for exported diagnostics figures",
)
# eval args
parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to evaluate")
parser.add_argument("--estimate_energy", action="store_true", help="Estimate average AC operations and energy per tested sample")
parser.add_argument("--energy_ac_cost_pj", type=float, default=25.63, help="Energy cost of one AC operation in pJ (hardware-dependent)")
parser.add_argument("--shuffle_context_window", action="store_true", help="Special test condition: shuffle the context window word order after constructing it")
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


sent_train_data, embedding_dim = ReadUPOSInputFile(CAST_INPUT_DIR / f"{args.input_file_prefix}_train.pkl", limit=None)
sent_test_data, _ = ReadUPOSInputFile(CAST_INPUT_DIR / f"{args.input_file_prefix}_test.pkl", limit=None)

# Sequence-to-sequence settings: use global max sentence length across train and test and pad shorter ones
max_train_len = max((len(s) for s in sent_train_data), default=0)
max_test_len = max((len(s) for s in sent_test_data), default=0)
sequence_length = max(max_train_len, max_test_len)
if sequence_length <= 0:
    raise ValueError("No valid sentences found in train/test to infer sequence length")

def collect_pos_tags(sentences):
    tags = set()
    for sentence in sentences:
        for token in sentence:
            if len(token) > 1:
                tags.add(token[1])
    return sorted(tags)

# Build label maps from the filtered sentences
pos_tags = collect_pos_tags(sent_train_data + sent_test_data)
label_to_idx = {tag: i for i, tag in enumerate(pos_tags)}
idx_to_label = {i: tag for tag, i in label_to_idx.items()}
num_labels = len(label_to_idx)

# Build sequence samples (pad to global sequence length). Apply limit after padding.
X_train, y_train, train_mask = shared_build_seq_samples(sent_train_data, embedding_dim, label_to_idx, seq_len=sequence_length)
X_test, y_test, test_mask = shared_build_seq_samples(sent_test_data, embedding_dim, label_to_idx, seq_len=sequence_length)

if args.limit is not None:
    train_limit = min(args.limit, X_train.shape[0])
    test_limit = min(args.limit, X_test.shape[0])
    X_train = X_train[:train_limit]
    y_train = y_train[:train_limit]
    train_mask = train_mask[:train_limit]
    X_test = X_test[:test_limit]
    y_test = y_test[:test_limit]
    test_mask = test_mask[:test_limit]

# token-level class counts (only real tokens)
valid_train_labels = y_train[train_mask]
valid_test_labels = y_test[test_mask]
train_class_counts = torch.bincount(valid_train_labels.flatten(), minlength=num_labels) if valid_train_labels.numel() > 0 else torch.zeros(num_labels, dtype=torch.long)
test_class_counts = torch.bincount(valid_test_labels.flatten(), minlength=num_labels) if valid_test_labels.numel() > 0 else torch.zeros(num_labels, dtype=torch.long)

# compute samples for each label in the training and test sets (after building samples)
# train_samples_per_label = [torch.where(y_train == i)[0] for i in range(num_labels)]
# test_samples_per_label = [torch.where(y_test == i)[0] for i in range(num_labels)]

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"X_train shape: {X_train.shape}  # [samples, seq_len, embed_dim]")
print(f"y_train shape: {y_train.shape}  # [samples, seq_len]")
print(f"POS tag count: {num_labels}")
print(f"Class count (train): {len(train_class_counts.tolist())}")
print(f"Class count (test): {len(test_class_counts.tolist())}")
print(f"POS tags: {pos_tags}")
print(f"Train class distribution: {train_class_counts.tolist()}")
print(f"Test class distribution: {test_class_counts.tolist()}")

# Create SNN+CRF model: processes token embeddings sequentially
net = SequencePOS_SNN(
    emb_dim=embedding_dim,
    hidden_size_1=args.num_hidden_1,
    hidden_size_2=args.num_hidden_2,
    num_tags=num_labels,
    n_steps=args.sim_steps,
    input_mode=input_mode,
    encoding_method=encoding_method,
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

# DataLoader should include masks so we can ignore padded tokens
train_ds = TensorDataset(X_train, y_train, train_mask)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

# Optimize all trainable model parameters
params = list(net.parameters())
optimizer = torch.optim.Adam(params, lr=args.learning_rate)
# CRF loss is computed internally by the model

total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

output_dir = Path(args.output_dir) or PROJECT_ROOT / "output_results" / "E_pos"
output_dir.mkdir(parents=True, exist_ok=True)

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_filename_base = "_".join(
    [
        args.output_file_prefix or "pos",
        now,
        f"e-{args.epochs}",
    ]
)

training_start_date = datetime.now()
metadata_file = output_dir / f"{run_filename_base}.json"
training_metadata = {
    "training_config": {
        "training_start_date": training_start_date.strftime("%Y-%m-%d %H:%M:%S"),
        "training_end_date": None,
        "training_duration_s": None,
        "task": "seq_pos_crf",
        "embedding_dim": int(embedding_dim),
        "sequence_length": int(sequence_length),
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
print(f"  Learn Beta: {args.learn_beta}")
print(f"  Threshold: {args.threshold}")
print(f"  Threshold layer scalars: {args.threshold_layer_scalars}")
print(f"  Learn Threshold: {args.learn_threshold}")
print(f"  Alpha: {alpha}")
print(f"  Learn Alpha: {args.learn_alpha}")
print(f"  Per-neuron params: {args.per_neuron_params}")
print(f"  Sequence length: {sequence_length}")
print(f"  Embedding dim: {embedding_dim}")
print(f"  Input size: {embedding_dim}")
print(f"  Hidden size 1: {args.num_hidden_1}")
print(f"  Hidden size 2: {args.num_hidden_2}")
print(f"  Output classes: {num_labels}")
print(f"  Num steps: {args.sim_steps}")
print(f"  Batch size: {args.batch_size}")
print(f"  Epochs: {args.epochs}")
print(f"  Learning rate: {args.learning_rate}")
print(f"  Total learnable parameters: {total_params}")
print(f'  LOSS: CRF loss computed internally by the model')
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
    running_correct_tokens = 0
    running_total_tokens = 0
    next_progress_print_at = progress_print_every_samples
    epoch_total_samples = len(train_loader.dataset)
    diagnostics_ran = False

    for xb, yb, mb in train_loader:
        xb = xb.to(device)  # (batch, seq_len, emb_dim)
        yb = yb.to(device)  # (batch, seq_len)
        mb = mb.to(device)  # (batch, seq_len)

        batch_size_eff = xb.shape[0]
        
        # Forward pass: CRF loss is computed internally
        loss = net(xb, tags=yb, mask=mb)
        
        # Get predictions via CRF Viterbi decoding
        preds = net(xb, mask=mb)  # list[list[int]]
        
        # Convert predictions to tensor for accuracy
        preds_tensor = torch.zeros_like(yb)
        for i, pred_seq in enumerate(preds):
            pred_len = min(len(pred_seq), sequence_length)
            preds_tensor[i, :pred_len] = torch.tensor(pred_seq[:pred_len], device=device, dtype=torch.long)

        # Compute accuracy on real tokens only
        running_correct_tokens += int(((preds_tensor == yb) & mb).sum().item())
        running_total_tokens += int(mb.sum().item())
        running_loss += (-loss).item() * batch_size_eff

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # progress by tokens
        running_total = min(epoch_total_samples, running_total_tokens)
        while running_total >= next_progress_print_at:
            progress_pct = (running_total / max(1, epoch_total_samples)) * 100.0
            epoch_elapsed_s = time.perf_counter() - epoch_start_time
            print(
                f"Epoch {epoch + 1}/{args.epochs} progress | "
                f"tokens: {running_total}/{epoch_total_samples} ({progress_pct:.2f}%) | "
                f"elapsed_s: {epoch_elapsed_s:.2f}"
            )
            next_progress_print_at += progress_print_every_samples

    epoch_loss = running_loss / max(1, running_total_tokens)
    epoch_acc = running_correct_tokens / max(1, running_total_tokens)
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

checkpoint = None
if args.save:
    model_output_path = output_dir / f"{run_filename_base}.pt"
    checkpoint = {
        "model_state_dict": net.state_dict(),
        "model_class": "SequencePOS_SNN",
        "model_config": {
            "embedding_dim": int(embedding_dim),
            "hidden_size_1": args.num_hidden_1,
            "hidden_size_2": args.num_hidden_2,
            "num_labels": num_labels,
            "n_steps": int(args.sim_steps),
            "input_mode": input_mode,
            "encoding_method": encoding_method,
            "beta": args.beta,
            "alpha": alpha,
            "decoding_method": DECODING_METHOD,
            "neuron_model": NEURON_MODEL,
            "sequence_length": int(sequence_length),
            "embedding_dim": int(embedding_dim),
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
    args.masks = test_mask
    args.model_config = training_metadata["training_config"]
    args.cli_args = args
    args.checkpoint = checkpoint
    args.estimate_energy = False
    results = shared_evaluate_model(args)

    training_metadata["results"]["test_loss"] = results["eval_loss"]
    training_metadata["results"]["test_accuracy"] = results["eval_accuracy"]

save_training_metadata(metadata_file, training_metadata)
print(f"\nTraining metadata exported to {metadata_file}")