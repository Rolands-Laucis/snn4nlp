from typing import Any
import json
import time
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from E_pos_model import SequencePOS_SNN
from readers import ReadUPOSInputFile
from snn_util import spike_encode

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAST_INPUT_DIR = PROJECT_ROOT / "input_data" / "cast_pos"


def build_pos_samples(
    sentences: list[list[list[Any]]],
    embedding_dim: int,
    label_to_idx: dict[str, int],
    window_size: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 1")

    pad = window_size // 2
    unk_vec = [0.0] * embedding_dim

    samples = []
    labels = []

    for sentence in sentences:
        for i in range(len(sentence)):
            window = []
            for j in range(i - pad, i + pad + 1):
                if j < 0 or j >= len(sentence):
                    window.append(unk_vec)
                else:
                    token = sentence[j]
                    emb = token[3:] if len(token) > 3 else unk_vec
                    window.append(emb)

            target_tag = sentence[i][1] if len(sentence[i]) > 1 else None
            if target_tag is None or target_tag not in label_to_idx:
                continue

            samples.append(window)
            labels.append(label_to_idx[target_tag])

    X = torch.tensor(samples, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y


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
    batch_size,
    device,
    n_steps,
    input_mode,
    encoding_method,
    loss_fn,
    estimate_energy=False,
    eac_pj=25.63,
):
    eval_ds = TensorDataset(features, labels)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    running_loss = 0.0
    running_correct = 0
    running_total = 0
    running_ac_ops = 0.0
    running_energy_pj = 0.0

    with torch.no_grad():
        for xb, yb in eval_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            spike_seq = spike_encode(
                xb,
                n_steps,
                input_mode=input_mode,
                encoding_method=encoding_method,
            ).to(device)

            if estimate_energy:
                batch_ac_ops, batch_energy_pj = estimate_batch_energy(model, spike_seq, eac_pj)
                running_ac_ops += float(batch_ac_ops.sum().item())
                running_energy_pj += float(batch_energy_pj.sum().item())

            spike_counts = model(spike_seq)
            loss = compute_classification_loss(loss_fn, yb, spike_counts=spike_counts)

            running_loss += loss.item() * xb.size(0)
            preds, _ = decode_predictions(spike_counts)
            running_correct += (preds == yb).sum().item()
            running_total += xb.size(0)

    avg_loss = running_loss / max(1, running_total)
    avg_acc = running_correct / max(1, running_total)
    avg_ac_ops = running_ac_ops / max(1, running_total) if estimate_energy else None
    avg_energy_pj = running_energy_pj / max(1, running_total) if estimate_energy else None
    return avg_loss, avg_acc, avg_ac_ops, avg_energy_pj


def evaluate_model(args: Namespace) -> dict:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer when provided")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = getattr(args, "model", None)
    x_data = getattr(args, "x_data", None)
    y_data = getattr(args, "y_data", None)

    if model is None:
        raise ValueError("args.model, args.x_data, and args.y_data are required for POS evaluation")
    if x_data is None or y_data is None:
        raise ValueError("args.x_data and args.y_data are required")

    model = model.to(device)
    model.eval()

    input_mode = getattr(args, "input_mode", "spatial").lower()
    encoding_method = getattr(args, "encoding_method", "poisson").lower()
    batch_size = getattr(args, "batch_size", 32)
    sim_steps = getattr(args, "sim_steps", 20)
    estimate_energy = getattr(args, "estimate_energy", False)
    eac_pj = getattr(args, "energy_ac_cost_pj", 25.63)

    loss_fn = nn.CrossEntropyLoss()

    eval_start = time.perf_counter()
    test_loss, test_acc, test_avg_ac_ops, test_avg_energy_pj = evaluate_batches(
        model,
        x_data,
        y_data,
        batch_size,
        device,
        sim_steps,
        input_mode,
        encoding_method,
        loss_fn,
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

    return results