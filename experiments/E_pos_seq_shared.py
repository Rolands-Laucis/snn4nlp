from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any
import json
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from E_pos_seq_model import SequencePOS_SNN
from readers import ReadUPOSInputFile
from snn_diagnostics import collect_forward_diagnostics, plot_layer_membrane_traces, plot_layer_spike_trains
from snn_util import parse_threshold_layer_scalars, spike_encode

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAST_INPUT_DIR = PROJECT_ROOT / "input_data" / "cast_pos"


def build_seq_samples(
    sentences: list[list[list[Any]]],
    embedding_dim: int,
    label_to_idx: dict[str, int],
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def estimate_batch_ac_operations(model, x_emb):
    """Estimate AC operations for a batch of embeddings."""
    if x_emb.ndim != 3:
        raise ValueError(f"x_emb must be rank-3 [batch, seq_len, emb_dim], got {tuple(x_emb.shape)}")

    if not all(hasattr(model, name) for name in ("fc1", "fc2", "lif1", "lif2", "linear_out")):
        raise ValueError("Energy estimation expects fc1/lif1/fc2/lif2/linear_out model structure")

    batch_size, seq_len, emb_dim = x_emb.shape
    device = x_emb.device
    dtype = torch.float32
    running_ac_ops = torch.zeros(batch_size, device=device, dtype=dtype)

    # Initialize SNN state
    syn1, mem1 = model.lif1.init_synaptic()
    syn2, mem2 = model.lif2.init_synaptic()

    with torch.no_grad():
        # Process each token sequentially
        for t in range(seq_len):
            x_t = x_emb[:, t, :]
            
            cur1 = model.fc1(x_t)
            spk1, syn1, mem1 = model.lif1(cur1, syn1, mem1)
            running_ac_ops += spk1.sum(dim=1).to(dtype) * float(model.fc2.out_features)

            cur2 = model.fc2(spk1)
            spk2, syn2, mem2 = model.lif2(cur2, syn2, mem2)
            running_ac_ops += spk2.sum(dim=1).to(dtype) * float(model.linear_out.out_features)

    return running_ac_ops

def evaluate_batches(
    model,
    features,
    labels,
    masks,
    batch_size,
    device,
    n_steps=None,
    input_mode=None,
    encoding_method=None,
    loss_fn=None,
    estimate_energy=False,
    eac_pj=25.63,
):
    """
    Evaluate the SNN+CRF model on embeddings (not spike-encoded).
    
    Parameters
    ----------
    features : (N, seq_len, emb_dim)
        Token embeddings
    labels : (N, seq_len)
        Gold tag indices
    masks : (N, seq_len)
        Boolean mask for real tokens
    Legacy parameters (n_steps, input_mode, encoding_method, loss_fn): ignored
    """
    eval_ds = TensorDataset(features, labels, masks)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    running_loss = 0.0
    running_correct_tokens = 0
    running_total_tokens = 0
    running_ac_ops = 0

    with torch.no_grad():
        for xb, yb, mb in eval_loader:
            xb = xb.to(device)  # (batch, seq_len, emb_dim)
            yb = yb.to(device)  # (batch, seq_len)
            mb = mb.to(device)  # (batch, seq_len)

            # Forward pass: CRF loss computed internally
            loss = model(xb, tags=yb, mask=mb)
            running_loss += (-loss).item() * int(yb.shape[0])

            # Get predictions via CRF Viterbi decoding
            preds = model(xb, mask=mb)  # list[list[int]]
            
            # Convert predictions to tensor for accuracy
            preds_tensor = torch.zeros_like(yb)
            for i, pred_seq in enumerate(preds):
                pred_len = min(len(pred_seq), yb.shape[1])
                preds_tensor[i, :pred_len] = torch.tensor(pred_seq[:pred_len], device=device, dtype=torch.long)

            # Compute accuracy on real tokens only
            running_correct_tokens += int(((preds_tensor == yb) & mb).sum().item())
            running_total_tokens += int(mb.sum().item())

            if estimate_energy:
                batch_ac_ops = estimate_batch_ac_operations(model, xb)
                running_ac_ops += int(batch_ac_ops.sum().item())

    avg_loss = running_loss / max(1, running_total_tokens) if running_total_tokens > 0 else 0.0
    avg_acc = running_correct_tokens / max(1, running_total_tokens) if running_total_tokens > 0 else 0.0
    avg_ac_ops = running_ac_ops / max(1, running_total_tokens) if estimate_energy else None
    avg_energy_pj = (avg_ac_ops * eac_pj) if estimate_energy else None
    return avg_loss, avg_acc, avg_ac_ops, avg_energy_pj


def load_model_from_checkpoint(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError("Checkpoint is missing required keys: model_state_dict/model_config")

    model_config = checkpoint["model_config"]
    cli_args = checkpoint.get("cli_args", {})

    def _optional_float(value, default=None):
        if value is None:
            return default
        return float(value)

    # Extract parameters
    embedding_dim = int(model_config.get("embedding_dim", cli_args.get("embedding_dim", 100)))
    
    model = SequencePOS_SNN(
        emb_dim=embedding_dim,
        hidden_size_1=int(model_config.get("hidden_size_1", cli_args.get("num_hidden_1", 256))),
        hidden_size_2=int(model_config.get("hidden_size_2", cli_args.get("num_hidden_2", 128))),
        num_tags=int(model_config.get("num_labels", cli_args.get("num_labels", 17))),
        n_steps=int(model_config.get("n_steps", cli_args.get("sim_steps", 20))),
        input_mode=model_config.get("input_mode", cli_args.get("input_mode", "spatial")),
        encoding_method=model_config.get("encoding_method", cli_args.get("encoding_method", "latency")),
        beta=_optional_float(model_config.get("beta", cli_args.get("beta", 0.5)), 0.5),
        alpha=_optional_float(model_config.get("alpha", cli_args.get("alpha", 0.5)), 0.5),
        threshold=_optional_float(model_config.get("threshold", cli_args.get("threshold")), None),
        learn_alpha=bool(model_config.get("learn_alpha", cli_args.get("learn_alpha", False))),
        learn_beta=bool(model_config.get("learn_beta", cli_args.get("learn_beta", False))),
        learn_threshold=bool(model_config.get("learn_threshold", cli_args.get("learn_threshold", False))),
        per_neuron_params=bool(model_config.get("per_neuron_params", cli_args.get("per_neuron_params", False))),
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
    masks = getattr(args, "masks", None)
    model_config = getattr(args, "model_config", {}) or {}
    cli_args = getattr(args, "cli_args", {}) or {}
    checkpoint = getattr(args, "checkpoint", {}) or {}
    limit = getattr(args, "limit", None)

    if isinstance(model_config, Namespace):
        model_config = vars(model_config)
    if isinstance(cli_args, Namespace):
        cli_args = vars(cli_args)

    if model is None:
        if model_path is None:
            raise ValueError("Either args.model or args.model_path must be provided")
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        model, checkpoint = load_model_from_checkpoint(model_path, device)
        model_config = checkpoint.get("model_config", {})
        cli_args = checkpoint.get("cli_args", {})
    else:
        model = model.to(device)
        model.eval()

    batch_size = getattr(args, "batch_size", None)
    if batch_size is None:
        batch_size = int(model_config.get("batch_size", cli_args.get("batch_size", 32)))
    else:
        try:
            batch_size = int(batch_size)
        except Exception:
            batch_size = int(float(batch_size))

    estimate_energy = getattr(args, "estimate_energy", False)
    eac_pj = getattr(args, "energy_ac_cost_pj", 25.63)
    input_file_prefix = getattr(args, "input_file_prefix", None) or model_config.get("input_file_prefix") or cli_args.get("input_file_prefix") or "pos_d100"
    split = getattr(args, "split", None) or cli_args.get("split", "test")
    split_file = CAST_INPUT_DIR / f"{input_file_prefix}_{split}.pkl"
    if not split_file.exists():
        raise FileNotFoundError(f"POS input file not found: {split_file}")

    sentences, embedding_dim = ReadUPOSInputFile(split_file, limit=limit)
    label_maps = checkpoint.get("label_maps", {}) if isinstance(checkpoint, dict) else {}
    label_to_idx = label_maps.get("label_to_idx") if isinstance(label_maps, dict) else None
    if not label_to_idx:
        tags = set()
        for sent in sentences:
            for token in sent:
                if len(token) > 1:
                    tags.add(token[1])
        label_to_idx = {tag: i for i, tag in enumerate(sorted(tags))}

    seq_len = int(model_config.get("sequence_length") or max((len(sentence) for sentence in sentences), default=0))
    if seq_len <= 0:
        raise ValueError("Unable to derive sequence length for evaluation")

    x_data, y_data, masks = build_seq_samples(sentences, embedding_dim, label_to_idx, seq_len=seq_len)

    if getattr(args, "diagnose", False):
        token_spike_trains = []
        sample_embeddings = x_data[:1].to(device)  # [1, seq_len, emb_dim]
        for token_idx in range(sample_embeddings.shape[1]):
            token_embeddings = sample_embeddings[:, token_idx, :].unsqueeze(1)  # [1, 1, emb_dim]
            token_spikes = spike_encode(
                token_embeddings,
                n_steps=int(model_config.get("n_steps", cli_args.get("sim_steps", 20))),
                input_mode=model_config.get("input_mode", cli_args.get("input_mode", "spatial")),
                encoding_method=model_config.get("encoding_method", cli_args.get("encoding_method", "latency")),
            )
            token_spike_trains.append(token_spikes)

        spike_seq = torch.cat(token_spike_trains, dim=0)
        diagnostics = collect_forward_diagnostics(model, spike_seq)
        plot_layer_spike_trains(diagnostics, sample_index=0, input_spikes=spike_seq)
        output_layer_name = list(diagnostics.keys())[-1]
        plot_layer_membrane_traces(diagnostics, layer_name=output_layer_name, sample_index=0)
        plt.show()
        return {}

    if masks is None:
        masks = torch.ones_like(y_data, dtype=torch.bool)

    eval_start = time.perf_counter()
    test_loss, test_acc, test_avg_ac_ops, test_avg_energy_pj = evaluate_batches(
        model=model,
        features=x_data,
        labels=y_data,
        masks=masks,
        batch_size=batch_size,
        device=device,
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

    output_json = getattr(args, "output_json", None)
    if output_json:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Saved evaluation results to {output_json}")

    return results
