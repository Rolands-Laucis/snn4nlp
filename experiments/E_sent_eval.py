import argparse
import json
import time
from pathlib import Path
from argparse import Namespace

import snntorch.functional as SF
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from E_sent_model import SequenceSentimentSNN
from readers import ReadSENTInputFile
from snn_util import spike_encode, parse_threshold_layer_scalars
from snn_diagnostics import collect_forward_diagnostics, plot_layer_spike_trains, plot_layer_membrane_traces
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAST_INPUT_DIR = PROJECT_ROOT / "input_data" / "cast_sent"


def build_sentiment_samples(samples, embedding_dim):
	x_list = []
	y_list = []

	for sample_idx, sample in enumerate(samples):
		if not isinstance(sample, (list, tuple)) or len(sample) < 2:
			raise ValueError(f"Invalid sample format at index {sample_idx}: expected [sequence_embeddings, binary_label]")

		token_embeddings = sample[0]
		label_value = sample[1]

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
		raise ValueError("No valid samples were produced for sentiment evaluation.")

	X = torch.stack(x_list, dim=0)
	y = torch.tensor(y_list, dtype=torch.long)
	return X, y


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


def compute_classification_loss(loss_fn, ttfs_loss_fn, targets, decoding_method="spike_count", spike_counts=None, ttfs_spk_rec=None):
	if decoding_method == "ttfs":
		if ttfs_spk_rec is None:
			raise ValueError("ttfs_spk_rec is required for TTFS loss")
		return ttfs_loss_fn(ttfs_spk_rec, targets)

	if spike_counts is None:
		raise ValueError("spike_counts is required for spike_count loss")
	return loss_fn(spike_counts, targets)


def get_ttfs_loss(loss_name):
	if loss_name == "ce_temporal_loss":
		return SF.ce_temporal_loss()
	if loss_name == "mse_temporal_loss":
		return SF.mse_temporal_loss()
	raise ValueError("ttfs temporal loss must be one of: ce_temporal_loss, mse_temporal_loss")


def estimate_batch_ac_operations(model, spike_seq):
	"""Estimate AC operations for one batch using the model's feedforward path.
	Assumes the model has a structure of fc1/lif1/fc2/lif2/fc3/lif3 and that spike_seq is [T, B, input_size].
	AC operations are estimated by counting per sample incoming spikes to each layer and multiplying by the number of synapses (output features), assuming all neurons are fully connected between layers.
	"""
	if spike_seq.ndim != 3:
		raise ValueError(f"spike_seq must be rank-3 [T, B, input_size], got shape: {tuple(spike_seq.shape)}")

	if not all(hasattr(model, layer_name) for layer_name in ("fc1", "fc2", "fc3", "lif1", "lif2", "lif3")):
		raise ValueError("Energy estimation expects the SequenceSentimentSNN fc1/lif1/fc2/lif2/fc3/lif3 structure")

	batch_size = spike_seq.shape[1]
	device = spike_seq.device
	dtype = torch.float32
	running_ac_ops = torch.zeros(batch_size, device=device, dtype=dtype)
	steps = spike_seq.shape[0]

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
		for step in range(steps):
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
	"""Return per-sample AC-operation and energy estimates for one batch."""
	per_sample_ac_ops = estimate_batch_ac_operations(model, spike_seq)
	per_sample_energy_pj = per_sample_ac_ops * float(eac_pj)
	return per_sample_ac_ops, per_sample_energy_pj


def load_model_from_checkpoint(model_path, device):
	checkpoint = torch.load(model_path, map_location=device)
	if "model_state_dict" not in checkpoint or "model_config" not in checkpoint:
		raise ValueError("Checkpoint is missing required keys: model_state_dict/model_config")

	model_config = checkpoint["model_config"]
	cli_args = checkpoint.get("cli_args", {})

	threshold_layer_scalars = parse_threshold_layer_scalars(cli_args.get("threshold_layer_scalars"))
	model = SequenceSentimentSNN(
		input_size=int(model_config["input_size"]),
		hidden_size_1=int(model_config["hidden_size_1"]),
		hidden_size_2=int(model_config["hidden_size_2"]),
		output_size=int(model_config["output_size"]),
		neuron_model_name=model_config.get("neuron_model", cli_args.get("neuron_model", "lif")),
		# these params shouldnt matter here, because its loading the state dict anyway:
		beta=model_config.get("beta", cli_args.get("beta")),
		alpha=model_config.get("alpha", cli_args.get("alpha")),
		threshold=cli_args.get("threshold"),
		threshold_layer_scalars=threshold_layer_scalars,
	)
	model.load_state_dict(checkpoint["model_state_dict"])
	model = model.to(device)
	model.eval()
	return model, checkpoint


def evaluate_batches(
	model,
	features,
	labels,
	batch_size,
	device,
	n_steps,
	input_mode,
	encoding_method,
	decoding_method,
	loss_fn,
	ttfs_loss_fn,
	estimate_energy=False,
	eac_pj=25.63,
):
	eval_ds = TensorDataset(features, labels)
	eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

	running_loss = 0.0
	running_correct = 0
	running_total = 0
	running_fallback = 0
	running_first_spike_time_sum = 0.0
	running_first_spike_time_count = 0
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

	avg_loss = running_loss / max(1, running_total)
	avg_acc = running_correct / max(1, running_total)
	fallback_rate = running_fallback / max(1, running_total)
	mean_first_spike_time = running_first_spike_time_sum / max(1, running_first_spike_time_count)
	avg_ac_ops = running_ac_ops / max(1, running_total) if estimate_energy else None
	avg_energy_pj = running_energy_pj / max(1, running_total) if estimate_energy else None
	return avg_loss, avg_acc, fallback_rate, mean_first_spike_time, avg_ac_ops, avg_energy_pj


def evaluate_model(args:Namespace) -> dict:
	if args.limit is not None and args.limit <= 0:
		raise ValueError("--limit must be a positive integer when provided")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	provided_model = getattr(args, "model", None)
	provided_x_data = getattr(args, "x_data", None)
	provided_y_data = getattr(args, "y_data", None)
	model_config = getattr(args, "model_config", {}) or {}
	cli_args = getattr(args, "cli_args", {}) or {}
	checkpoint = getattr(args, "checkpoint", {}) or {}
	model_path = getattr(args, "model_path", None)

	if provided_model is not None:
		if provided_x_data is None or provided_y_data is None:
			raise ValueError("When args.model is provided, args.x_data and args.y_data are also required")
		model = provided_model.to(device)
		model.eval()
		X_eval = provided_x_data
		y_eval = provided_y_data
		if isinstance(model_config, dict) and not model_config and hasattr(model, "fc1") and hasattr(model.fc1, "in_features"):
			model_config = {
				"input_size": int(model.fc1.in_features),
				"hidden_size_1": int(model.fc1.out_features),
				"hidden_size_2": int(model.fc2.out_features),
				"output_size": int(model.fc3.out_features),
			}
	else:
		if model_path is None:
			raise ValueError("Either args.model or args.model_path must be provided")
		model_path = Path(model_path)
		if not model_path.exists():
			raise FileNotFoundError(f"Checkpoint not found: {model_path}")

		model, checkpoint = load_model_from_checkpoint(model_path, device)
		model_config = checkpoint["model_config"]
		cli_args = checkpoint.get("cli_args", {})

	input_mode = (args.input_mode or model_config.get("input_mode") or "spatial").lower()
	encoding_method = (args.encoding_method or model_config.get("encoding_method") or "poisson").lower()
	decoding_method = (args.decoding_method or model_config.get("decoding_method") or "spike_count").lower()
	ttfs_temporal_loss_name = (
		args.ttfs_temporal_loss
		or cli_args.get("ttfs_temporal_loss")
		or "ce_temporal_loss"
	).lower()
	estimate_energy = getattr(args, "estimate_energy", False)
	eac_pj = getattr(args, "energy_ac_cost_pj", None)

	batch_size = args.batch_size if args.batch_size is not None else int(cli_args.get("batch_size", 32))
	sim_steps = args.sim_steps if args.sim_steps is not None else int(model_config.get("sim_steps", 20))

	if provided_model is None:
		split_file = CAST_INPUT_DIR / f"{args.input_file_prefix}_{args.split}.pkl"
		sent_data, embedding_dim, emb_normalization_mode = ReadSENTInputFile(split_file, limit=args.limit)
		X_eval, y_eval = build_sentiment_samples(sent_data, embedding_dim)
	else:
		X_eval = provided_x_data
		y_eval = provided_y_data
		if args.limit is not None:
			X_eval = X_eval[: args.limit]
			y_eval = y_eval[: args.limit]
		if not isinstance(X_eval, torch.Tensor):
			X_eval = torch.as_tensor(X_eval)
		if not isinstance(y_eval, torch.Tensor):
			y_eval = torch.as_tensor(y_eval)
		embedding_dim = int(X_eval.shape[-1]) if X_eval.ndim >= 2 else None

	# Optional diagnostics: plot spike rasters and output-layer membrane for first sample
	if getattr(args, "diagnose", False):
		first_x = X_eval[:1]
		spike_seq = spike_encode(first_x, sim_steps, input_mode=input_mode, encoding_method=encoding_method).to(device)
		diagnostics = collect_forward_diagnostics(model, spike_seq)
		# spike raster
		spike_fig, _ = plot_layer_spike_trains(diagnostics, sample_index=0, input_spikes=spike_seq, model_name=getattr(args, "diagnose_title", ""))
		# output layer membrane traces
		output_layer_name = list(diagnostics.keys())[-1]
		mem_fig, _ = plot_layer_membrane_traces(diagnostics, layer_name=output_layer_name, sample_index=0)
		# apply optional title
		# diagnose_title = getattr(args, "diagnose_title", None)
		# if diagnose_title:
		# 	try:
		# 		spike_fig.suptitle(diagnose_title)
		# 	except Exception:
		# 		pass
		# 	try:
		# 		mem_fig.suptitle(diagnose_title)
		# 	except Exception:
		# 		pass
		plt.show()
		return

	if provided_model is None and int(model_config.get("embedding_dim", embedding_dim)) != int(embedding_dim):
		raise ValueError(
			f"Embedding dim mismatch between data and checkpoint: data={embedding_dim}, checkpoint={model_config.get('embedding_dim')}"
		)

	sequence_length = int(X_eval.shape[1])
	expected_input_size = sequence_length * embedding_dim if input_mode == "spatial" else embedding_dim
	checkpoint_input_size = int(model_config.get("input_size", expected_input_size))
	if provided_model is None and expected_input_size != checkpoint_input_size:
		raise ValueError(
			"Input size mismatch between evaluation data and checkpoint config: "
			f"data={expected_input_size}, checkpoint={checkpoint_input_size}"
		)

	loss_fn = nn.CrossEntropyLoss()
	ttfs_loss_fn = get_ttfs_loss(ttfs_temporal_loss_name)

	eval_start = time.perf_counter()
	test_loss, test_acc, test_ttfs_fallback_rate, test_ttfs_mean_first_spike_time, test_avg_ac_ops, test_avg_energy_pj = evaluate_batches(
		model,
		X_eval,
		y_eval,
		batch_size,
		device,
		sim_steps,
		input_mode,
		encoding_method,
		decoding_method,
		loss_fn,
		ttfs_loss_fn,
		estimate_energy=estimate_energy,
		eac_pj=eac_pj or None,
	)
	eval_time_ms = (time.perf_counter() - eval_start) * 1000.0

	results = dict(model_config) | {
		"model_source": "args.model" if provided_model is not None else "model_path",
		"model_path": str(model_path) if model_path is not None else None,
		"samples": int(X_eval.shape[0]),
		"batch_size": int(batch_size),
		"eval_time_ms": float(eval_time_ms),
		"eval_loss": float(test_loss),
		"eval_accuracy": float(test_acc),
		"ttfs_fallback_rate": float(test_ttfs_fallback_rate),
		"ttfs_mean_first_spike_time": float(test_ttfs_mean_first_spike_time),
		"emb_normalization_mode": emb_normalization_mode if provided_model is None else None,
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
	if decoding_method == "ttfs":
		print(f"TTFS fallback rate: {results['ttfs_fallback_rate']:.4f}")
		print(f"TTFS mean first spike time (fired output neurons): {results['ttfs_mean_first_spike_time']:.4f}")
	if estimate_energy:
		print(f"Average AC operations per sample: {results['avg_ac_operations_per_sample']:.2f}")
		print(f"Average energy per sample: {results['avg_energy_pj_per_sample']:.2f} pJ ({results['avg_energy_nj_per_sample']:.4f} nJ)")

	if getattr(args, "output_json", False):
		output_json = Path(args.output_json)
		output_json.parent.mkdir(parents=True, exist_ok=True)
		with open(output_json, "w", encoding="utf-8") as handle:
			json.dump(results, handle, indent=2)
		print(f"Saved evaluation results to {output_json}")

	return results


def main():
	parser = argparse.ArgumentParser(description="Evaluate a saved SNN sentiment model checkpoint")
	parser.add_argument("--model_path", type=str, required=True, help="Path to a saved .pt checkpoint")
	parser.add_argument("--input_file_prefix", type=str, default="sent_d50", help="Prefix for cast sentiment input files")
	parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to evaluate")
	parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for quick evaluations")
	parser.add_argument("--batch_size", type=int, default=None, help="Batch size override (defaults to checkpoint cli arg)")
	parser.add_argument("--sim_steps", type=int, default=None, help="Simulation steps override (defaults to checkpoint model config)")
	parser.add_argument("--input_mode", type=str, default="spatial", choices=["spatial", "temporal"], help="Input mode override")
	parser.add_argument("--encoding_method", type=str, default="poisson", choices=["poisson", "latency"], help="Encoding method override")
	parser.add_argument("--decoding_method", type=str, default="spike_count", choices=["spike_count", "ttfs"], help="Decoding method override")
	parser.add_argument("--ttfs_temporal_loss", type=str, default="ce_temporal_loss", choices=["ce_temporal_loss", "mse_temporal_loss"], help="TTFS temporal loss override")
	parser.add_argument("--estimate_energy", action="store_true", help="Estimate average AC operations and energy per tested sample")
	parser.add_argument("--energy_ac_cost_pj", type=float, default=25.63, help="Energy cost of one AC operation in pJ (hardware-dependent)")
	parser.add_argument("--diagnose_title", type=str, default=None, help="Optional title for diagnostic plots")
	parser.add_argument("--diagnose", action="store_true", help="Show spike trains and output-layer membrane trace for first sample")
	parser.add_argument("--output_json", type=str, default=None, help="Optional path to save evaluation results as JSON")
	args = parser.parse_args()

	# If not provided, place it next to the model checkpoint with a related name.
	if not args.output_json:
		args.output_json = Path(args.model_path).parent / f"eval_{Path(args.model_path).stem}.json"

	evaluate_model(args)


if __name__ == "__main__":
	main()
