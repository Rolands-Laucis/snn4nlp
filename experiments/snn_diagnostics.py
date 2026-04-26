from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch


@dataclass
class LayerDiagnostics:
    name: str
    spikes: torch.Tensor  # [T, B, N]
    membrane: torch.Tensor  # [T, B, N]
    threshold: float


def _resolve_threshold(neuron_layer: torch.nn.Module, default: float = 1.0) -> float:
    threshold = getattr(neuron_layer, "threshold", default)
    if isinstance(threshold, torch.Tensor):
        if threshold.numel() == 0:
            return default
        return float(threshold.detach().flatten()[0].cpu().item())
    try:
        return float(threshold)
    except (TypeError, ValueError):
        return default


def _step_neuron_layer(
    neuron_layer: torch.nn.Module,
    current: torch.Tensor,
    syn_state: torch.Tensor | None,
    mem_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Run one neuron layer step and normalize outputs across snnTorch variants."""
    class_name = neuron_layer.__class__.__name__

    if class_name in {"Synaptic", "QLIF"}:
        out = neuron_layer(current, syn_state, mem_state)
    else:
        out = neuron_layer(current, mem_state)

    if isinstance(out, tuple):
        if len(out) == 3:
            spk, syn_next, mem_next = out
            return spk, syn_next, mem_next
        if len(out) == 2:
            spk, mem_next = out
            syn_next = getattr(neuron_layer, "syn", syn_state)
            return spk, syn_next, mem_next
        if len(out) == 1:
            spk = out[0]
            syn_next = getattr(neuron_layer, "syn", syn_state)
            mem_next = getattr(neuron_layer, "mem", mem_state)
            return spk, syn_next, mem_next

    # init_hidden=True layers may return spikes only.
    spk = out
    syn_next = getattr(neuron_layer, "syn", syn_state)
    mem_next = getattr(neuron_layer, "mem", mem_state)
    return spk, syn_next, mem_next


def _find_layer_pairs(model: torch.nn.Module) -> list[tuple[str, str]]:
    """
    Locate (fcN, lifN) pairs by index for experiment scripts.

    This matches the common structure used by E1/E2 models where each dense
    layer is followed by a spiking neuron layer.
    """
    pairs: list[tuple[str, str]] = []
    idx = 1
    while True:
        fc_name = f"fc{idx}"
        lif_name = f"lif{idx}"
        if hasattr(model, fc_name) and hasattr(model, lif_name):
            pairs.append((fc_name, lif_name))
            idx += 1
            continue
        break

    if not pairs:
        raise ValueError("No (fcN, lifN) layer pairs found on the provided model.")

    return pairs


def collect_forward_diagnostics(
    model: torch.nn.Module,
    spike_seq: torch.Tensor,
    layer_pairs: list[tuple[str, str]] | None = None,
) -> dict[str, LayerDiagnostics]:
    """
    Collect membrane potentials and spikes over time for each spiking layer.

    Args:
        model: Experiment model with fc/lif layer pairs (e.g., E1/E2 networks).
        spike_seq: Input spike sequence with shape [T, B, input_size].
        layer_pairs: Optional explicit list of (fc_layer_name, neuron_layer_name).

    Returns:
        Dict keyed by neuron layer name (lif1, lif2, ...), each containing:
          - spikes: [T, B, N]
          - membrane: [T, B, N]
          - threshold: scalar
    """
    if spike_seq.ndim != 3:
        raise ValueError(f"spike_seq must be rank-3 [T, B, input_size], got shape: {tuple(spike_seq.shape)}")

    pairs = layer_pairs or _find_layer_pairs(model)
    batch_size = spike_seq.shape[1]
    device = spike_seq.device
    dtype = spike_seq.dtype

    # Initialize hidden state per layer.
    mem_state: dict[str, torch.Tensor] = {}
    syn_state: dict[str, torch.Tensor | None] = {}
    spk_history: dict[str, list[torch.Tensor]] = {}
    mem_history: dict[str, list[torch.Tensor]] = {}

    for fc_name, lif_name in pairs:
        fc_layer = getattr(model, fc_name)
        out_features = fc_layer.out_features
        mem_state[lif_name] = torch.zeros(batch_size, out_features, device=device, dtype=dtype)

        neuron_layer = getattr(model, lif_name)
        if neuron_layer.__class__.__name__ in {"Synaptic", "QLIF"}:
            syn_state[lif_name] = torch.zeros(batch_size, out_features, device=device, dtype=dtype)
        else:
            syn_state[lif_name] = None

        spk_history[lif_name] = []
        mem_history[lif_name] = []

    for t in range(spike_seq.shape[0]):
        x_t = spike_seq[t]

        for fc_name, lif_name in pairs:
            fc_layer = getattr(model, fc_name)
            neuron_layer = getattr(model, lif_name)

            current = fc_layer(x_t)
            spk_t, syn_next, mem_next = _step_neuron_layer(
                neuron_layer,
                current,
                syn_state[lif_name],
                mem_state[lif_name],
            )

            syn_state[lif_name] = syn_next
            mem_state[lif_name] = mem_next
            x_t = spk_t

            spk_history[lif_name].append(spk_t.detach().cpu())
            mem_history[lif_name].append(mem_next.detach().cpu())

    diagnostics: dict[str, LayerDiagnostics] = {}
    for _, lif_name in pairs:
        neuron_layer = getattr(model, lif_name)
        diagnostics[lif_name] = LayerDiagnostics(
            name=lif_name,
            spikes=torch.stack(spk_history[lif_name], dim=0),
            membrane=torch.stack(mem_history[lif_name], dim=0),
            threshold=_resolve_threshold(neuron_layer),
        )

    return diagnostics


def plot_layer_spike_trains(
    diagnostics: dict[str, LayerDiagnostics],
    sample_index: int = 0,
    point_size: float = 8.0,
    figsize_per_layer: tuple[float, float] = (10.0, 2.8),
    input_spikes: torch.Tensor | None = None,
    input_layer_name: str = "input",
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Plot spike rasters for each layer using one sample from the batch.

    By default this plots the first sample in the batch, as requested.
    """
    if not diagnostics:
        raise ValueError("diagnostics is empty.")

    layer_entries: list[tuple[str, torch.Tensor]] = []
    if input_spikes is not None:
        if input_spikes.ndim != 3:
            raise ValueError(
                f"input_spikes must be rank-3 [T, B, N], got shape: {tuple(input_spikes.shape)}"
            )
        layer_entries.append((input_layer_name, input_spikes.detach().cpu()))

    for layer_name, layer_diag in diagnostics.items():
        layer_entries.append((layer_name, layer_diag.spikes))

    n_layers = len(layer_entries)
    fig, axes = plt.subplots(
        nrows=n_layers,
        ncols=1,
        figsize=(figsize_per_layer[0], figsize_per_layer[1] * n_layers),
        squeeze=False,
    )

    for row, (layer_name, spikes) in enumerate(layer_entries):
        ax = axes[row][0]
        # spikes: [T, B, N]

        if sample_index < 0 or sample_index >= spikes.shape[1]:
            raise IndexError(
                f"sample_index={sample_index} out of range for batch size {spikes.shape[1]}"
            )

        sample_spikes = spikes[:, sample_index, :]  # [T, N]
        spike_points = (sample_spikes > 0).nonzero(as_tuple=False)

        if spike_points.numel() > 0:
            time_idx = spike_points[:, 0].numpy()
            neuron_idx = spike_points[:, 1].numpy()
            ax.scatter(time_idx, neuron_idx, s=point_size, marker=".", alpha=1, linewidths=0)

        ax.set_title(f"{layer_name}: spike train raster (sample={sample_index})")
        ax.set_xlabel("time step")
        ax.set_ylabel("neuron id")
        ax.set_ylim(-0.5, sample_spikes.shape[1] - 0.5)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    return fig, [axes[i][0] for i in range(n_layers)]


def plot_neuron_membrane_trace(
    diagnostics: dict[str, LayerDiagnostics],
    layer_name: str,
    neuron_index: int,
    sample_index: int = 0,
    include_spike_markers: bool = True,
    show_legend: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot membrane potential over time for one neuron and its threshold line.
    """
    if layer_name not in diagnostics:
        raise KeyError(f"Unknown layer '{layer_name}'. Available: {list(diagnostics.keys())}")

    layer = diagnostics[layer_name]
    mem = layer.membrane  # [T, B, N]
    spk = layer.spikes  # [T, B, N]

    if sample_index < 0 or sample_index >= mem.shape[1]:
        raise IndexError(f"sample_index={sample_index} out of range for batch size {mem.shape[1]}")
    if neuron_index < 0 or neuron_index >= mem.shape[2]:
        raise IndexError(f"neuron_index={neuron_index} out of range for neuron count {mem.shape[2]}")

    mem_trace = mem[:, sample_index, neuron_index].numpy()
    spk_trace = spk[:, sample_index, neuron_index].numpy()
    steps = list(range(mem.shape[0]))

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(steps, mem_trace, linewidth=1.8, label="membrane potential")
    ax.axhline(layer.threshold, linestyle="--", linewidth=1.2, label=f"threshold={layer.threshold:.3f}")

    if include_spike_markers:
        spike_times = [t for t, val in enumerate(spk_trace) if val > 0]
        if spike_times:
            spike_values = [mem_trace[t] for t in spike_times]
            ax.scatter(spike_times, spike_values, s=24, c="red", label="spike")

    ax.set_title(
        f"{layer_name} neuron {neuron_index}: membrane vs threshold (sample={sample_index})"
    )
    ax.set_xlabel("time step")
    ax.set_ylabel("membrane potential")
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(loc="best")
    return ax


def plot_all_layer_membrane_traces(
    diagnostics: dict[str, LayerDiagnostics],
    sample_index: int = 0,
    include_spike_markers: bool = True,
    figure_width: float = 11.0,
    subplot_height: float = 2.2,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Plot all layer-neuron membrane traces in one vertically stacked figure.

    Subplots are ordered by layer and then neuron index (0..N-1) within each
    layer, so rows are stacked vertically by neuron index.
    """
    if not diagnostics:
        raise ValueError("diagnostics is empty.")

    entries: list[tuple[str, int]] = []
    for layer_name, layer in diagnostics.items():
        neuron_count = layer.membrane.shape[2]
        for neuron_index in range(neuron_count):
            entries.append((layer_name, neuron_index))

    if not entries:
        raise ValueError("No neurons found in diagnostics.")

    n_rows = len(entries)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=1,
        figsize=(figure_width, subplot_height * n_rows),
        squeeze=False,
        sharex=True,
    )

    flat_axes = [axes[i][0] for i in range(n_rows)]
    for ax, (layer_name, neuron_index) in zip(flat_axes, entries):
        plot_neuron_membrane_trace(
            diagnostics,
            layer_name=layer_name,
            neuron_index=neuron_index,
            sample_index=sample_index,
            include_spike_markers=include_spike_markers,
            show_legend=False,
            ax=ax,
        )

    # Add one shared legend at figure level to avoid repeating on every row.
    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.tight_layout()
    return fig, flat_axes


def plot_layer_membrane_traces(
    diagnostics: dict[str, LayerDiagnostics],
    layer_name: str,
    sample_index: int = 0,
    include_spike_markers: bool = True,
    figure_width: float = 11.0,
    subplot_height: float = 2.2,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Plot all neuron membrane traces from one layer in a single vertical stack.

    For an output layer with two neurons, this produces two subplots.
    """
    if not diagnostics:
        raise ValueError("diagnostics is empty.")
    if layer_name not in diagnostics:
        raise KeyError(f"Unknown layer '{layer_name}'. Available: {list(diagnostics.keys())}")

    neuron_count = diagnostics[layer_name].membrane.shape[2]
    if neuron_count <= 0:
        raise ValueError(f"Layer '{layer_name}' has no neurons to plot.")

    fig, axes = plt.subplots(
        nrows=neuron_count,
        ncols=1,
        figsize=(figure_width, subplot_height * neuron_count),
        squeeze=False,
        sharex=True,
    )

    flat_axes = [axes[i][0] for i in range(neuron_count)]
    for neuron_index, ax in enumerate(flat_axes):
        plot_neuron_membrane_trace(
            diagnostics,
            layer_name=layer_name,
            neuron_index=neuron_index,
            sample_index=sample_index,
            include_spike_markers=include_spike_markers,
            show_legend=False,
            ax=ax,
        )

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.tight_layout()
    return fig, flat_axes


def run_and_visualize(
    model: torch.nn.Module,
    spike_seq: torch.Tensor,
    neuron_layer_name: str,
    neuron_index: int,
    sample_index: int = 0,
) -> dict[str, LayerDiagnostics]:
    """
    Convenience helper: collect diagnostics then create both requested plots.
    """
    diagnostics = collect_forward_diagnostics(model, spike_seq)
    plot_layer_spike_trains(diagnostics, sample_index=sample_index, input_spikes=spike_seq)
    plot_neuron_membrane_trace(
        diagnostics,
        layer_name=neuron_layer_name,
        neuron_index=neuron_index,
        sample_index=sample_index,
    )
    return diagnostics
