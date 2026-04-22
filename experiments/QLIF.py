"""Custom QLIF neuron for snnTorch.

@article{Wan2025,
abstract = {Spiking neural networks (SNNs) offer biologically inspired computation but remain underexplored for continuous regression tasks in scientific machine learning. In this work, we introduce and systematically evaluate Quadratic Integrate-and-Fire (QIF) neurons as an alternative to the conventional Leaky Integrate-and-Fire (LIF) model in both directly trained SNNs and ANN-to-SNN conversion frameworks. The QIF neuron exhibits smooth and differentiable spiking dynamics, enabling gradient-based training and stable optimization within architectures such as multilayer perceptrons (MLPs), Deep Operator Networks (DeepONets), and Physics-Informed Neural Networks (PINNs). Across benchmarks on function approximation, operator learning, and partial differential equation (PDE) solving, QIF-based networks yield smoother, more accurate, and more stable predictions than their LIF counterparts, which suffer from discontinuous time-step responses and jagged activation surfaces. These results position the QIF neuron as a computational bridge between spiking and continuous-valued deep learning, advancing the integration of neuroscience-inspired dynamics into physics-informed and operator-learning frameworks.},
archivePrefix = {arXiv},
arxivId = {2511.06614},
author = {Wan, Ruyin and Karniadakis, George Em and Stinis, Panos},
eprint = {2511.06614},
pages = {1--20},
title = {{From LIF to QIF: Toward Differentiable Spiking Neurons for Scientific Machine Learning}},
url = {http://arxiv.org/abs/2511.06614},
year = {2025}
}

The original stub described a quadratic integrate-and-fire style model with a
synaptic current state. This module turns that idea into a usable
`snntorch.SpikingNeuron` subclass so it can be dropped into an `nn.Module`
graph, reset with `snntorch.utils.reset`, and used like the built-in neurons.

The implementation keeps the snnTorch conventions for:
- thresholding and surrogate gradients
- hidden-state handling with `init_hidden`
- reset mechanisms (`subtract`, `zero`, `none`)
- optional learnable decay parameters

The discrete-time update is an Euler-style approximation of the commented
equations. The input tensor is treated as the sampled spike drive that replaces
the Dirac delta term from the continuous form.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import snntorch as snn

__all__ = ["QLIF"]


class QLIF(snn.SpikingNeuron):
	"""Quadratic leaky integrate-and-fire neuron.

	Parameters
	----------
	alpha : float or torch.Tensor
		Synaptic decay factor. Values are clamped to ``[0, 1]`` during the
		forward pass, matching the built-in snnTorch neuron behavior.
	beta : float or torch.Tensor
		Membrane decay factor. Values are clamped to ``[0, 1]`` during the
		forward pass.
	threshold : float, default=1.0
		Firing threshold.
	spike_grad : callable, optional
		Surrogate gradient function. Defaults to snnTorch's built-in behavior
		when omitted.
	init_hidden : bool, default=False
		If True, hidden state is stored on the instance and the forward pass
		returns spikes only.
	inhibition : bool, default=False
		Enable winner-take-all inhibition for dense layers.
	learn_alpha : bool, default=False
		Make the synaptic decay learnable.
	learn_beta : bool, default=False
		Make the membrane decay learnable.
	learn_threshold : bool, default=False
		Make the threshold learnable.
	reset_mechanism : {"subtract", "zero", "none"}
		Reset mode after a spike.
	state_quant : callable or bool, default=False
		Optional quantization function for hidden states.
	output : bool, default=False
		If True, return hidden states even when ``init_hidden=True``.
	reset_delay : bool, default=True
		If False, apply reset immediately after a spike.
	lambda_ : float, default=1.0
		Leak coefficient used in the quadratic membrane update.
	v_rest : float, default=0.0
		Resting membrane potential.
	i0 : float, default=0.0
		Baseline current term.
	quadratic_coef : float, default=1.0
		Multiplier for the quadratic drive term ``V(V - 1)``.
	dt : float, default=1.0
		Euler integration step.
	"""

	def __init__(
		self,
		alpha,
		beta,
		threshold=1.0,
		spike_grad=None,
		surrogate_disable=False,
		init_hidden=False,
		inhibition=False,
		learn_alpha=False,
		learn_beta=False,
		learn_threshold=False,
		reset_mechanism="subtract",
		state_quant=False,
		output=False,
		reset_delay=True,
		lambda_=1.0,
		v_rest=0.0,
		i0=0.0,
		quadratic_coef=1.0,
		dt=1.0,
	):
		super().__init__(
			threshold=threshold,
			spike_grad=spike_grad,
			surrogate_disable=surrogate_disable,
			init_hidden=init_hidden,
			inhibition=inhibition,
			learn_threshold=learn_threshold,
			reset_mechanism=reset_mechanism,
			state_quant=state_quant,
			output=output,
		)

		self._register_state_parameter("alpha", alpha, learn_alpha)
		self._register_state_parameter("beta", beta, learn_beta)
		self.register_buffer("lambda_", torch.as_tensor(lambda_, dtype=torch.float32))
		self.register_buffer("v_rest", torch.as_tensor(v_rest, dtype=torch.float32))
		self.register_buffer("i0", torch.as_tensor(i0, dtype=torch.float32))
		self.register_buffer(
			"quadratic_coef", torch.as_tensor(quadratic_coef, dtype=torch.float32)
		)
		self.register_buffer("dt", torch.as_tensor(dt, dtype=torch.float32))

		self._init_syn_and_mem()

		if self.reset_mechanism_val == 0:
			self.state_function = self._base_sub
		elif self.reset_mechanism_val == 1:
			self.state_function = self._base_zero
		elif self.reset_mechanism_val == 2:
			self.state_function = self._base_int

		self.reset_delay = reset_delay

	def _register_state_parameter(self, name, value, learnable):
		tensor = torch.as_tensor(value, dtype=torch.float32)
		if learnable:
			self.register_parameter(name, nn.Parameter(tensor.clone()))
		else:
			self.register_buffer(name, tensor.clone())

	def _init_syn_and_mem(self):
		self.syn = torch.zeros(1, dtype=torch.float32)
		self.mem = torch.zeros(1, dtype=torch.float32)

	def _ensure_state_shape(self, input_):
		if not isinstance(self.syn, torch.Tensor) or self.syn.shape != input_.shape:
			self.syn = torch.zeros_like(input_, device=input_.device)
		if not isinstance(self.mem, torch.Tensor) or self.mem.shape != input_.shape:
			self.mem = torch.zeros_like(input_, device=input_.device)

	def _base_state_function(self, input_):
		alpha = self.alpha.clamp(0, 1)
		beta = self.beta.clamp(0, 1)

		syn = alpha * self.syn + input_
		quadratic_drive = self.quadratic_coef * self.mem * (self.mem - 1.0)
		leak_drive = -self.lambda_ * (self.mem - self.v_rest)
		mem = beta * self.mem + self.dt * (quadratic_drive + leak_drive + syn + self.i0)
		return syn, mem

	def _base_state_reset_zero(self, input_):
		syn = self.alpha.clamp(0, 1) * self.syn + input_
		quadratic_drive = self.quadratic_coef * self.mem * (self.mem - 1.0)
		leak_drive = -self.lambda_ * (self.mem - self.v_rest)
		mem = self.dt * (quadratic_drive + leak_drive + syn + self.i0)
		return 0, mem

	def _base_sub(self, input_):
		syn, mem = self._base_state_function(input_)
		mem = mem - self.reset * self.threshold
		return syn, mem

	def _base_zero(self, input_):
		syn, mem = self._base_state_function(input_)
		syn2, mem2 = self._base_state_reset_zero(input_)
		syn -= syn2 * self.reset
		mem -= mem2 * self.reset
		return syn, mem

	def _base_int(self, input_):
		return self._base_state_function(input_)

	def forward(self, input_, syn=None, mem=None):
		if syn is not None:
			self.syn = syn
		if mem is not None:
			self.mem = mem

		if self.init_hidden and (syn is not None or mem is not None):
			raise TypeError(
				"`mem` or `syn` should not be passed as an argument while `init_hidden=True`"
			)

		self._ensure_state_shape(input_)

		self.reset = self.mem_reset(self.mem)
		self.syn, self.mem = self.state_function(input_)

		if self.state_quant:
			self.mem = self.state_quant(self.mem)
			self.syn = self.state_quant(self.syn)

		if self.inhibition:
			spk = self.fire_inhibition(self.mem.size(0), self.mem)
		else:
			spk = self.fire(self.mem)

		if not self.reset_delay:
			do_reset = spk / self.graded_spikes_factor - self.reset
			if self.reset_mechanism_val == 0:
				self.mem = self.mem - do_reset * self.threshold
			elif self.reset_mechanism_val == 1:
				self.mem = self.mem - do_reset * self.mem

		if self.output:
			return spk, self.syn, self.mem
		if self.init_hidden:
			return spk
		return spk, self.syn, self.mem


if __name__ == "__main__":
	neuron = QLIF(alpha=0.9, beta=0.95, init_hidden=False)
	sample_input = torch.rand(4, 8)
	spk, syn, mem = neuron(sample_input)
	print("spk shape:", tuple(spk.shape))
	print("syn shape:", tuple(syn.shape))
	print("mem shape:", tuple(mem.shape))