
from readers import ReadRawEmbeddingsFile
import pickle
from pathlib import Path
import argparse
import os
import numpy as np
import torch
# from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DATA_DIR = PROJECT_ROOT / 'input_data'

parser = argparse.ArgumentParser(description='Train an SNN for UPOS tagging')
parser.add_argument('--limit', type=int, default=None, help='Limit the number of sentences for testing (default: 100)')
parser.add_argument('--embeddings_path', type=str, default=INPUT_DATA_DIR / 'word_embeddings' / 'glove' / 'wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt', help='Path to the embeddings file')
parser.add_argument('--out_path', type=str, default=None, help='Path to save the cast embeddings')
parser.add_argument('--normalization_mode', type=str, default='sigmoid', choices=['l2', 'rescale', 'tanh', 'sigmoid'], help='Normalization mode: l2 (unit-norm then global min-max to [0,1]), rescale (global min-max to [0,1]), tanh ((tanh(x) + 1) / 2), or sigmoid (sigmoid(4*x))')
args = parser.parse_args()

assert args.embeddings_path is not None, '--embeddings_path is required'
assert args.out_path is not None, '--out_path is required'

EMBEDDINGS_PATH = args.embeddings_path
assert os.path.exists(EMBEDDINGS_PATH), f"Embeddings file not found at {EMBEDDINGS_PATH}"

print(f"Reading embeddings from {EMBEDDINGS_PATH} with limit={args.limit} and normalization_mode={args.normalization_mode}...")

embeddings, _, dim = ReadRawEmbeddingsFile(EMBEDDINGS_PATH, limit=args.limit)
#GLOVE contains 1 vector that is 49 dim instead of 50. Crazy. Filter out any vectors that don't match the expected dimension.
embeddings = {word: vec for word, vec in embeddings.items() if len(vec) == dim}
print(f"Read {len(embeddings)} embeddings with dimension {dim}")
# length_counts = Counter(len(vec) for vec in embeddings.values()) 
# print("Embedding dimension counts:", length_counts)

final_scalar_range: tuple[float, float] | None = None

if embeddings:
	words = list(embeddings.keys())
	matrix = np.array([embeddings[word] for word in words], dtype=np.float32)

	if args.normalization_mode == 'l2':
		norms = np.linalg.norm(matrix, axis=1, keepdims=True)
		zero_norm_count = int(np.count_nonzero(norms.squeeze(-1) == 0.0))
		norms_safe = np.where(norms == 0.0, 1.0, norms)
		normalized = matrix / norms_safe

		pre_min = float(np.min(normalized))
		pre_max = float(np.max(normalized))
		outside_mask = (normalized < -1.0) | (normalized > 1.0)
		outside_count = int(np.count_nonzero(outside_mask))
		total_values = int(normalized.size)

		clamped = np.clip(normalized, -1.0, 1.0)
		post_min = float(np.min(clamped))
		post_max = float(np.max(clamped))
		clamped_count = int(np.count_nonzero(clamped != normalized))

		print("L2 normalization + clamp stats:")
		print(f"- Zero-norm vectors: {zero_norm_count}")
		print(f"- Normalized scalar range before clamp: [{pre_min:.6f}, {pre_max:.6f}]")
		print(
			f"- Scalars outside [-1, 1] before clamp: {outside_count}/{total_values} "
			f"({(outside_count / total_values) * 100:.6f}%)"
		)
		print(
			f"- Scalars changed by clamp: {clamped_count}/{total_values} "
			f"({(clamped_count / total_values) * 100:.6f}%)"
		)
		print(f"- Final scalar range after clamp: [{post_min:.6f}, {post_max:.6f}]")

		l2_min = float(np.min(clamped))
		l2_max = float(np.max(clamped))
		l2_span = l2_max - l2_min
		if l2_span == 0.0:
			processed = np.zeros_like(clamped)
			print(f"- L2 scalar range before rescale: [{l2_min:.6f}, {l2_max:.6f}] (degenerate)")
		else:
			processed = (clamped - l2_min) / l2_span
			print(f"- L2 scalar range before rescale: [{l2_min:.6f}, {l2_max:.6f}]")
			print(f"- L2 rescale width: {l2_span:.6f}")

		rescaled_min = float(np.min(processed))
		rescaled_max = float(np.max(processed))
		print(f"- Final scalar range after L2+rescale: [{rescaled_min:.6f}, {rescaled_max:.6f}]")
		final_scalar_range = (rescaled_min, rescaled_max)
	elif args.normalization_mode == 'rescale':
		raw_min = float(np.min(matrix))
		raw_max = float(np.max(matrix))
		raw_span = raw_max - raw_min

		if raw_span == 0.0:
			# All scalars are identical; keep values finite and within [0, 1].
			processed = np.zeros_like(matrix)
			print("Rescale normalization stats:")
			print(f"- Raw scalar range before rescale: [{raw_min:.6f}, {raw_max:.6f}] (degenerate)")
		else:
			processed = (matrix - raw_min) / raw_span
			print("Rescale normalization stats:")
			print(f"- Raw scalar range before rescale: [{raw_min:.6f}, {raw_max:.6f}]")
			print(f"- Raw range width: {raw_span:.6f}")

		post_min = float(np.min(processed))
		post_max = float(np.max(processed))
		print(f"- Final scalar range after rescale: [{post_min:.6f}, {post_max:.6f}]")
		final_scalar_range = (post_min, post_max)
	elif args.normalization_mode == 'tanh':
		processed_tensor = (torch.tanh(torch.from_numpy(matrix)) + 1.0) / 2.0
		processed = processed_tensor.cpu().numpy()
		post_min = float(np.min(processed))
		post_max = float(np.max(processed))
		print("Tanh normalization stats:")
		print("- Applied (tanh(x) + 1) / 2 mapping")
		print(f"- Final scalar range after tanh: [{post_min:.6f}, {post_max:.6f}]")
		final_scalar_range = (post_min, post_max)
	elif args.normalization_mode == 'sigmoid':
		# Sharper sigmoid mapping; multiplies inputs by 4 before sigmoid
		processed_tensor = torch.sigmoid(4.0 * torch.from_numpy(matrix)) #*4 because it makes the sigmoid steeper, pushing more values closer to 0 or 1. This seems intuitive to use up more of the bulk values in the desired range
		processed = processed_tensor.cpu().numpy()
		post_min = float(np.min(processed))
		post_max = float(np.max(processed))
		print("Sigmoid normalization stats:")
		print("- Applied sigmoid(4*x) mapping")
		print(f"- Final scalar range after sigmoid: [{post_min:.6f}, {post_max:.6f}]")
		final_scalar_range = (round(post_min, 6), round(post_max, 6))

	processed_mean = float(np.mean(processed))
	processed_median = float(np.median(processed))
	print(f"- Final scalar mean: {processed_mean:.6f}")
	print(f"- Final scalar median: {processed_median:.6f}")

	embeddings = {word: processed[i].tolist() for i, word in enumerate(words)}
else:
	print("No embeddings loaded; skipping normalization.")

output_path = Path(args.out_path)
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open('wb') as f:
	pickle.dump((embeddings, dim, final_scalar_range, args.normalization_mode), f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved cast embeddings to {output_path}")