from readers import ReadRawEmbeddingsFile
import pickle
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DATA_DIR = PROJECT_ROOT / 'input_data'

parser = argparse.ArgumentParser(description='Train an SNN for UPOS tagging')
parser.add_argument('--limit', type=int, default=100, help='Limit the number of sentences for testing (default: 100)')
parser.add_argument('--embeddings_path', type=str, default=INPUT_DATA_DIR / 'word_embeddings' / 'glove' / 'wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt', help='Path to the embeddings file')
parser.add_argument('--out_path', type=str, default=None, help='Path to save the cast embeddings')
args = parser.parse_args()

assert args.embeddings_path is not None, '--embeddings_path is required'
assert args.out_path is not None, '--out_path is required'

EMBEDDINGS_PATH = args.embeddings_path
assert EMBEDDINGS_PATH.exists(), f"Embeddings file not found at {EMBEDDINGS_PATH}"

embeddings, dim = ReadRawEmbeddingsFile(EMBEDDINGS_PATH, limit=args.limit)

output_path = Path(args.out_path)
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open('wb') as f:
	pickle.dump((embeddings, dim), f, protocol=pickle.HIGHEST_PROTOCOL)
