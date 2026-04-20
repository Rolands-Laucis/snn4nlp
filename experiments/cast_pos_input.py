from readers import ReadConlluFile, ReadEmbeddingsFile
from pathlib import Path
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Train an SNN for UPOS tagging')
parser.add_argument('--limit', type=int, default=100, help='Limit the number of sentences for testing (default: 100)')
args = parser.parse_args()

limit = args.limit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DATA_DIR = PROJECT_ROOT / 'input_data'
UD_DIR = INPUT_DATA_DIR / 'ud_gum'
EMBEDDINGS_PATH = INPUT_DATA_DIR / 'word_embeddings' / 'glove' / 'wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt'
CAST_POS_DIR = INPUT_DATA_DIR / 'cast_pos'

#load the UD dataset
UD_train, _ = ReadConlluFile(UD_DIR / 'en_gum-ud-train.conllu', limit=limit)
UD_dev, _ = ReadConlluFile(UD_DIR / 'en_gum-ud-dev.conllu', limit=limit)
UD_test, _ = ReadConlluFile(UD_DIR / 'en_gum-ud-test.conllu', limit=limit)  # Limit to 1000 sentences for testing
# print(len(UD_train), len(UD_dev), len(UD_test))

#load word embeddings
embeddings, embd_count, embedding_dim = ReadEmbeddingsFile(EMBEDDINGS_PATH, limit=limit)
# print(embd_count, embedding_dim)

def get_unk_vector(embeddings, dim):
    """Pick an UNK vector from GloVe if available, else return zero vector."""
    for k in ["<UNK>", "<unk>", "unk", "[UNK]"]:
        if k in embeddings:
            return embeddings[k]
    return [0.0] * dim
unk_vector = get_unk_vector(embeddings, embedding_dim)

# cast and save the input for UPOS tagging
CAST_POS_DIR.mkdir(parents=True, exist_ok=True)

datasets = {
    'train': UD_train,
    'dev': UD_dev,
    'test': UD_test
}
for dataset in datasets.keys():
    output_path = CAST_POS_DIR / f'pos_d{embedding_dim}_{dataset}.tsv'
    with output_path.open('w', encoding='utf-8') as out:
        # for i, sentence in enumerate(datasets[dataset]):
        for sentence in tqdm(datasets[dataset], desc=f"Writing {dataset}", unit="sentences"):
            if limit is not None and len(sentence) > limit:
                break
            for word_info in sentence:
                word = word_info[0]
                vector = embeddings.get(word, unk_vector)
                out.write('\t'.join(map(str, word_info + vector)) + '\n')
            out.write('\n') # sentence separator
                # exit(0)