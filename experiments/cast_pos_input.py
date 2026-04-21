from readers import ReadConlluFile, ReadPickledEmbeddingsFile
from pathlib import Path
from tqdm import tqdm
import argparse
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DATA_DIR = PROJECT_ROOT / 'input_data'
UD_DIR = INPUT_DATA_DIR / 'ud_gum'

parser = argparse.ArgumentParser(description='Train an SNN for UPOS tagging')
parser.add_argument('--limit', type=int, default=100, help='Limit the number of sentences for testing (default: 100)')
parser.add_argument('--embeddings_path', type=str, default=INPUT_DATA_DIR / 'word_embeddings' / 'glove' / 'glove_50d.pkl', help='Path to the embeddings file')
parser.add_argument('--out_folder', type=str, default=INPUT_DATA_DIR / 'cast_pos', help='Path to save the cast embeddings')
args = parser.parse_args()

assert args.embeddings_path, '--embeddings_path is required'
assert args.out_folder, '--out_folder is required'

limit = args.limit

EMBEDDINGS_PATH = args.embeddings_path
OUT_DIR = args.out_folder

# print info about this script
print(f"Preparing/casting input files for UPOS tagging with the following settings:")
print(f"  - Limit: {limit}")

#load the UD dataset
UD_train, _ = ReadConlluFile(UD_DIR / 'en_gum-ud-train.conllu', limit=limit)
UD_dev, _ = ReadConlluFile(UD_DIR / 'en_gum-ud-dev.conllu', limit=limit)
UD_test, _ = ReadConlluFile(UD_DIR / 'en_gum-ud-test.conllu', limit=limit)  # Limit to 1000 sentences for testing
print(len(UD_train), len(UD_dev), len(UD_test))

#load word embeddings
embeddings, embd_count, embedding_dim = ReadPickledEmbeddingsFile(EMBEDDINGS_PATH, limit=limit)
print('Embeddings:', embd_count, 'Dimension:', embedding_dim)

def get_unk_vector(embeddings, dim):
    """Pick an UNK vector from GloVe if available, else return zero vector."""
    for k in ["<UNK>", "<unk>", "unk", "[UNK]"]:
        if k in embeddings:
            return embeddings[k]
    return [0.0] * dim
unk_vector = get_unk_vector(embeddings, embedding_dim)

# cast and save the input for UPOS tagging
OUT_DIR.mkdir(parents=True, exist_ok=True)

datasets = {
    'train': UD_train,
    'dev': UD_dev,
    'test': UD_test
}
for dataset in datasets.keys():
    output_path = OUT_DIR / f'pos_d{embedding_dim}_{dataset}.pkl'
    serialized_sentences = []
    for sentence in tqdm(datasets[dataset], desc=f"Serializing {dataset}", unit="sentences"):
        if limit is not None and len(sentence) > limit:
            break

        serialized_sentence = []
        for word_info in sentence:
            word = word_info[0]
            vector = embeddings.get(word, unk_vector)
            serialized_sentence.append(word_info + vector)
        serialized_sentences.append(serialized_sentence)

    with output_path.open('wb') as out:
        pickle.dump(serialized_sentences, out, protocol=pickle.HIGHEST_PROTOCOL)