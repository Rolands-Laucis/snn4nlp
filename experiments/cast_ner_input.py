from readers import ReadIOB2File, ReadPickledEmbeddingsFile
from pathlib import Path
from tqdm import tqdm
import argparse
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DATA_DIR = PROJECT_ROOT / 'input_data'
NER_DIR = INPUT_DATA_DIR / 'ner'

parser = argparse.ArgumentParser(description='Train an SNN for NER tagging')
parser.add_argument('--limit', type=int, default=100, help='Limit the number of sentences for testing (default: 100)')
parser.add_argument('--embeddings_path', type=str, default=INPUT_DATA_DIR / 'word_embeddings' / 'glove' / 'glove_50d.pkl', help='Path to the embeddings file')
parser.add_argument('--out_folder', type=str, default=INPUT_DATA_DIR / 'cast_ner', help='Path to save the cast embeddings')
args = parser.parse_args()

assert args.embeddings_path, '--embeddings_path is required'
assert args.out_folder, '--out_folder is required'

limit = args.limit

EMBEDDINGS_PATH = args.embeddings_path
OUT_DIR = args.out_folder

# print info about this script
print(f"Preparing/casting input files for NER tagging with the following settings:")
print(f"  - Limit: {limit}")

#load the NER dataset
NER_train, _ = ReadIOB2File(NER_DIR / 'en_ewt-ud-train.iob2', limit=limit)
NER_dev, _ = ReadIOB2File(NER_DIR / 'en_ewt-ud-dev.iob2', limit=limit)
NER_test, _ = ReadIOB2File(NER_DIR / 'en_ewt-ud-test.iob2', limit=limit)  # Limit to 1000 sentences for testing
print(len(NER_train), len(NER_dev), len(NER_test))

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

# cast and save the input for NER tagging
OUT_DIR.mkdir(parents=True, exist_ok=True)

datasets = {
    'train': NER_train,
    'dev': NER_dev,
    'test': NER_test
}
for dataset in datasets.keys():
    output_path = OUT_DIR / f'ner_d{embedding_dim}_{dataset}.pkl'
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