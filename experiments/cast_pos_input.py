from readers import ReadConlluFile, ReadPickledEmbeddingsFile, GetEmbeddingUnkVector
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DATA_DIR = PROJECT_ROOT / 'input_data'
UD_DIR = INPUT_DATA_DIR / 'ud_ewt'

parser = argparse.ArgumentParser(description='Train an SNN for UPOS tagging')
parser.add_argument('--limit', type=int, default=None, help='Limit the number of sentences for testing (default: 100)')
parser.add_argument('--min_sentence_length', type=int, default=5, help='Minimum sentence length (default: 5)')
parser.add_argument('--max_sentence_length', type=int, default=40, help='Maximum sentence length (default: 40)')
parser.add_argument('--embeddings_path', type=str, default=INPUT_DATA_DIR / 'word_embeddings' / 'glove' / 'glove_50d.pkl', help='Path to the embeddings file')
parser.add_argument('--out_folder', type=str, default=INPUT_DATA_DIR / 'cast_pos', help='Path to save the cast embeddings')
args = parser.parse_args()

assert args.embeddings_path, '--embeddings_path is required'
assert args.out_folder, '--out_folder is required'

EMBEDDINGS_PATH = args.embeddings_path
OUT_DIR = args.out_folder
EMBEDDINGS_PATH = Path(EMBEDDINGS_PATH)
OUT_DIR = Path(OUT_DIR)

# print info about this script
print(f"Preparing/casting input files for UPOS tagging with the following settings:")
print(f"  - Limit: {args.limit}")

#load the UD dataset
UD_train, _ = ReadConlluFile(UD_DIR / 'en_ewt-ud-train.conllu', min_sentence_length=args.min_sentence_length, max_sentence_length=args.max_sentence_length, limit=args.limit)
UD_dev, _ = ReadConlluFile(UD_DIR / 'en_ewt-ud-dev.conllu', min_sentence_length=args.min_sentence_length, max_sentence_length=args.max_sentence_length, limit=args.limit)
UD_test, _ = ReadConlluFile(UD_DIR / 'en_ewt-ud-test.conllu', min_sentence_length=args.min_sentence_length, max_sentence_length=args.max_sentence_length, limit=args.limit)  # Limit to 1000 sentences for testing
print(len(UD_train), len(UD_dev), len(UD_test))

#load word embeddings
embeddings, embd_count, embedding_dim = ReadPickledEmbeddingsFile(EMBEDDINGS_PATH, limit=args.limit)
print('Embeddings:', embd_count, 'Dimension:', embedding_dim)

unk_vector = GetEmbeddingUnkVector(embeddings, embedding_dim)

# cast and save the input for UPOS tagging
OUT_DIR.mkdir(parents=True, exist_ok=True)

datasets = {
    'train': UD_train,
    'dev': UD_dev,
    'test': UD_test
}
for dataset in datasets.keys():
    output_path = OUT_DIR / f'pos_d{embedding_dim}_{dataset}.pkl'
    metadata_path = OUT_DIR / f'pos_d{embedding_dim}_{dataset}.metadata.json'
    source_sentence_count = len(datasets[dataset])
    source_sentence_lengths = [len(sentence) for sentence in datasets[dataset]]
    serialized_sentences = []
    for sentence in tqdm(datasets[dataset], desc=f"Serializing {dataset}", unit="sentences"):
        if args.limit is not None and len(sentence) > args.limit:
            break

        serialized_sentence = []
        for word_info in sentence:
            word = word_info[0]
            vector = embeddings.get(word, unk_vector)
            serialized_sentence.append(word_info + vector)
        serialized_sentences.append(serialized_sentence)

    with output_path.open('wb') as out:
        pickle.dump(serialized_sentences, out, protocol=pickle.HIGHEST_PROTOCOL)

    exported_sentence_lengths = [len(sentence) for sentence in serialized_sentences]
    metadata = {
        'dataset': dataset,
        'pickle_path': str(output_path),
        'sentence_count_source': source_sentence_count,
        'sentence_count_exported': len(serialized_sentences),
        'source_sentence_length_min': min(source_sentence_lengths) if source_sentence_lengths else 0,
        'source_sentence_length_max': max(source_sentence_lengths) if source_sentence_lengths else 0,
        'exported_sentence_length_min': min(exported_sentence_lengths) if exported_sentence_lengths else 0,
        'exported_sentence_length_max': max(exported_sentence_lengths) if exported_sentence_lengths else 0,
        'min_sentence_length_used': args.min_sentence_length,
        'max_sentence_length_used': args.max_sentence_length,
        'limit_used': args.limit,
        'embeddings_path_used': str(EMBEDDINGS_PATH),
        'embedding_dim': embedding_dim,
    }
    with metadata_path.open('w', encoding='utf-8') as out_meta:
        json.dump(metadata, out_meta, indent=2)