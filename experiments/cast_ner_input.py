from readers import ReadIOB2File, ReadPickledEmbeddingsFile, GetEmbeddingUnkVector
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DATA_DIR = PROJECT_ROOT / 'input_data'
NER_DIR = INPUT_DATA_DIR / 'ner'

parser = argparse.ArgumentParser(description='Train an SNN for NER tagging')
parser.add_argument('--limit', type=int, default=None, help='Limit the number of sentences for testing (default: 100)')
parser.add_argument('--min_sentence_length', type=int, default=4, help='Minimum sentence length (default: 4)')
parser.add_argument('--max_sentence_length', type=int, default=30, help='Maximum sentence length (default: 30)')
parser.add_argument('--embeddings_path', type=str, default=INPUT_DATA_DIR / 'word_embeddings' / 'glove' / 'glove_50d.pkl', help='Path to the embeddings file')
parser.add_argument('--out_folder', type=str, default=INPUT_DATA_DIR / 'cast_ner', help='Path to save the cast embeddings')
args = parser.parse_args()

assert args.embeddings_path, '--embeddings_path is required'
assert args.out_folder, '--out_folder is required'

EMBEDDINGS_PATH = args.embeddings_path
OUT_DIR = args.out_folder
EMBEDDINGS_PATH = Path(EMBEDDINGS_PATH)
OUT_DIR = Path(OUT_DIR)

# print info about this script
print(f"Preparing/casting input files for NER tagging with the following settings:")
print(f"  - Limit: {args.limit}")

#load the NER dataset
NER_train, _ = ReadIOB2File(NER_DIR / 'en_ewt-ud-train.iob2', min_sentence_length=args.min_sentence_length, max_sentence_length=args.max_sentence_length, limit=args.limit)
NER_dev, _ = ReadIOB2File(NER_DIR / 'en_ewt-ud-dev.iob2', min_sentence_length=args.min_sentence_length, max_sentence_length=args.max_sentence_length, limit=args.limit)
NER_test, _ = ReadIOB2File(NER_DIR / 'en_ewt-ud-test.iob2', min_sentence_length=args.min_sentence_length, max_sentence_length=args.max_sentence_length, limit=args.limit)  # Limit to 1000 sentences for testing
print(len(NER_train), len(NER_dev), len(NER_test))

#load word embeddings
embeddings, embd_count, embedding_dim = ReadPickledEmbeddingsFile(EMBEDDINGS_PATH, limit=args.limit)
print('Embeddings:', embd_count, 'Dimension:', embedding_dim)

unk_vector = GetEmbeddingUnkVector(embeddings, embedding_dim)

# cast and save the input for NER tagging
OUT_DIR.mkdir(parents=True, exist_ok=True)

datasets = {
    'train': NER_train,
    'dev': NER_dev,
    'test': NER_test
}
for dataset in datasets.keys():
    output_path = OUT_DIR / f'ner_d{embedding_dim}_{dataset}.pkl'
    metadata_path = OUT_DIR / f'ner_d{embedding_dim}_{dataset}.metadata.json'
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