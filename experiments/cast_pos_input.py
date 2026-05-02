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
parser.add_argument('--max_sentence_length', type=int, default=None, help='Maximum sentence length (default: 40)')
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
print(vars(args))

#load the UD dataset
UD_train, _ = ReadConlluFile(UD_DIR / 'en_ewt-ud-train.conllu', min_sentence_length=args.min_sentence_length, max_sentence_length=args.max_sentence_length, limit=args.limit)
UD_dev, _ = ReadConlluFile(UD_DIR / 'en_ewt-ud-dev.conllu', min_sentence_length=args.min_sentence_length, max_sentence_length=args.max_sentence_length, limit=args.limit)
UD_test, _ = ReadConlluFile(UD_DIR / 'en_ewt-ud-test.conllu', min_sentence_length=args.min_sentence_length, max_sentence_length=args.max_sentence_length, limit=args.limit)  # Limit to 1000 sentences for testing
UD_train += UD_dev  # Combine train and dev for training
del UD_dev  # Remove dev set as it's now part of train
print(len(UD_train), len(UD_test))

#load word embeddings
embeddings, embedding_dim, embedding_range, emb_normalization_mode = ReadPickledEmbeddingsFile(EMBEDDINGS_PATH, limit=args.limit)
print('Embeddings:', len(embeddings), 'Dimension:', embedding_dim, 'Embedding scalar range:', embedding_range, 'Normalization mode:', emb_normalization_mode)

unk_vector = GetEmbeddingUnkVector(embeddings, embedding_dim)

# cast and save the input for UPOS tagging
OUT_DIR.mkdir(parents=True, exist_ok=True)

datasets = {
    'train': UD_train,
    'test': UD_test
}
# exports the pickle in the format of [[lemma, upos, xpos, embd_1, embd_2, ..., embd_n], ...]
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
        for word_info in sentence: #word_info is a list of the form [lemma, upos, xpos]
            word = word_info[0] #lemma
            vector = embeddings.get(word, unk_vector) #embedding
            serialized_sentence.append(word_info + vector) # [lemma, upos, xpos, embd_1, embd_2, ..., embd_n]
        serialized_sentences.append(serialized_sentence)

    # sanity check: print the first serialized sentence
    # if serialized_sentences:
    #     print(f"First serialized sentence for {dataset}:")
    #     sent = serialized_sentences[0]
    #     print(f"{dataset} 0th: {datasets[dataset][0]}", len(datasets[dataset][0]))
    #     print(f"serialized_sentences 0th: {[s[:3] for s in sent]}", len(sent))

    #     for word_info in sent[:3]: #print the first 3 words of the first serialized sentence
    #         print(word_info[:3], "Embedding sample:", word_info[3:6]) #print lemma, upos, xpos and first 5 dimensions of the embedding
    # exit(0)

    with output_path.open('wb') as out:
        pickle.dump(serialized_sentences, out, protocol=pickle.HIGHEST_PROTOCOL)

    exported_sentence_lengths = [len(sentence) for sentence in serialized_sentences]
    metadata = {
        'dataset': dataset,
        'pickle_path': str(output_path),
        'sentence_count_source': source_sentence_count,
        'sentence_count_exported': len(serialized_sentences),
        'exported_sentence_length_min': min(exported_sentence_lengths) if exported_sentence_lengths else 0,
        'exported_sentence_length_max': max(exported_sentence_lengths) if exported_sentence_lengths else 0,
        # 'min_sentence_length_used': args.min_sentence_length,
        # 'max_sentence_length_used': args.max_sentence_length,
        'limit_used': args.limit,
        'embeddings_path_used': str(EMBEDDINGS_PATH),
        'embedding_dim': embedding_dim,
    }
    with metadata_path.open('w', encoding='utf-8') as out_meta:
        json.dump(metadata, out_meta, indent=2)