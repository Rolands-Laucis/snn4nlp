from readers import ReadParquetFile, ReadPickledEmbeddingsFile, GetEmbeddingUnkVector
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DATA_DIR = PROJECT_ROOT / 'input_data'
UD_DIR = INPUT_DATA_DIR / 'ud_ewt'

parser = argparse.ArgumentParser(description='Train an SNN for UPOS tagging')
parser.add_argument('--limit', type=int, default=None, help='Limit the number of sentences for testing (default: 100)')
parser.add_argument('--min_sentence_length', type=int, default=5, help='Minimum sentence length (default: 5)')
parser.add_argument('--max_sentence_length', type=int, default=None, help='Maximum sentence length (default: 40)')
parser.add_argument('--embeddings_path', type=str, default=INPUT_DATA_DIR / 'word_embeddings' / 'glove' / 'glove_50d.pkl', help='Path to the embeddings file')
parser.add_argument('--out_folder', type=str, default=INPUT_DATA_DIR / 'cast_sent', help='Path to save the cast embeddings')
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

#load the dataset
train = ReadParquetFile(INPUT_DATA_DIR / 'sst-2' / 'train-00000-of-00001.parquet')
# dev = ReadParquetFile(INPUT_DATA_DIR / 'sst-2' / 'validation-00000-of-00001.parquet')
test = ReadParquetFile(INPUT_DATA_DIR / 'sst-2' / 'validation-00000-of-00001.parquet')
print('Original dataset sizes:')
print(len(train), len(test))
combined = pd.concat([train, test], ignore_index=True)
# exit(0)

#filter sentences by length
if args.min_sentence_length or args.max_sentence_length:
    def filter_by_length(df, min_len, max_len):
        if min_len and max_len:
            return df[df['sentence'].apply(lambda x: min_len <= len(x.split()) <= max_len)]
        elif min_len:
            return df[df['sentence'].apply(lambda x: len(x.split()) >= min_len)]
        elif max_len:
            return df[df['sentence'].apply(lambda x: len(x.split()) <= max_len)]
        else:
            return df

    combined = filter_by_length(combined, args.min_sentence_length, args.max_sentence_length)

# pad sentences to the same length (args.max_sentence_length) with a <PAD> token
if args.max_sentence_length:
    def pad_sentences(df, max_len):
        df = df.copy()
        df['sentence'] = df['sentence'].apply(
            lambda x: x + (' ' + ' '.join(['<PAD>'] * max(0, max_len - len(x.split()))) if max_len > len(x.split()) else '')
        )
        return df

    combined = pad_sentences(combined, args.max_sentence_length)

train = combined.sample(frac=0.9)
test = combined.drop(train.index).reset_index(drop=True)
train = train.reset_index(drop=True)
# print(train.head())
print('Filtered and restructured dataset sizes:')
print(len(train), len(test))
# print label count
print('Train label distribution:')
print(train['label'].value_counts())
# exit(0)

# Plot histogram with mean line
if False:
    # Combine datasets temporarily for visualization
    combined_sentences = list(train['sentence']) + list(test['sentence'])
    sentence_lengths = [len(sentence.split()) if isinstance(sentence, str) else len(sentence) for sentence in combined_sentences]
    plt.figure(figsize=(10, 6))
    plt.hist(sentence_lengths, bins=50, edgecolor='black', alpha=0.7)
    mean_length = np.mean(sentence_lengths)
    plt.axvline(mean_length, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_length:.2f}')
    plt.xlabel('Sentence Length (words)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sentence Lengths (Train + Dev + Test)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()


#load word embeddings
embeddings, embd_count, embedding_dim = ReadPickledEmbeddingsFile(EMBEDDINGS_PATH, limit=args.limit)
print('Embeddings:', embd_count, 'Dimension:', embedding_dim)

unk_vector = GetEmbeddingUnkVector(embeddings, embedding_dim)

# cast and save the input for sentiment classification
OUT_DIR.mkdir(parents=True, exist_ok=True)

datasets = {
    'train': train,
    'test': test
}
for dataset in datasets.keys():
    output_path = OUT_DIR / f'sent_d{embedding_dim}_{dataset}.pkl'
    metadata_path = OUT_DIR / f'sent_d{embedding_dim}_{dataset}.metadata.json'
    source_sentence_count = len(datasets[dataset])
    source_sentence_lengths = [
        len(sentence.split()) if isinstance(sentence, str) else len(sentence)
        for sentence in datasets[dataset]['sentence']
    ]
    serialized_samples = []
    pad_vector = [0.0] * embedding_dim

    for _, row in tqdm(datasets[dataset].iterrows(), total=len(datasets[dataset]), desc=f"Serializing {dataset}", unit="sentences"):
        sentence = row['sentence']
        label = row['label']

        tokens = sentence.split() #if isinstance(sentence, str) else list(sentence)
        sentence_vectors = []
        for token in tokens:
            if token == '<PAD>':
                vector = pad_vector
            else:
                vector = embeddings.get(token)
                if vector is None:
                    vector = embeddings.get(token.lower(), unk_vector)
            sentence_vectors.append(vector)

        serialized_samples.append([sentence_vectors, label])

    with output_path.open('wb') as out:
        pickle.dump(serialized_samples, out, protocol=pickle.HIGHEST_PROTOCOL)

    exported_sentence_lengths = [len(sample[0]) for sample in serialized_samples]
    metadata = {
        'dataset': dataset,
        'pickle_path': str(output_path),
        'sentence_count_source': source_sentence_count,
        'sentence_count_exported': len(serialized_samples),
        'source_sentence_length_min': min(source_sentence_lengths) if source_sentence_lengths else 0,
        'source_sentence_length_max': max(source_sentence_lengths) if source_sentence_lengths else 0,
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