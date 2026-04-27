from tqdm import tqdm
from pathlib import Path
import pickle
from typing import Any, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

Sentence = list[list[str]]
Embeddings = dict[str, list[float]]

# https://github.com/UniversalDependencies/UD_English-EWT
def ReadConlluFile(
    path: str,
    min_sentence_length: int = 5,
    max_sentence_length: int | None = None,
    limit: int | None = None,
) -> tuple[list[Sentence], int]:
    """Read UD CoNLL-U data and return tokenized sentences with POS fields."""
    sentences: list[Sentence] = []
    file_path = Path(path)
    with file_path.open('r', encoding='utf-8') as f:
        words: Sentence = []
        prev_num = 0
        for line in tqdm(f, desc=f"Reading {file_path}", unit="lines"):
            if limit and len(sentences) >= limit:
                break
            
            line = line.strip()
            if line and line[0] != '#':
                try:
                    current_num = int(line.split('\t')[0])
                    if current_num <= prev_num:
                        if (not min_sentence_length or len(words) >= min_sentence_length) and (not max_sentence_length or len(words) <= max_sentence_length):
                            sentences.append(words)
                        words = []
                    annots = line.split('\t')
                    words.append([annots[0].lower()] + annots[3:5]) #NOTE lowercase the word to better match the embeddings, and only keep the word and UPOS tag
                    prev_num = current_num
                except (ValueError, IndexError):
                    pass
            
        if (min_sentence_length and len(words) >= min_sentence_length) and (max_sentence_length and len(words) <= max_sentence_length):
            sentences.append(words)
    return sentences, len(sentences)

# https://github.com/UniversalNER/UNER_English-EWT
def ReadIOB2File(
    path: str,
    min_sentence_length: int = 5,
    max_sentence_length: int | None = None,
    limit: int | None = None,
) -> tuple[list[Sentence], int]:
    """Read IOB2 NER data and return sentences as [token, ner_tag] pairs."""
    sentences: list[Sentence] = []
    file_path = Path(path)
    with file_path.open('r', encoding='utf-8') as f:
        words: Sentence = []
        prev_num = 0
        for line in tqdm(f, desc=f"Reading {file_path}", unit="lines"):
            if limit and len(sentences) >= limit:
                break
            
            line = line.strip()
            if line and line[0] != '#':
                try:
                    current_num = int(line.split('\t')[0])
                    if current_num <= prev_num:
                        if (not min_sentence_length or len(words) >= min_sentence_length) and (not max_sentence_length or len(words) <= max_sentence_length):
                            sentences.append(words)
                        words = []
                    annots = line.split('\t')
                    words.append([annots[0].lower(), annots[1]]) #NOTE lowercase the word to better match the embeddings
                    prev_num = current_num
                except (ValueError, IndexError):
                    pass
            
        if (not min_sentence_length or len(words) >= min_sentence_length) and (not max_sentence_length or len(words) <= max_sentence_length):
            sentences.append(words)
    return sentences, len(sentences)

def ReadRawEmbeddingsFile(path: str, limit: int | None = None) -> tuple[Embeddings, int, int]:
    """Read plain-text embeddings and return (embeddings, vocab_size, embedding_dim)."""
    embeddings: Embeddings = {}
    file_path = Path(path)
    with file_path.open('r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Reading embeddings", unit="lines"):
            parts = line.strip().split(' ')
            word = parts[0]
            vector = list(map(float, parts[1:]))
            embeddings[word] = vector
            if limit and len(embeddings) >= limit:
                break
    embedding_dim = len(next(iter(embeddings.values()))) if embeddings else 0
    return embeddings, len(embeddings.keys()), embedding_dim

def ReadPickledEmbeddingsFile(path: str, limit: int | None = None) -> tuple[Embeddings, int, int]:
    """Read pickled embeddings from a dict or (dict, dim) tuple payload."""
    file_path = Path(path)
    with file_path.open('rb') as f:
        payload: Any = pickle.load(f)

    if isinstance(payload, tuple):
        embeddings: Any = payload[0]
        stored_dim = payload[1] if len(payload) > 1 else None
    else:
        embeddings = payload
        stored_dim = None

    if not isinstance(embeddings, dict):
        raise ValueError('Pickle file must contain an embeddings dict or (embeddings, dim) tuple')

    if limit is not None:
        embeddings = dict(list(embeddings.items())[:limit])

    if embeddings:
        embedding_dim = len(next(iter(embeddings.values())))
    elif stored_dim is not None:
        embedding_dim = stored_dim
    else:
        embedding_dim = 0

    return embeddings, len(embeddings.keys()), int(embedding_dim)

def ReadUPOSInputFile(path: str, limit: int | None = None) -> tuple[list[Any], int]:
    """Read pickled UPOS model input and infer embedding dimension."""
    file_path = Path(path)
    with file_path.open('rb') as f:
        sentences: list[Any] = pickle.load(f)

    if limit is not None:
        sentences = sentences[:limit]
    
    if sentences and sentences[0] and len(sentences[0][0]) > 3:
        embedding_dim = len(sentences[0][0][3:])
    else:
        embedding_dim = 0
    return sentences, int(embedding_dim)

def ReadNERInputFile(path: str, limit: int | None = None) -> tuple[list[Any], int]:
    """Read pickled NER model input and infer embedding dimension."""
    file_path = Path(path)
    with file_path.open('rb') as f:
        sentences: list[Any] = pickle.load(f)

    if limit is not None:
        sentences = sentences[:limit]
    
    if sentences and sentences[0] and len(sentences[0][0]) > 2:
        embedding_dim = len(sentences[0][0][2:])
    else:
        embedding_dim = 0
    return sentences, int(embedding_dim)

def GetEmbeddingUnkVector(embeddings: dict[str, Sequence[float]], dim: int) -> list[float]:
    """Return an existing UNK embedding if present, otherwise a zero vector."""
    for k in ["<UNK>", "<unk>", "<unknown>", "[UNK]"]:
        if k in embeddings:
            return list(embeddings[k])
    return [0.0] * dim

def ReadParquetFile(path: str, limit: int | None = None) -> "pd.DataFrame":
    """Read a parquet dataset and optionally truncate rows."""
    import pandas as pd

    df = pd.read_parquet(path)
    if limit is not None:
        df = df.head(limit)
    return df

def ReadSENTInputFile(path: str, limit: int | None = None) -> tuple[list[Any], int]:
    """Read pickled sentence-classification input and infer embedding size."""
    file_path = Path(path)
    with file_path.open('rb') as f:
        sentences: list[Any] = pickle.load(f)

    if limit is not None:
        sentences = sentences[:limit]
    
    if sentences and sentences[0]:
        embedding_dim = len(sentences[0][0][0])
    else:
        embedding_dim = 0
    return sentences, int(embedding_dim)

# if __name__ == "__main__":
#     sent, emb = ReadSENTInputFile('input_data\cast_sent\sent_d50_dev.pkl')
#     label_tags = [
#         sentence[1]
#         for sentence in sent
#     ]
#     sentence_length = len(sent[0][0])
#     print(f"Read {len(sent)} sentences with embedding dim {emb}", len(label_tags), label_tags[:10], sentence_length)
