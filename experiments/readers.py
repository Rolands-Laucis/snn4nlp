from tqdm import tqdm
from pathlib import Path
import pickle

# https://github.com/UniversalDependencies/UD_English-EWT
def ReadConlluFile(path:str, min_sentence_length:int = 4, max_sentence_length:int = 30, limit:int = None):
    sentences = []
    file_path = Path(path)
    with file_path.open('r', encoding='utf-8') as f:
        words = []
        prev_num = 0
        for line in tqdm(f, desc=f"Reading {file_path}", unit="lines"):
            if limit and len(sentences) >= limit:
                break
            
            line = line.strip()
            if line and line[0] != '#':
                try:
                    current_num = int(line.split('\t')[0])
                    if current_num <= prev_num:
                        if len(words) >= min_sentence_length and len(words) <= max_sentence_length:
                            sentences.append(words)
                        words = []
                    annots = line.split('\t')
                    words.append([annots[0].lower()] + annots[3:5]) #NOTE lowercase the word to better match the embeddings, and only keep the word and UPOS tag
                    prev_num = current_num
                except (ValueError, IndexError):
                    pass
            
        if len(words) >= min_sentence_length and len(words) <= max_sentence_length:
            sentences.append(words)
    return sentences, len(sentences)

# https://github.com/UniversalNER/UNER_English-EWT
def ReadIOB2File(path:str, min_sentence_length:int = 4, max_sentence_length:int = 30, limit:int = None):
    sentences = []
    file_path = Path(path)
    with file_path.open('r', encoding='utf-8') as f:
        words = []
        prev_num = 0
        for line in tqdm(f, desc=f"Reading {file_path}", unit="lines"):
            if limit and len(sentences) >= limit:
                break
            
            line = line.strip()
            if line and line[0] != '#':
                try:
                    current_num = int(line.split('\t')[0])
                    if current_num <= prev_num:
                        if len(words) >= min_sentence_length and len(words) <= max_sentence_length:
                            sentences.append(words)
                        words = []
                    annots = line.split('\t')
                    words.append([annots[0].lower(), annots[1]]) #NOTE lowercase the word to better match the embeddings
                    prev_num = current_num
                except (ValueError, IndexError):
                    pass
            
        if len(words) >= min_sentence_length and len(words) <= max_sentence_length:
            sentences.append(words)
    return sentences, len(sentences)

def ReadRawEmbeddingsFile(path:str, limit:int = None):
    embeddings = {}
    file_path = Path(path)
    with file_path.open('r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Reading embeddings", unit="lines"):
            parts = line.strip().split(' ')
            word = parts[0]
            vector = list(map(float, parts[1:]))
            embeddings[word] = vector
            if limit and len(embeddings) >= limit:
                break
    return embeddings, len(embeddings.keys()), len(embeddings['the'])

def ReadPickledEmbeddingsFile(path:str, limit:int = None):
    file_path = Path(path)
    with file_path.open('rb') as f:
        payload = pickle.load(f)

    if isinstance(payload, tuple):
        embeddings = payload[0]
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

    return embeddings, len(embeddings.keys()), embedding_dim

def ReadUPOSInputFile(path:str, limit:int = None):
    file_path = Path(path)
    with file_path.open('rb') as f:
        sentences = pickle.load(f)

    if limit is not None:
        sentences = sentences[:limit]
    
    if sentences and sentences[0] and len(sentences[0][0]) > 3:
        embedding_dim = len(sentences[0][0][3:])
    else:
        embedding_dim = 0
    return sentences, embedding_dim

def ReadNERInputFile(path:str, limit:int = None):
    file_path = Path(path)
    with file_path.open('rb') as f:
        sentences = pickle.load(f)

    if limit is not None:
        sentences = sentences[:limit]
    
    if sentences and sentences[0] and len(sentences[0][0]) > 2:
        embedding_dim = len(sentences[0][0][2:])
    else:
        embedding_dim = 0
    return sentences, embedding_dim