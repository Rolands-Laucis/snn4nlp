from tqdm import tqdm
from pathlib import Path

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
                    annots = line.split('\t')[2:5]
                    words.append(annots)
                    prev_num = current_num
                except (ValueError, IndexError):
                    pass
            
        if len(words) >= min_sentence_length and len(words) <= max_sentence_length:
            sentences.append(words)
    return sentences, len(sentences)

def ReadEmbeddingsFile(path:str, limit:int = None):
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
    return embeddings, embeddings.keys(), len(embeddings['the'])

def ReadUPOSInputFile(path:str, limit:int = None):
    sentences = []
    current_sentence = []
    file_path = Path(path)
    with file_path.open('r', encoding='utf-8') as f:
        i = 0
        for line in tqdm(f, desc=f"Reading UPOS input {file_path}", unit="lines"):
            i += 1
            if limit and i > limit:
                break

            line = line.strip()
            if not line:  # Empty line indicates sentence separator
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                parts = line.split('\t')
                word_info = parts[:3] # lemma, upos, xpos
                vector = list(map(float, parts[3:])) # embedding vector
                current_sentence.append(word_info + vector)
        
        if current_sentence:  # Append last sentence if exists
            sentences.append(current_sentence)
    
    if sentences and sentences[0] and len(sentences[0][0]) > 3:
        embedding_dim = len(sentences[0][0][3:])
    else:
        embedding_dim = 0
    return sentences, embedding_dim