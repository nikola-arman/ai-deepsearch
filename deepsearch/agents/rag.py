from transformers import AutoTokenizer, AutoModel
from functools import lru_cache
import gdown
import os
import zipfile
import numpy as np
import json
import faiss

@lru_cache(maxsize=1)
def _load_model():
    tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
    model = AutoModel.from_pretrained("albert/albert-base-v2")
    return tokenizer, model

EMBEDDINGS_ID = '1kDeAh-hgfU6YKQvJlOoXYTSJSQb-8yCx'

def search(query: str) -> list[str]:
    os.makedirs('cache', exist_ok=True)

    if not os.path.exists('cache/faiss.index'):
        gdown.download(id=EMBEDDINGS_ID, output='cache/embeddings.zip')

        with zipfile.ZipFile('cache/embeddings.zip', 'r') as zip_ref:
            zip_ref.extractall('cache')
            
        embeddings = np.load('cache/embeddings.npy')
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, 'cache/faiss.index')

    index = faiss.read_index('cache/faiss.index')
    meta = json.load(open('cache/meta.json'))

    tokenizer, model = _load_model()
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()

    _, indices = index.search(embeddings, 10)
    return [meta[i] for i in indices[0]]