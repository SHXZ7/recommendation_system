import numpy as np
import faiss
from pathlib import Path

CF_DIR = Path("models/collaborative")
INDEX_DIR = Path("retrieval")

INDEX_DIR.mkdir(parents=True, exist_ok=True)

def load_movie_embeddings():
    embeddings = np.load(CF_DIR / "movie_embeddings.npy")
    return embeddings.astype("float32")

def build_ann_index(embeddings):
    dim = embeddings.shape[1]

    # Inner product index (for dot product similarity)
    index = faiss.IndexFlatIP(dim)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    index.add(embeddings)
    return index

def save_index(index):
    faiss.write_index(index, str(INDEX_DIR / "movie_index.faiss"))

if __name__ == "__main__":
    movie_embeddings = load_movie_embeddings()
    index = build_ann_index(movie_embeddings)
    save_index(index)

    print("ANN index built and saved successfully")
