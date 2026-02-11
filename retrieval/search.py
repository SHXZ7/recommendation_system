import numpy as np
import faiss
from pathlib import Path

CF_DIR = Path("models/collaborative")
INDEX_DIR = Path("retrieval")


def load_index():
    return faiss.read_index(str(INDEX_DIR / "movie_index.faiss"))


def load_embeddings():
    user_embeddings = np.load(CF_DIR / "user_embeddings.npy").astype("float32")
    movie_embeddings = np.load(CF_DIR / "movie_embeddings.npy").astype("float32")
    return user_embeddings, movie_embeddings


def search_movies(user_idx, k=10):
    index = load_index()
    user_embeddings, _ = load_embeddings()

    user_vector = user_embeddings[user_idx:user_idx+1]
    faiss.normalize_L2(user_vector)

    scores, indices = index.search(user_vector, k)
    return indices[0], scores[0]

if __name__ == "__main__":
    indices, scores = search_movies(user_idx=0, k=10)
    print("Top movie indices:", indices)
    print("Scores:", scores)
