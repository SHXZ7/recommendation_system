import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("models/collaborative")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_train_data():
    return pd.read_csv(DATA_DIR / "train.csv")

def create_mappings(df):
    user_ids = df["user_id"].unique()
    movie_ids = df["movie_id"].unique()

    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    movie_to_idx = {m: i for i, m in enumerate(movie_ids)}

    idx_to_user = {i: u for u, i in user_to_idx.items()}
    idx_to_movie = {i: m for m, i in movie_to_idx.items()}

    return user_to_idx, movie_to_idx, idx_to_user, idx_to_movie

def initialize_embeddings(num_users, num_movies, embedding_dim=32):
    user_embeddings = np.random.normal(
        scale=0.1, size=(num_users, embedding_dim)
    )
    movie_embeddings = np.random.normal(
        scale=0.1, size=(num_movies, embedding_dim)
    )
    return user_embeddings, movie_embeddings

def train_mf(
    df,
    user_to_idx,
    movie_to_idx,
    user_embeddings,
    movie_embeddings,
    lr=0.01,
    epochs=5,
    neg_samples=5
):
    num_movies = movie_embeddings.shape[0]

    for epoch in range(epochs):
        df = df.sample(frac=1).reset_index(drop=True)
        total_loss = 0.0

        for _, row in tqdm(df.iterrows(), total=len(df)):
            u = user_to_idx[row["user_id"]]
            i = movie_to_idx[row["movie_id"]]

            # Positive sample
            pos_score = np.dot(user_embeddings[u], movie_embeddings[i])
            pos_loss = -np.log(sigmoid(pos_score))
            total_loss += pos_loss

            grad = sigmoid(pos_score) - 1
            user_embeddings[u] -= lr * grad * movie_embeddings[i]
            movie_embeddings[i] -= lr * grad * user_embeddings[u]

            # Negative samples
            for _ in range(neg_samples):
                j = np.random.randint(num_movies)
                neg_score = np.dot(user_embeddings[u], movie_embeddings[j])
                neg_loss = -np.log(1 - sigmoid(neg_score))
                total_loss += neg_loss

                grad = sigmoid(neg_score)
                user_embeddings[u] -= lr * grad * movie_embeddings[j]
                movie_embeddings[j] -= lr * grad * user_embeddings[u]

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return user_embeddings, movie_embeddings

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def save_embeddings(user_embeddings, movie_embeddings):
    np.save(OUTPUT_DIR / "user_embeddings.npy", user_embeddings)
    np.save(OUTPUT_DIR / "movie_embeddings.npy", movie_embeddings)

if __name__ == "__main__":
    train_df = load_train_data()

    user_to_idx, movie_to_idx, idx_to_user, idx_to_movie = create_mappings(train_df)

    user_embs, movie_embs = initialize_embeddings(
        num_users=len(user_to_idx),
        num_movies=len(movie_to_idx),
        embedding_dim=32
    )

    user_embs, movie_embs = train_mf(
        train_df,
        user_to_idx,
        movie_to_idx,
        user_embs,
        movie_embs,
        epochs=5
    )

    save_embeddings(user_embs, movie_embs)

    print("Matrix Factorization training complete")
