import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/processed")
BASELINE_DIR = Path("models/baseline")
CF_DIR = Path("models/collaborative")


def load_validation_data():
    return pd.read_csv(DATA_DIR / "val.csv")

def load_popularity_recommendations(top_k=10):
    df = pd.read_csv(BASELINE_DIR / "top_movies.csv")
    return df["movie_id"].head(top_k).tolist()

def build_user_ground_truth(val_df):
    return (
        val_df.groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    if not recommended_k:
        return 0.0
    return len(set(recommended_k) & relevant) / k

def recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    if not relevant:
        return 0.0
    return len(set(recommended_k) & relevant) / len(relevant)

def ndcg_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]

    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)

    ideal_dcg = sum(
        1.0 / np.log2(i + 2)
        for i in range(min(len(relevant), k))
    )

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg


def evaluate_popularity(k=10):
    val_df = load_validation_data()
    ground_truth = build_user_ground_truth(val_df)
    recommendations = load_popularity_recommendations(top_k=k)

    precisions = []
    recalls = []
    ndcgs = []

    for user_id, relevant_items in ground_truth.items():
        precisions.append(
            precision_at_k(recommendations, relevant_items, k)
        )
        recalls.append(
            recall_at_k(recommendations, relevant_items, k)
        )
        ndcgs.append(
            ndcg_at_k(recommendations, relevant_items, k)
        )

    print(f"Popularity Baseline @ {k}")
    print(f"Precision@{k}: {np.mean(precisions):.4f}")
    print(f"Recall@{k}: {np.mean(recalls):.4f}")
    print(f"NDCG@{k}: {np.mean(ndcgs):.4f}")

def load_cf_embeddings():
    user_embeddings = np.load(CF_DIR / "user_embeddings.npy")
    movie_embeddings = np.load(CF_DIR / "movie_embeddings.npy")
    return user_embeddings, movie_embeddings


def load_cf_mappings():
    train_df = pd.read_csv(DATA_DIR / "train.csv")

    user_ids = train_df["user_id"].unique()
    movie_ids = train_df["movie_id"].unique()

    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    movie_to_idx = {m: i for i, m in enumerate(movie_ids)}

    return user_to_idx, movie_to_idx, user_ids, movie_ids

def recommend_cf_for_user(
    user_id,
    user_embeddings,
    movie_embeddings,
    user_to_idx,
    movie_ids,
    user_seen_items,
    k=10
):
    if user_id not in user_to_idx:
        return []

    u_idx = user_to_idx[user_id]
    user_vector = user_embeddings[u_idx]

    scores = movie_embeddings @ user_vector

    ranked_indices = np.argsort(scores)[::-1]

    recommendations = []
    seen = user_seen_items.get(user_id, set())

    for idx in ranked_indices:
        movie_id = movie_ids[idx]
        if movie_id not in seen:
            recommendations.append(movie_id)
        if len(recommendations) == k:
            break

    return recommendations

def evaluate_collaborative_filtering(k=10):
    val_df = load_validation_data()
    ground_truth = build_user_ground_truth(val_df)
    user_seen_items = build_user_seen_items()
    user_embeddings, movie_embeddings = load_cf_embeddings()
    user_to_idx, _, user_ids, movie_ids = load_cf_mappings()

    precisions = []
    recalls = []
    ndcgs = []

    for user_id, relevant_items in ground_truth.items():
        recommendations = recommend_cf_for_user(
    user_id,
    user_embeddings,
    movie_embeddings,
    user_to_idx,
    movie_ids,
    user_seen_items,
    k=k
)


        if not recommendations:
            continue

        precisions.append(
            precision_at_k(recommendations, relevant_items, k)
        )
        recalls.append(
            recall_at_k(recommendations, relevant_items, k)
        )
        ndcgs.append(
            ndcg_at_k(recommendations, relevant_items, k)
        )

    print(f"Collaborative Filtering @ {k}")
    print(f"Precision@{k}: {np.mean(precisions):.4f}")
    print(f"Recall@{k}:    {np.mean(recalls):.4f}")
    print(f"NDCG@{k}:      {np.mean(ndcgs):.4f}")

def build_user_seen_items():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    return (
        train_df.groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )


if __name__ == "__main__":
    evaluate_popularity(k=10)
    print()
    evaluate_collaborative_filtering(k=10)
