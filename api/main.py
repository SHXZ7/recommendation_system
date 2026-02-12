from fastapi import FastAPI
import time
import numpy as np
import faiss
from pathlib import Path
import pandas as pd
from pydantic import BaseModel
from datetime import datetime
import random

app = FastAPI(title="Real-Time Recommendation API")

DATA_DIR = Path("data/processed")
CF_DIR = Path("models/collaborative")
RETRIEVAL_DIR = Path("retrieval")
FEEDBACK_FILE = Path("monitoring/feedback_log.csv")
MOVIES_FILE = Path("data/raw/movies.dat")

user_embeddings = np.load(CF_DIR / "user_embeddings.npy").astype("float32")
movie_embeddings = np.load(CF_DIR / "movie_embeddings.npy").astype("float32")

index = faiss.read_index(str(RETRIEVAL_DIR / "movie_index.faiss"))

train_df = pd.read_csv(DATA_DIR / "train.csv")

# Precompute popularity scores
popularity_counts = train_df["movie_id"].value_counts().to_dict()
max_popularity = max(popularity_counts.values())


user_ids = train_df["user_id"].unique()
movie_ids = train_df["movie_id"].unique()

user_to_idx = {u: i for i, u in enumerate(user_ids)}
idx_to_movie = {i: m for i, m in enumerate(movie_ids)}

user_seen_items = (
    train_df.groupby("user_id")["movie_id"]
    .apply(set)
    .to_dict()
)

movies_df = pd.read_csv(
    MOVIES_FILE,
    sep="::",
    engine="python",
    encoding="latin-1",
    names=["movie_id", "title", "genres"]
)


movie_id_to_title = dict(
    zip(movies_df.movie_id, movies_df.title)
)

def get_popular_movies(k=10):
    # Get top 50 popular movies
    popular_movies = (
        train_df["movie_id"]
        .value_counts()
        .head(50)
        .index.tolist()
    )

    diverse_movies = []
    selected_genres = set()

    for movie_id in popular_movies:
        movie_row = movies_df[movies_df.movie_id == movie_id]

        if movie_row.empty:
            continue

        genres = movie_row["genres"].values[0]
        primary_genre = genres.split("|")[0]

        # Ensure genre diversity
        if primary_genre not in selected_genres:
            diverse_movies.append(movie_id)
            selected_genres.add(primary_genre)

        if len(diverse_movies) == k:
            break

    return [
        {
            "movie_id": int(mid),
            "title": movie_id_to_title.get(mid, "Unknown"),
            "reason": "Popular among all users"
        }
        for mid in diverse_movies
    ]


class FeedbackRequest(BaseModel):
    user_id: int
    movie_id: int
    experiment_group: str

def recommend_movies(user_id: int, k: int = 10):

    # Smart Cold-Start
    if user_id not in user_to_idx:
        return {
            "user_id": user_id,
            "recommendations": get_popular_movies(k),
            "latency_ms": 0.0
        }

    # A/B experiment assignment
    experiment_group = random.choice(["control", "hybrid"])

    start = time.time()

    u_idx = user_to_idx[user_id]
    user_vector = user_embeddings[u_idx:u_idx + 1]
    faiss.normalize_L2(user_vector)

    # Over-fetch for filtering
    scores, indices = index.search(user_vector, k + 20)

    seen = user_seen_items.get(user_id, set())

    hybrid_candidates = []

    for rank_position, idx in enumerate(indices[0]):
        movie_id = int(idx_to_movie[int(idx)])

        if movie_id in seen:
            continue

        cf_score = 1 - (rank_position / len(indices[0]))

        pop_score = popularity_counts.get(movie_id, 0) / max_popularity

        if experiment_group == "control":
            final_score = cf_score
        else:
            final_score = 0.8 * cf_score + 0.2 * pop_score

        hybrid_candidates.append((movie_id, final_score, pop_score))

    # Sort by hybrid score descending
    hybrid_candidates = sorted(
        hybrid_candidates,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []
    selected_genres = set()

    for movie_id, hybrid_score, pop_score in hybrid_candidates:

        # Get movie genre
        movie_row = movies_df[movies_df.movie_id == movie_id]
        if movie_row.empty:
            continue

        genres = movie_row["genres"].values[0]
        primary_genre = genres.split("|")[0]

        # Enforce diversity
        if primary_genre in selected_genres:
            continue

        # Determine explanation
        if pop_score > 0.7:
            reason = "Popular among similar users"
        else:
            reason = "Based on your viewing history"

        recommendations.append({
            "movie_id": movie_id,
            "title": movie_id_to_title.get(movie_id, "Unknown"),
            "reason": reason
        })

        selected_genres.add(primary_genre)

        if len(recommendations) == k:
            break

    latency_ms = (time.time() - start) * 1000

    return {
        "user_id": user_id,
        "experiment_group": experiment_group,
        "recommendations": recommendations,
        "latency_ms": round(latency_ms, 2)
    }


@app.get("/recommend/{user_id}")
def get_recommendations(user_id: int, k: int = 10):
    return recommend_movies(user_id, k)

@app.post("/feedback")
def log_feedback(feedback: FeedbackRequest):
    timestamp = datetime.utcnow().isoformat()

    row = f"{feedback.user_id},{feedback.movie_id},{feedback.experiment_group},{timestamp}\n"

    with open(FEEDBACK_FILE, "a") as f:
        f.write(row)

    return {
        "status": "success",
        "message": "Feedback logged",
        "user_id": feedback.user_id,
        "movie_id": feedback.movie_id
    }
