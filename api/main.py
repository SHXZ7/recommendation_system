from fastapi import FastAPI
import time
import numpy as np
import faiss
from pathlib import Path
import pandas as pd
from pydantic import BaseModel
from datetime import datetime

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

class FeedbackRequest(BaseModel):
    user_id: int
    movie_id: int


def recommend_movies(user_id: int, k: int = 10):
    if user_id not in user_to_idx:
        return []

    start = time.time()

    u_idx = user_to_idx[user_id]
    user_vector = user_embeddings[u_idx:u_idx + 1]
    faiss.normalize_L2(user_vector)

    scores, indices = index.search(user_vector, k + 20)

    seen = user_seen_items.get(user_id, set())
    recommendations = []

    for idx in indices[0]:
        movie_id = int(idx_to_movie[int(idx)])
        if movie_id not in seen:
            recommendations.append({
            "movie_id": movie_id,
            "title": movie_id_to_title.get(movie_id, "Unknown")
        })
        if len(recommendations) == k:
            break

    latency_ms = (time.time() - start) * 1000

    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "latency_ms": round(latency_ms, 2)
    }

@app.get("/recommend/{user_id}")
def get_recommendations(user_id: int, k: int = 10):
    return recommend_movies(user_id, k)

@app.post("/feedback")
def log_feedback(feedback: FeedbackRequest):
    timestamp = datetime.utcnow().isoformat()

    row = f"{feedback.user_id},{feedback.movie_id},{timestamp}\n"

    with open(FEEDBACK_FILE, "a") as f:
        f.write(row)

    return {
        "status": "success",
        "message": "Feedback logged",
        "user_id": feedback.user_id,
        "movie_id": feedback.movie_id
    }
