import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("models/baseline")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_train_data():
    """Load preprocessed training interactions"""
    return pd.read_csv(DATA_DIR / "train.csv")


def build_popularity_model(train_df: pd.DataFrame, top_k: int = 20):
    """
    Build popularity-based recommendations.
    Returns top_k most interacted movies.
    """
    movie_counts = (
        train_df.groupby("movie_id")["interaction"]
        .count()
        .reset_index(name="interaction_count")
        .sort_values("interaction_count", ascending=False)
    )

    return movie_counts.head(top_k)


if __name__ == "__main__":
    train_df = load_train_data()

    top_movies = build_popularity_model(train_df, top_k=20)

    # Save results
    top_movies.to_csv(OUTPUT_DIR / "top_movies.csv", index=False)

    print("Popularity baseline built successfully")
    print(top_movies.head())
