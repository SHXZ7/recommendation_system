import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_ratings():
    return pd.read_csv(
        RAW_DIR / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

def preprocess(ratings: pd.DataFrame) -> pd.DataFrame:
    # Convert to implicit feedback
    ratings = ratings[ratings["rating"] >= 4].copy()
    ratings["interaction"] = 1

    # Filter sparse users and items
    user_counts = ratings["user_id"].value_counts()
    item_counts = ratings["movie_id"].value_counts()

    ratings = ratings[
        ratings["user_id"].isin(user_counts[user_counts >= 20].index)
        & ratings["movie_id"].isin(item_counts[item_counts >= 20].index)
    ]

    return ratings[["user_id", "movie_id", "interaction", "timestamp"]]

def train_val_split(df: pd.DataFrame):
    df = df.sort_values(["user_id", "timestamp"])

    train_rows = []
    val_rows = []

    for user_id, user_df in df.groupby("user_id"):
        split_idx = int(len(user_df) * 0.8)
        train_rows.append(user_df.iloc[:split_idx])
        val_rows.append(user_df.iloc[split_idx:])

    train_df = pd.concat(train_rows)
    val_df = pd.concat(val_rows)

    return train_df, val_df

if __name__ == "__main__":
    ratings = load_ratings()
    ratings = preprocess(ratings)

    train_df, val_df = train_val_split(ratings)

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)

    print("Preprocessing complete")
    print(f"Train interactions: {len(train_df)}")
    print(f"Validation interactions: {len(val_df)}")
    print(f"Users: {ratings['user_id'].nunique()}")
    print(f"Items: {ratings['movie_id'].nunique()}")
