"""
train.py — Train all MoodShelf models.

Steps:
  1. Download GoEmotions (once) → train EmotionModel
  2. Train Word2Vec on book corpus → ContentBasedModel
  3. Train SVD on ratings → CollaborativeModel
  4. Generate synthetic feature matrix → train HybridRecommender

Run:
    python train.py
    python train.py --skip-emotion   (if you already trained emotion model)
"""
import argparse, os, sys, urllib.request
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from models.emotion_model  import EmotionModel
from models.content_based  import ContentBasedModel
from models.collaborative  import CollaborativeModel
from models.hybrid         import HybridRecommender

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

GOEMOTIONS_URL = "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv"
GOEMOTIONS_TSV = os.path.join(DATA_DIR, "goemotions_train.tsv")


def download_goemotions():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(GOEMOTIONS_TSV):
        print("[train] GoEmotions already downloaded.")
        return

    print("[train] Downloading GoEmotions dataset (~9MB)...")
    csv_path = os.path.join(DATA_DIR, "goemotions_raw.csv")
    urllib.request.urlretrieve(GOEMOTIONS_URL, csv_path)

    # Convert CSV to the TSV format our EmotionModel expects
    df = pd.read_csv(csv_path)
    # columns: text, emotion_ids (space-separated), ... 
    # Normalize to TSV: text | comma-separated-ids | split
    df["labels"] = df.iloc[:, 1:29].apply(
        lambda row: ",".join([str(i) for i, v in enumerate(row) if v == 1]), axis=1
    )
    df["split"] = "train"
    df[["text","labels","split"]].to_csv(GOEMOTIONS_TSV, sep="\t", index=False, header=False)
    print(f"[train] Saved → {GOEMOTIONS_TSV}")


def train_emotion(skip=False):
    if skip:
        print("[train] Skipping emotion model training.")
        return
    download_goemotions()
    model = EmotionModel()
    model.train(GOEMOTIONS_TSV)
    print("[train] ✓ EmotionModel saved.")


def train_content():
    books_df = pd.read_csv(os.path.join(DATA_DIR, "books.csv"))
    model = ContentBasedModel(dim=64)
    model.fit(books_df)
    print("[train] ✓ ContentBasedModel (Word2Vec) saved.")


def train_collab():
    ratings_df = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    model = CollaborativeModel(n_components=10)
    model.fit(ratings_df, evaluate=True)
    print("[train] ✓ CollaborativeModel (SVD) saved.")


def train_hybrid():
    """
    Build training data for the meta-learner:
    For each (user, book) pair in ratings, compute the 3 signal scores
    and use the actual rating as the target.
    """
    books_df   = pd.read_csv(os.path.join(DATA_DIR, "books.csv"))
    ratings_df = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))

    content_model = ContentBasedModel()
    content_model.load()
    collab_model = CollaborativeModel()
    collab_model.load()

    # We don't have per-rating mood text, so we approximate emotion score
    # by computing how well each book matches each core emotion's theme prior
    # using the trained Word2Vec content model.
    from models.emotion_model import EMOTION_THEME_PRIOR, CORE_EMOTIONS

    # Precompute per-emotion content scores for every book
    emotion_book_scores = {}
    for em in CORE_EMOTIONS:
        themes = EMOTION_THEME_PRIOR[em]
        emotion_book_scores[em] = content_model.score_by_themes(themes)

    features, targets = [], []

    for _, row in ratings_df.iterrows():
        uid  = row["user_id"]
        bid  = row["book_id"]
        rating = row["rating"] / 5.0   # normalize to [0,1]

        # Emotion score: pick the emotion whose themes best match this book
        b_idx_list = books_df[books_df["book_id"] == bid].index
        if len(b_idx_list) == 0:
            continue
        b_idx = b_idx_list[0]
        e_score = max(emotion_book_scores[em][b_idx] for em in CORE_EMOTIONS)

        # Content score: similarity between this book and user's other rated books
        other_books = ratings_df[(ratings_df["user_id"] == uid) & (ratings_df["book_id"] != bid)]["book_id"].tolist()
        c_arr = content_model.score_by_history(other_books)
        c_score = float(c_arr[b_idx]) if b_idx < len(c_arr) else 0.0

        # CF score
        cf_arr = collab_model.score_for_user(uid)
        cf_dict = collab_model.scores_to_dict(cf_arr)
        cf_score = cf_dict.get(bid, 0.0)

        features.append([e_score, c_score, cf_score])
        targets.append(rating)

    X = np.array(features)
    y = np.array(targets)
    print(f"[train] Hybrid training samples: {len(X)}")

    hybrid = HybridRecommender()
    hybrid.train(X, y)
    print("[train] ✓ HybridRecommender (Ridge meta-learner) saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-emotion", action="store_true", help="Skip emotion model training")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    print("\n=== MoodShelf Training Pipeline ===\n")

    train_emotion(skip=args.skip_emotion)
    train_content()
    train_collab()
    train_hybrid()

    print("\n✅ All models trained and saved to ./saved_models/")
