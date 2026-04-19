"""
prepare_data.py — Converts Book-Crossing dataset → MoodShelf books.csv + ratings.csv

Steps:
  1. Filter explicit ratings only (Rating > 0)
  2. Keep only active users (>= min_user_ratings) and popular books (>= min_book_ratings)
  3. Normalize ratings from 1-10 → 1-5 scale
  4. Infer genre/themes from title keywords (since BX has no genre column)
  5. Save clean books.csv and ratings.csv

Usage:
    python prepare_data.py \
        --books   BX-Books.csv \
        --ratings BX-Book-Ratings.csv \
        --out-dir data/
"""

import argparse, re, os
import pandas as pd
import numpy as np

# ── Keyword → genre/theme inference (since BX has no genre field) ────────────
# Title keywords → (genre, themes)
TITLE_RULES = [
    (r"\b(murder|detective|mystery|crime|killer|death|suspect)\b",
        "Mystery",    "mystery,crime,suspense,detective"),
    (r"\b(love|heart|kiss|romance|passion|desire|wedding)\b",
        "Romance",    "love,romance,relationships,emotion"),
    (r"\b(dragon|magic|wizard|witch|sword|quest|elf|fantasy)\b",
        "Fantasy",    "magic,adventure,fantasy,quest,power"),
    (r"\b(space|galaxy|robot|alien|future|sci.?fi|planet|star)\b",
        "Sci-Fi",     "space,science,future,technology,discovery"),
    (r"\b(war|battle|history|historical|civil|revolution|empire)\b",
        "History",    "history,war,society,power,change"),
    (r"\b(ghost|horror|fear|dark|terror|haunted|evil|nightmare)\b",
        "Horror",     "fear,darkness,suspense,supernatural"),
    (r"\b(life|memoir|story|journey|true|real|biography|diary)\b",
        "Memoir",     "memoir,life,identity,resilience,growth"),
    (r"\b(secret|spy|agent|mission|thriller|danger|escape)\b",
        "Thriller",   "suspense,danger,adventure,survival"),
    (r"\b(child|kids|young|teen|school|family|grow|coming)\b",
        "YA/Family",  "family,growth,coming-of-age,identity"),
    (r"\b(god|faith|spirit|soul|prayer|church|religion|sacred)\b",
        "Spirituality","spirituality,faith,meaning,purpose,peace"),
    (r"\b(success|habit|leader|productivity|mind|think|power)\b",
        "Self-Help",  "growth,productivity,success,mindset,change"),
    (r"\b(cook|recipe|food|eat|kitchen|chef|bak)\b",
        "Food",       "food,creativity,culture,pleasure"),
]
DEFAULT_GENRE  = "Fiction"
DEFAULT_THEMES = "life,story,journey,discovery,humanity"


def infer_genre_themes(title: str):
    t = title.lower()
    for pattern, genre, themes in TITLE_RULES:
        if re.search(pattern, t):
            return genre, themes
    return DEFAULT_GENRE, DEFAULT_THEMES


def normalize_rating(r):
    """Scale 1-10 → 1-5."""
    return round((r / 10.0) * 5, 1)


def run(books_path, ratings_path, out_dir,
        min_book_ratings=20, min_user_ratings=10, max_users=5000):

    os.makedirs(out_dir, exist_ok=True)

    print("[pipeline] Loading raw data...")
    books_raw   = pd.read_csv(books_path,   sep=";", encoding="latin-1", on_bad_lines="skip")
    ratings_raw = pd.read_csv(ratings_path, sep=";", encoding="latin-1", on_bad_lines="skip")

    # Normalize column names
    books_raw.columns   = [c.strip().lower().replace("-","_").replace(" ","_") for c in books_raw.columns]
    ratings_raw.columns = [c.strip().lower().replace("-","_").replace(" ","_") for c in ratings_raw.columns]

    print(f"[pipeline] Raw: {len(books_raw)} books | {len(ratings_raw)} ratings")

    # ── Step 1: explicit ratings only ───────────────────────────────────────
    ratings = ratings_raw[ratings_raw["rating"] > 0].copy()
    print(f"[pipeline] Explicit ratings: {len(ratings)}")

    # ── Step 2: filter active users ─────────────────────────────────────────
    user_counts = ratings["user_id"].value_counts()
    active_users = user_counts[user_counts >= min_user_ratings].index
    ratings = ratings[ratings["user_id"].isin(active_users)]
    print(f"[pipeline] After user filter (>={min_user_ratings} ratings): {len(ratings)} ratings, {len(active_users)} users")

    # Cap users to keep dataset manageable
    if len(active_users) > max_users:
        top_users = user_counts[user_counts >= min_user_ratings].head(max_users).index
        ratings = ratings[ratings["user_id"].isin(top_users)]
        print(f"[pipeline] Capped to top {max_users} users: {len(ratings)} ratings")

    # ── Step 3: filter popular books ────────────────────────────────────────
    book_counts = ratings["isbn"].value_counts()
    popular_books = book_counts[book_counts >= min_book_ratings].index
    ratings = ratings[ratings["isbn"].isin(popular_books)]
    print(f"[pipeline] After book filter (>={min_book_ratings} ratings): {len(ratings)} ratings, {len(popular_books)} books")

    # ── Step 4: build books.csv ─────────────────────────────────────────────
    books_filtered = books_raw[books_raw["isbn"].isin(ratings["isbn"].unique())].copy()

    # Rename columns to MoodShelf schema
    col_map = {"isbn": "book_id", "title": "title", "author": "author",
               "year": "year", "publisher": "publisher"}
    books_filtered = books_filtered.rename(columns=col_map)
    books_filtered = books_filtered[["book_id","title","author"]].drop_duplicates("book_id")

    # Infer genre + themes from title
    books_filtered[["genre","themes"]] = books_filtered["title"].apply(
        lambda t: pd.Series(infer_genre_themes(str(t)))
    )

    # Compute avg_rating from ratings data
    avg_ratings = (
        ratings.groupby("isbn")["rating"]
        .mean()
        .apply(normalize_rating)
        .reset_index()
        .rename(columns={"isbn": "book_id", "rating": "avg_rating"})
    )
    books_filtered = books_filtered.merge(avg_ratings, on="book_id", how="left")

    # Add description = "title by author" (BX has no descriptions)
    books_filtered["description"] = (
        books_filtered["title"] + " by " + books_filtered["author"].fillna("Unknown")
    )

    books_out = books_filtered[["book_id","title","author","genre","themes","description","avg_rating"]]
    books_out = books_out.dropna(subset=["book_id","title"])

    # ── Step 5: build ratings.csv ────────────────────────────────────────────
    ratings_out = ratings[ratings["isbn"].isin(books_out["book_id"])].copy()
    ratings_out = ratings_out.rename(columns={"user_id": "user_id", "isbn": "book_id"})
    ratings_out["rating"] = ratings_out["rating"].apply(normalize_rating)
    ratings_out = ratings_out[["user_id","book_id","rating"]]

    # ── Save ─────────────────────────────────────────────────────────────────
    books_path_out   = os.path.join(out_dir, "books.csv")
    ratings_path_out = os.path.join(out_dir, "ratings.csv")

    books_out.to_csv(books_path_out,   index=False)
    ratings_out.to_csv(ratings_path_out, index=False)

    print(f"\n[pipeline] ✓ Saved {len(books_out)} books    → {books_path_out}")
    print(f"[pipeline] ✓ Saved {len(ratings_out)} ratings → {ratings_path_out}")
    print(f"[pipeline]   Unique users: {ratings_out['user_id'].nunique()}")
    print(f"[pipeline]   Avg ratings per book: {ratings_out.groupby('book_id').size().mean():.1f}")
    print(f"[pipeline]   Avg ratings per user: {ratings_out.groupby('user_id').size().mean():.1f}")
    print("\n✅ Data ready. Now run: python train.py --skip-emotion")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--books",   required=True, help="Path to BX-Books.csv")
    parser.add_argument("--ratings", required=True, help="Path to BX-Book-Ratings.csv")
    parser.add_argument("--out-dir", default="data/", help="Output directory")
    parser.add_argument("--min-book-ratings", type=int, default=20)
    parser.add_argument("--min-user-ratings", type=int, default=10)
    parser.add_argument("--max-users",        type=int, default=5000)
    args = parser.parse_args()

    run(args.books, args.ratings, args.out_dir,
        args.min_book_ratings, args.min_user_ratings, args.max_users)
