"""
recommender.py
--------------
Main pipeline for MoodShelf. Wires together:
  - EmotionModel
  - ContentBasedModel
  - CollaborativeModel
  - HybridRecommender

Usage:
    from recommender import MoodShelf
    ms = MoodShelf()
    ms.load_data()
    results = ms.recommend(
        mood_text="I'm feeling overwhelmed and anxious lately",
        user_id="u3",
        reading_history=[4, 6],
        top_n=5
    )
"""

import numpy as np
import pandas as pd
from pathlib import Path

from models.emotion_model import EmotionModel
from models.content_based import ContentBasedModel
from models.collaborative import CollaborativeModel
from models.hybrid import HybridRecommender


DATA_DIR = Path(__file__).parent / "data"


class MoodShelf:
    def __init__(
        self,
        emotion_weight: float = 0.40,
        content_weight: float = 0.35,
        collab_weight: float = 0.25,
    ):
        self.emotion_model = EmotionModel()
        self.content_model = ContentBasedModel()
        self.collab_model = CollaborativeModel(n_components=10)
        self.hybrid = HybridRecommender(emotion_weight, content_weight, collab_weight)
        self.books_df = None
        self.ratings_df = None
        self._fitted = False

    def load_data(
        self,
        books_path: str = None,
        ratings_path: str = None,
    ):
        books_path = books_path or DATA_DIR / "books.csv"
        ratings_path = ratings_path or DATA_DIR / "ratings.csv"

        self.books_df = pd.read_csv(books_path)
        self.ratings_df = pd.read_csv(ratings_path)

        print(f"[MoodShelf] Loaded {len(self.books_df)} books | "
              f"{len(self.ratings_df)} ratings")

        self.content_model.fit(self.books_df)
        self.collab_model.fit(self.ratings_df)
        self._fitted = True

    def recommend(
        self,
        mood_text: str,
        user_id: str = None,
        reading_history: list = None,
        top_n: int = 5,
    ) -> dict:
        """
        Full recommendation pipeline.

        Args:
            mood_text:       Free-text description of current mood / journal entry.
            user_id:         Known user ID (for collaborative filtering).
            reading_history: List of book_ids the user has already read.
            top_n:           Number of recommendations to return.

        Returns:
            A dict with:
              - "emotions":     detected emotion scores
              - "themes":       target themes derived from mood
              - "books":        DataFrame of recommended books with scores
              - "explanations": list of explanation strings
        """
        if not self._fitted:
            raise RuntimeError("Call load_data() first.")

        reading_history = reading_history or []
        n_books = len(self.books_df)

        # ── Step 1: Detect emotions ──────────────────────────────────────────
        emotion_scores = self.emotion_model.detect(mood_text)
        target_themes = self.emotion_model.get_relevant_themes(emotion_scores)
        dominant = self.emotion_model.dominant_emotion(emotion_scores)

        print(f"\n[Pipeline] Dominant emotion: {dominant}")
        print(f"[Pipeline] Top emotions: {dict(list(emotion_scores.items())[:3])}")
        print(f"[Pipeline] Target themes: {target_themes[:6]}")

        # ── Step 2: Content scores (mood themes) ────────────────────────────
        emotion_content_scores = self.content_model.score_by_themes(target_themes)

        # ── Step 3: Content scores (reading history) ────────────────────────
        history_content_scores = self.content_model.score_by_history(reading_history)

        # Blend the two content signals (60% mood, 40% history)
        combined_content = 0.6 * emotion_content_scores + 0.4 * history_content_scores

        # ── Step 4: Collaborative scores ────────────────────────────────────
        if user_id and user_id in self.collab_model.user_index:
            cf_dict = self.collab_model.scores_to_book_dict(
                self.collab_model.score_for_user(user_id)
            )
        else:
            cf_dict = self.collab_model.scores_to_book_dict(
                self.collab_model.score_for_new_user(reading_history)
            )

        # Map CF scores to book order in books_df
        cf_scores = np.array([
            cf_dict.get(bid, 0.0) for bid in self.books_df["book_id"]
        ])

        # ── Step 5: Hybrid fusion ────────────────────────────────────────────
        recs = self.hybrid.recommend(
            books_df=self.books_df,
            emotion_scores_by_book=emotion_content_scores,
            content_scores_by_book=combined_content,
            collab_scores_by_book=cf_scores,
            exclude_book_ids=reading_history,
            top_n=top_n,
        )

        explanations = [self.hybrid.explain(row) for _, row in recs.iterrows()]

        return {
            "emotions": emotion_scores,
            "dominant_emotion": dominant,
            "themes": target_themes[:6],
            "books": recs,
            "explanations": explanations,
        }

    def pretty_print(self, results: dict):
        print("\n" + "═" * 60)
        print("  🌙  MoodShelf Recommendations")
        print("═" * 60)

        em = results["emotions"]
        top_ems = list(em.items())[:3]
        print(f"\n  Detected mood:  {', '.join(f'{k} ({v:.0%})' for k, v in top_ems)}")
        print(f"  Target themes:  {', '.join(results['themes'])}")
        print()

        for i, (_, row) in enumerate(results["books"].iterrows()):
            exp = results["explanations"][i]
            print(f"  {i+1}. {row['title']}  by {row['author']}")
            print(f"     Genre: {row['genre']}  |  ★ {row['avg_rating']}")
            print(f"     {exp}")
            score_bar = (
                f"     Scores → emotion:{row['score_emotion']:.2f}  "
                f"content:{row['score_content']:.2f}  "
                f"collab:{row['score_collab']:.2f}  "
                f"final:{row['score_final']:.2f}"
            )
            print(score_bar)
            print()
        print("═" * 60)
