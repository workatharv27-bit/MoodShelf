"""
hybrid.py
---------
Fuses three score signals into a final ranked list:

  1. Mood/Emotion score  — how well a book's themes match the user's emotional state
  2. Content score       — TF-IDF similarity based on themes + reading history
  3. Collaborative score — SVD-based predicted rating from similar users

The weights are tunable and can be adjusted per use case.
"""

import numpy as np
import pandas as pd


class HybridRecommender:
    def __init__(
        self,
        emotion_weight: float = 0.40,
        content_weight: float = 0.35,
        collab_weight: float = 0.25,
    ):
        """
        Default weights:
        - Emotion is prioritised slightly (this is our unique angle)
        - Content provides personalized theme matching
        - Collaborative adds social signal
        """
        total = emotion_weight + content_weight + collab_weight
        self.w_emotion = emotion_weight / total
        self.w_content = content_weight / total
        self.w_collab = collab_weight / total

    def recommend(
        self,
        books_df: pd.DataFrame,
        emotion_scores_by_book: np.ndarray,   # shape (n_books,)
        content_scores_by_book: np.ndarray,   # shape (n_books,)
        collab_scores_by_book: np.ndarray,    # shape (n_books,)
        exclude_book_ids: list = None,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Combine signals and return top N recommended books as a DataFrame.
        """
        def safe_norm(arr):
            arr = np.array(arr, dtype=float)
            mn, mx = arr.min(), arr.max()
            if mx - mn < 1e-9:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        e = safe_norm(emotion_scores_by_book)
        c = safe_norm(content_scores_by_book)
        cf = safe_norm(collab_scores_by_book)

        final_scores = self.w_emotion * e + self.w_content * c + self.w_collab * cf

        results = books_df.copy().reset_index(drop=True)
        results["score_emotion"] = e
        results["score_content"] = c
        results["score_collab"] = cf
        results["score_final"] = final_scores

        # Exclude already-read books
        if exclude_book_ids:
            results = results[~results["book_id"].isin(exclude_book_ids)]

        results = results.sort_values("score_final", ascending=False).head(top_n)
        return results[
            ["book_id", "title", "author", "genre", "themes",
             "description", "avg_rating",
             "score_emotion", "score_content", "score_collab", "score_final"]
        ].reset_index(drop=True)

    def explain(self, row: pd.Series) -> str:
        """
        Generate a short human-readable explanation for a recommendation.
        """
        parts = []
        if row["score_emotion"] > 0.5:
            parts.append("strongly matches your current mood")
        elif row["score_emotion"] > 0.2:
            parts.append("fits your emotional state")

        if row["score_content"] > 0.5:
            parts.append("closely aligned with your reading taste")
        elif row["score_content"] > 0.2:
            parts.append("similar to books you've enjoyed")

        if row["score_collab"] > 0.5:
            parts.append("highly rated by readers like you")
        elif row["score_collab"] > 0.2:
            parts.append("popular with similar readers")

        if not parts:
            parts.append("a well-rounded pick for you")

        return "📖 " + ", ".join(parts).capitalize() + "."
