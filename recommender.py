"""
recommender.py — MoodShelf inference pipeline. Loads trained models and runs recommendations.
"""
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.emotion_model import EmotionModel
from models.content_based import ContentBasedModel
from models.collaborative import CollaborativeModel
from models.hybrid        import HybridRecommender

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class MoodShelf:
    def __init__(self):
        self.emotion  = EmotionModel()
        self.content  = ContentBasedModel()
        self.collab   = CollaborativeModel()
        self.hybrid   = HybridRecommender()
        self.books_df = None

    def load(self):
        self.books_df = pd.read_csv(os.path.join(DATA_DIR, "books.csv"))
        self.emotion.load()
        self.content.load()
        self.collab.load()
        try:
            self.hybrid.load()
        except FileNotFoundError:
            print("[MoodShelf] Hybrid meta-learner not found, using fallback weights.")
        print(f"[MoodShelf] Ready. {len(self.books_df)} books loaded.")

    def recommend(self, mood_text: str, user_id: str = None,
                  reading_history: list = None, top_n: int = 5) -> dict:
        reading_history = reading_history or []

        emotion_scores = self.emotion.detect(mood_text)
        themes         = self.emotion.get_relevant_themes(emotion_scores)
        dominant       = self.emotion.dominant_emotion(emotion_scores)

        e_scores  = self.content.score_by_themes(themes)
        c_scores  = 0.6 * e_scores + 0.4 * self.content.score_by_history(reading_history)

        if user_id and user_id in self.collab.user_index:
            cf_arr = self.collab.score_for_user(user_id)
        else:
            cf_arr = self.collab.score_for_new_user(reading_history)
        cf_scores = np.array([self.collab.scores_to_dict(cf_arr).get(bid, 0.0)
                               for bid in self.books_df["book_id"]])

        recs = self.hybrid.recommend(
            self.books_df, e_scores, c_scores, cf_scores,
            exclude_ids=reading_history, top_n=top_n
        )
        explanations = [self.hybrid.explain(row) for _, row in recs.iterrows()]

        return {
            "emotions": emotion_scores,
            "dominant": dominant,
            "themes":   themes,
            "books":    recs,
            "explanations": explanations,
        }

    def pretty_print(self, r: dict):
        print("\n" + "═"*60)
        top3 = list(r["emotions"].items())[:3]
        print(f"  Mood: {', '.join(f'{e}({s:.0%})' for e,s in top3)}")
        print(f"  Themes: {', '.join(r['themes'])}\n")
        for i, (_, row) in enumerate(r["books"].iterrows()):
            print(f"  {i+1}. {row['title']}  [{row['genre']}] ★{row['avg_rating']}")
            print(f"     {r['explanations'][i]}")
            print(f"     e:{row['score_emotion']:.2f} c:{row['score_content']:.2f} "
                  f"cf:{row['score_collab']:.2f} → {row['score_final']:.2f}\n")
        print("═"*60)
