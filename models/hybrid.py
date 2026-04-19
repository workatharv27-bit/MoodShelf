"""
hybrid.py — Ridge Regression meta-learner that learns optimal signal blending.

Instead of fixed weights, we train a Ridge model on (emotion_score,
content_score, collab_score) → actual_rating. This learns which signals
matter more per context from real feedback data.
"""
import pickle, os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../saved_models/hybrid_meta.pkl")


class HybridRecommender:
    def __init__(self):
        self.meta = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.fitted = False
        # fallback weights if meta-learner not trained yet
        self._fallback = np.array([0.40, 0.35, 0.25])

    def train(self, feature_matrix: np.ndarray, targets: np.ndarray, save: bool = True):
        """
        feature_matrix: (n_samples, 3) — [emotion_score, content_score, collab_score]
        targets:        (n_samples,)   — actual user ratings (normalized 0-1)
        """
        X = self.scaler.fit_transform(feature_matrix)
        cv_scores = cross_val_score(self.meta, X, targets, cv=3, scoring="neg_mean_squared_error")
        rmse = (-cv_scores.mean()) ** 0.5
        print(f"[HybridMeta] CV RMSE={rmse:.4f}")
        self.meta.fit(X, targets)
        self.fitted = True
        print(f"[HybridMeta] Learned coefficients: emotion={self.meta.coef_[0]:.3f} "
              f"content={self.meta.coef_[1]:.3f} collab={self.meta.coef_[2]:.3f}")
        if save:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump((self.meta, self.scaler), f)

    def load(self):
        with open(MODEL_PATH, "rb") as f:
            self.meta, self.scaler = pickle.load(f)
        self.fitted = True

    def _norm(self, arr):
        arr = np.array(arr, dtype=float)
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9) if mx > mn else arr

    def recommend(self, books_df, e_scores, c_scores, cf_scores,
                  exclude_ids=None, top_n=5):
        e = self._norm(e_scores)
        c = self._norm(c_scores)
        cf = self._norm(cf_scores)

        features = np.stack([e, c, cf], axis=1)   # (n_books, 3)

        if self.fitted:
            X = self.scaler.transform(features)
            final = self.meta.predict(X)
        else:
            final = features @ self._fallback

        results = books_df.copy().reset_index(drop=True)
        results["score_emotion"] = e
        results["score_content"] = c
        results["score_collab"]  = cf
        results["score_final"]   = final

        if exclude_ids:
            results = results[~results["book_id"].isin(exclude_ids)]

        results = results.sort_values("score_final", ascending=False).head(top_n)
        return results[["book_id","title","author","genre","themes","description",
                         "avg_rating","score_emotion","score_content",
                         "score_collab","score_final"]].reset_index(drop=True)

    def explain(self, row) -> str:
        parts = []
        if row["score_emotion"] > 0.5:  parts.append("strongly matches your mood")
        elif row["score_emotion"] > 0.2: parts.append("fits your emotional state")
        if row["score_content"] > 0.5:  parts.append("aligned with your reading taste")
        elif row["score_content"] > 0.2: parts.append("similar to books you've read")
        if row["score_collab"] > 0.5:   parts.append("loved by readers like you")
        elif row["score_collab"] > 0.2:  parts.append("popular with similar readers")
        return ("📖 " + ", ".join(parts or ["a well-rounded pick for you"]).capitalize() + ".")
