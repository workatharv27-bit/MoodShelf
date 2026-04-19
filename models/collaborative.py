"""
collaborative.py
----------------
Collaborative filtering using matrix factorization (SVD via sklearn's
TruncatedSVD). Works with the user-book ratings matrix.

We use sklearn instead of the Surprise library so there are no extra deps.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class CollaborativeModel:
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = None    # (n_users, k)
        self.item_factors = None    # (n_books, k)
        self.user_index = {}        # user_id -> row index
        self.book_index = {}        # book_id -> col index
        self.book_ids = []          # ordered list of book ids
        self.ratings_matrix = None
        self.fitted = False

    def fit(self, ratings_df: pd.DataFrame):
        """
        Build and factorize the user-item ratings matrix.
        ratings_df columns: user_id, book_id, rating
        """
        # Pivot to matrix: rows=users, cols=books
        matrix = ratings_df.pivot_table(
            index="user_id", columns="book_id", values="rating", fill_value=0
        )
        self.ratings_matrix = matrix
        self.user_index = {uid: i for i, uid in enumerate(matrix.index)}
        self.book_index = {bid: i for i, bid in enumerate(matrix.columns)}
        self.book_ids = list(matrix.columns)

        R = matrix.values.astype(float)

        # Normalize rows (user vectors) before decomposition
        R_norm = normalize(R, norm="l2")

        # Fit SVD
        n_comp = min(self.n_components, min(R.shape) - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.user_factors = self.svd.fit_transform(R_norm)  # (n_users, k)
        self.item_factors = self.svd.components_.T           # (n_books, k)
        self.fitted = True

        print(f"[Collaborative] Fitted | Users: {R.shape[0]} | "
              f"Books: {R.shape[1]} | Components: {n_comp} | "
              f"Explained variance: {self.svd.explained_variance_ratio_.sum():.2%}")

    def score_for_user(self, user_id: str) -> np.ndarray:
        """
        Predict ratings for all books for a known user.
        Returns a score array indexed by self.book_ids order.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before scoring.")

        if user_id not in self.user_index:
            return np.zeros(len(self.book_ids))

        u_idx = self.user_index[user_id]
        u_vec = self.user_factors[u_idx]             # (k,)
        scores = self.item_factors @ u_vec           # (n_books,)

        # Zero out books the user already rated
        rated_books = self.ratings_matrix.columns[
            self.ratings_matrix.loc[user_id] > 0
        ].tolist()
        for bid in rated_books:
            if bid in self.book_index:
                scores[self.book_index[bid]] = 0.0

        # Normalize to [0, 1]
        if scores.max() > 0:
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return scores

    def score_for_new_user(self, liked_book_ids: list) -> np.ndarray:
        """
        For a brand-new user with no history in the matrix,
        approximate CF scores via item-item similarity using SVD item factors.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before scoring.")

        scores = np.zeros(len(self.book_ids))
        for bid in liked_book_ids:
            if bid in self.book_index:
                seed_vec = self.item_factors[self.book_index[bid]]
                sims = self.item_factors @ seed_vec
                scores += sims

        if len(liked_book_ids) > 0:
            scores /= len(liked_book_ids)

        if scores.max() > 0:
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return scores

    def scores_to_book_dict(self, scores: np.ndarray) -> dict:
        """Map score array back to {book_id: score} dict."""
        return {bid: float(scores[i]) for i, bid in enumerate(self.book_ids)}
