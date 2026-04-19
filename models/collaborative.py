"""
collaborative.py — SVD matrix factorization with train/test RMSE evaluation.
"""
import pickle, os
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../saved_models/cf_svd.pkl")


class CollaborativeModel:
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = None
        self.item_factors = None
        self.user_index = {}
        self.book_ids = []
        self.ratings_matrix = None
        self.fitted = False

    def fit(self, ratings_df: pd.DataFrame, evaluate: bool = True, save: bool = True):
        matrix = ratings_df.pivot_table(index="user_id", columns="book_id", values="rating", fill_value=0)
        self.ratings_matrix = matrix
        self.user_index = {uid: i for i, uid in enumerate(matrix.index)}
        self.book_ids = list(matrix.columns)

        R = matrix.values.astype(float)

        if evaluate:
            self._evaluate(ratings_df)

        R_norm = normalize(R, norm="l2")
        n_comp = min(self.n_components, min(R.shape) - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.user_factors = self.svd.fit_transform(R_norm)
        self.item_factors = self.svd.components_.T
        self.fitted = True

        var = self.svd.explained_variance_ratio_.sum()
        print(f"[Collaborative] SVD fitted | components={n_comp} | explained_var={var:.2%}")

        if save:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump((self.svd, self.user_factors, self.item_factors,
                             self.user_index, self.book_ids, self.ratings_matrix), f)

    def _evaluate(self, ratings_df: pd.DataFrame):
        train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
        train_mat = train_df.pivot_table(index="user_id", columns="book_id", values="rating", fill_value=0)

        n_comp = min(self.n_components, min(train_mat.shape) - 1)
        svd_eval = TruncatedSVD(n_components=n_comp, random_state=42)
        UF = svd_eval.fit_transform(normalize(train_mat.values.astype(float), norm="l2"))
        IF = svd_eval.components_.T

        preds, actuals = [], []
        u_idx = {u: i for i, u in enumerate(train_mat.index)}
        b_idx = {b: i for i, b in enumerate(train_mat.columns)}

        for _, row in test_df.iterrows():
            if row["user_id"] in u_idx and row["book_id"] in b_idx:
                p = float(UF[u_idx[row["user_id"]]] @ IF[b_idx[row["book_id"]]])
                preds.append(p)
                actuals.append(row["rating"])

        if preds:
            rmse = mean_squared_error(actuals, preds) ** 0.5
            print(f"[Collaborative] Eval RMSE={rmse:.4f} on {len(preds)} test pairs")

    def load(self):
        with open(MODEL_PATH, "rb") as f:
            (self.svd, self.user_factors, self.item_factors,
             self.user_index, self.book_ids, self.ratings_matrix) = pickle.load(f)
        self.fitted = True

    def score_for_user(self, user_id: str) -> np.ndarray:
        if user_id not in self.user_index:
            return np.zeros(len(self.book_ids))
        u = self.user_factors[self.user_index[user_id]]
        scores = self.item_factors @ u
        rated = self.ratings_matrix.columns[self.ratings_matrix.loc[user_id] > 0].tolist()
        for bid in rated:
            if bid in self.book_ids:
                scores[self.book_ids.index(bid)] = 0.0
        mn, mx = scores.min(), scores.max()
        return (scores - mn) / (mx - mn + 1e-9)

    def score_for_new_user(self, liked_book_ids: list) -> np.ndarray:
        scores = np.zeros(len(self.book_ids))
        for bid in liked_book_ids:
            if bid in self.book_ids:
                v = self.item_factors[self.book_ids.index(bid)]
                scores += self.item_factors @ v
        if liked_book_ids:
            scores /= len(liked_book_ids)
        mn, mx = scores.min(), scores.max()
        return (scores - mn) / (mx - mn + 1e-9)

    def scores_to_dict(self, scores: np.ndarray) -> dict:
        return {bid: float(scores[i]) for i, bid in enumerate(self.book_ids)}
