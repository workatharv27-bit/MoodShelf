"""
content_based.py
----------------
Content-based filtering using TF-IDF on book themes + descriptions.
Given a list of target themes (from emotion analysis), returns similarity
scores for each book in the catalog.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),   # unigrams + bigrams
            max_features=500,
            sublinear_tf=True,    # dampens high-frequency terms
        )
        self.tfidf_matrix = None
        self.books_df = None
        self.fitted = False

    def fit(self, books_df: pd.DataFrame):
        """
        Fit the TF-IDF vectorizer on the book corpus.
        Combines: themes + genre + description into a single text field.
        """
        self.books_df = books_df.copy().reset_index(drop=True)

        # Build a rich text representation per book
        self.books_df["content"] = (
            self.books_df["themes"].fillna("") + " " +
            self.books_df["genre"].fillna("") + " " +
            self.books_df["description"].fillna("")
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(self.books_df["content"])
        self.fitted = True
        print(f"[ContentBased] Fitted on {len(self.books_df)} books | "
              f"Vocab size: {len(self.vectorizer.vocabulary_)}")

    def score_by_themes(self, target_themes: list) -> np.ndarray:
        """
        Given a list of target theme strings, returns a similarity score
        array (shape: n_books,) representing how well each book matches.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before scoring.")

        # Build a pseudo-query document from the target themes
        query = " ".join(target_themes)
        query_vec = self.vectorizer.transform([query])

        # Cosine similarity between query and all books
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        return similarities

    def score_by_book(self, book_id: int) -> np.ndarray:
        """
        Given a book_id the user has read/liked, return similarity scores
        to all other books (item-to-item content similarity).
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before scoring.")

        idx = self.books_df[self.books_df["book_id"] == book_id].index
        if len(idx) == 0:
            return np.zeros(len(self.books_df))

        book_vec = self.tfidf_matrix[idx[0]]
        similarities = cosine_similarity(book_vec, self.tfidf_matrix).flatten()
        similarities[idx[0]] = 0   # exclude the book itself
        return similarities

    def score_by_history(self, liked_book_ids: list) -> np.ndarray:
        """
        Average content similarity across multiple liked books.
        This is a simple user profile built from reading history.
        """
        if not liked_book_ids:
            return np.zeros(len(self.books_df))

        scores = np.zeros(len(self.books_df))
        for bid in liked_book_ids:
            scores += self.score_by_book(bid)
        return scores / len(liked_book_ids)

    def get_book_index(self, book_id: int) -> int:
        result = self.books_df[self.books_df["book_id"] == book_id].index
        return result[0] if len(result) > 0 else -1
