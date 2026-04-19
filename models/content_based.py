"""
content_based.py — Word2Vec trained on book descriptions + themes.
Book vectors = mean of word vectors. Similarity via cosine.
"""
import pickle, os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../saved_models/w2v.pkl")


def _tokenize(text: str) -> list:
    import re
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


class Word2VecSimple:
    """Lightweight skip-gram Word2Vec using numpy — no gensim needed."""

    def __init__(self, dim: int = 64, window: int = 3, epochs: int = 15, lr: float = 0.01):
        self.dim = dim
        self.window = window
        self.epochs = epochs
        self.lr = lr
        self.W = None   # input embeddings  (vocab, dim)
        self.C = None   # context embeddings (vocab, dim)
        self.vocab = {}
        self.id2word = []

    def _build_vocab(self, sentences):
        freq = {}
        for s in sentences:
            for w in s:
                freq[w] = freq.get(w, 0) + 1
        self.id2word = [w for w, c in sorted(freq.items(), key=lambda x: -x[1]) if c >= 1]
        self.vocab = {w: i for i, w in enumerate(self.id2word)}

    def _pairs(self, sentences):
        for s in sentences:
            ids = [self.vocab[w] for w in s if w in self.vocab]
            for i, center in enumerate(ids):
                ctx = ids[max(0,i-self.window):i] + ids[i+1:i+self.window+1]
                for c in ctx:
                    yield center, c

    def train(self, sentences):
        self._build_vocab(sentences)
        V = len(self.vocab)
        self.W = np.random.randn(V, self.dim).astype(np.float32) * 0.01
        self.C = np.random.randn(V, self.dim).astype(np.float32) * 0.01

        pairs = list(self._pairs(sentences))
        if not pairs:
            return

        for epoch in range(self.epochs):
            np.random.shuffle(pairs)
            loss = 0.0
            for center, ctx in pairs:
                # positive
                h = self.W[center]
                c = self.C[ctx]
                score = np.dot(h, c)
                sig = 1 / (1 + np.exp(-np.clip(score, -10, 10)))
                grad = self.lr * (1 - sig)
                self.W[center] += grad * c
                self.C[ctx]    += grad * h
                loss += -np.log(sig + 1e-9)
                # 3 negatives
                negs = np.random.randint(0, V, 3)
                for n in negs:
                    s2 = np.dot(self.W[center], self.C[n])
                    sig2 = 1 / (1 + np.exp(-np.clip(s2, -10, 10)))
                    g2 = self.lr * (-sig2)
                    self.W[center] += g2 * self.C[n]
                    self.C[n]      += g2 * self.W[center]
                    loss += -np.log(1 - sig2 + 1e-9)

            if (epoch + 1) % 5 == 0:
                print(f"  [W2V] epoch {epoch+1}/{self.epochs}  loss={loss/len(pairs):.4f}")

    def vector(self, word: str) -> np.ndarray | None:
        idx = self.vocab.get(word)
        return self.W[idx] if idx is not None else None

    def sentence_vector(self, tokens: list) -> np.ndarray:
        vecs = [self.W[self.vocab[w]] for w in tokens if w in self.vocab]
        return np.mean(vecs, axis=0) if vecs else np.zeros(self.dim)


class ContentBasedModel:
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.w2v = Word2VecSimple(dim=dim)
        self.book_vectors = None   # (n_books, dim)
        self.books_df = None
        self.fitted = False

    def fit(self, books_df: pd.DataFrame, save: bool = True):
        self.books_df = books_df.copy().reset_index(drop=True)

        corpus_text = (
            books_df["themes"].fillna("") + " " +
            books_df["genre"].fillna("") + " " +
            books_df["description"].fillna("")
        )
        sentences = [_tokenize(t) for t in corpus_text]

        print("[ContentBased] Training Word2Vec...")
        self.w2v.train(sentences)

        self.book_vectors = np.array([self.w2v.sentence_vector(s) for s in sentences])
        self.fitted = True
        print(f"[ContentBased] Book vectors: {self.book_vectors.shape}")

        if save:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump((self.w2v, self.book_vectors, self.books_df), f)

    def load(self):
        with open(MODEL_PATH, "rb") as f:
            self.w2v, self.book_vectors, self.books_df = pickle.load(f)
        self.fitted = True

    def score_by_themes(self, themes: list) -> np.ndarray:
        query_vec = self.w2v.sentence_vector(themes).reshape(1, -1)
        sims = cosine_similarity(query_vec, self.book_vectors).flatten()
        return sims

    def score_by_history(self, book_ids: list) -> np.ndarray:
        if not book_ids:
            return np.zeros(len(self.books_df))
        idxs = [self.books_df[self.books_df["book_id"] == bid].index for bid in book_ids]
        idxs = [i[0] for i in idxs if len(i) > 0]
        if not idxs:
            return np.zeros(len(self.books_df))
        profile = np.mean(self.book_vectors[idxs], axis=0, keepdims=True)
        sims = cosine_similarity(profile, self.book_vectors).flatten()
        for i in idxs:
            sims[i] = 0.0
        return sims
