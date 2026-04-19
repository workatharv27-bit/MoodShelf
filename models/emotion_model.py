"""
emotion_model.py — TF-IDF + Logistic Regression trained on GoEmotions.
"""
import pickle, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../saved_models/emotion_clf.pkl")

LABEL_NAMES = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization",
    "relief","remorse","sadness","surprise","neutral"
]

GOEMOTIONS_MAP = {
    "joy":        ["joy","amusement","excitement","gratitude","love","optimism","relief","pride"],
    "sadness":    ["sadness","grief","disappointment","remorse"],
    "anxiety":    ["fear","nervousness"],
    "anger":      ["anger","annoyance","disapproval","disgust"],
    "curiosity":  ["curiosity","surprise","realization","confusion"],
    "calm":       ["admiration","approval","caring","neutral"],
    "loneliness": ["embarrassment","desire"],
    "hope":       ["optimism","relief","gratitude"],
}
CORE_EMOTIONS = list(GOEMOTIONS_MAP.keys())

EMOTION_THEME_PRIOR = {
    "joy":        ["adventure","humor","journey","friendship","love"],
    "sadness":    ["healing","grief","resilience","meaning","memoir"],
    "anxiety":    ["calm","mindfulness","peace","acceptance","clarity"],
    "anger":      ["justice","philosophy","truth","purpose","society"],
    "curiosity":  ["history","science","discovery","thinking","evolution"],
    "calm":       ["wisdom","spirituality","nature","presence","simplicity"],
    "loneliness": ["connection","empathy","love","identity","belonging"],
    "hope":       ["dreams","growth","change","purpose","inspiration"],
}


class EmotionModel:
    def __init__(self):
        self.pipeline = None
        self.mlb = None
        self.fitted = False

    def _map_labels(self, raw_labels):
        inv = {s: core for core, subs in GOEMOTIONS_MAP.items() for s in subs}
        mapped = list({inv[l] for l in raw_labels if l in inv})
        return mapped or ["calm"]

    def train(self, train_tsv: str, save: bool = True):
        df = pd.read_csv(train_tsv, sep="\t", header=None, names=["text","labels","split"])
        df = df.dropna(subset=["text","labels"])

        def decode(label_str):
            indices = [int(x) for x in str(label_str).split(",") if x.strip().isdigit()]
            raw = [LABEL_NAMES[i] for i in indices if i < len(LABEL_NAMES)]
            return self._map_labels(raw)

        df["core"] = df["labels"].apply(decode)
        self.mlb = MultiLabelBinarizer(classes=CORE_EMOTIONS)
        Y = self.mlb.fit_transform(df["core"])

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=30000, sublinear_tf=True)),
            ("clf",   OneVsRestClassifier(LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")))
        ])
        self.pipeline.fit(df["text"], Y)
        self.fitted = True
        print(f"[EmotionModel] Trained on {len(df)} samples.")

        if save:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump((self.pipeline, self.mlb), f)

    def load(self):
        with open(MODEL_PATH, "rb") as f:
            self.pipeline, self.mlb = pickle.load(f)
        self.fitted = True

    def detect(self, text: str) -> dict:
        if not self.fitted:
            raise RuntimeError("Call train() or load() first.")
        probs = self.pipeline.predict_proba([text])
        scores = {e: float(probs[i][0][1]) for i, e in enumerate(CORE_EMOTIONS)}
        total = sum(scores.values()) or 1.0
        return dict(sorted({e: round(s/total, 4) for e, s in scores.items()}.items(), key=lambda x: -x[1]))

    def dominant_emotion(self, scores: dict) -> str:
        return max(scores, key=scores.get)

    def get_relevant_themes(self, scores: dict, top_n: int = 6) -> list:
        from collections import defaultdict
        w = defaultdict(float)
        for emotion, score in scores.items():
            for theme in EMOTION_THEME_PRIOR.get(emotion, []):
                w[theme] += score
        return [t for t, _ in sorted(w.items(), key=lambda x: -x[1])[:top_n]]
