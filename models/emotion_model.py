"""
emotion_model.py
----------------
Detects emotions from user-written text using a keyword-based + lexicon approach.
Returns a dict of emotion scores (0.0 - 1.0) across 8 core emotions.

For local use without heavy transformers. Can be swapped for a HuggingFace
DistilBERT model if you have GPU/RAM available (see the commented section).
"""

import re
from collections import defaultdict

# ── Emotion lexicon ──────────────────────────────────────────────────────────
# Each emotion maps to keywords that strongly signal it.
EMOTION_LEXICON = {
    "joy": [
        "happy", "excited", "great", "wonderful", "fantastic", "elated",
        "joyful", "cheerful", "grateful", "blessed", "thrilled", "delighted",
        "amazing", "love", "celebrate", "fun", "laugh", "smile", "positive",
        "awesome", "energetic", "motivated", "inspired", "hopeful", "lucky"
    ],
    "sadness": [
        "sad", "depressed", "down", "unhappy", "miserable", "grief",
        "lonely", "heartbroken", "crying", "tears", "lost", "empty",
        "hopeless", "devastated", "sorrow", "mourn", "blue", "gloomy",
        "upset", "hurt", "pain", "miss", "missing", "abandoned", "alone"
    ],
    "anxiety": [
        "anxious", "worried", "nervous", "stressed", "overwhelmed", "panic",
        "fear", "scared", "afraid", "tense", "uneasy", "restless",
        "dread", "apprehensive", "jittery", "paranoid", "uncertain",
        "pressure", "too much", "can't cope", "out of control", "burden"
    ],
    "anger": [
        "angry", "furious", "rage", "frustrated", "annoyed", "irritated",
        "mad", "bitter", "resentful", "outraged", "hateful", "hostile",
        "betrayed", "fed up", "sick of", "tired of", "hate", "livid",
        "infuriated", "disgusted", "unfair", "injustice", "wrong"
    ],
    "curiosity": [
        "curious", "wondering", "interested", "fascinated", "intrigued",
        "explore", "discover", "learn", "question", "research", "investigate",
        "find out", "understand", "how", "why", "what if", "suppose",
        "thinking about", "pondering", "reflect", "contemplating"
    ],
    "calm": [
        "calm", "peaceful", "relaxed", "serene", "tranquil", "quiet",
        "still", "at ease", "content", "comfortable", "settled", "centered",
        "mindful", "present", "okay", "fine", "balanced", "steady",
        "breathing", "rest", "slow", "gentle", "soft"
    ],
    "loneliness": [
        "lonely", "isolated", "alone", "disconnected", "left out",
        "no one", "nobody", "friendless", "forgotten", "invisible",
        "excluded", "withdrawn", "social", "missing connection",
        "need someone", "no friends", "apart", "separate"
    ],
    "hope": [
        "hope", "optimistic", "looking forward", "better", "future",
        "new beginning", "change", "grow", "improve", "possible",
        "opportunity", "potential", "believe", "can do", "will",
        "someday", "dream", "aspire", "try again", "recovery", "healing"
    ],
}

# ── Emotion → Genre/Theme mapping ───────────────────────────────────────────
# What kind of books fit which emotional state?
EMOTION_THEME_MAP = {
    "joy":        ["adventure", "humor", "journey", "friendship", "love", "celebration"],
    "sadness":    ["healing", "grief", "resilience", "meaning", "love", "memoir"],
    "anxiety":    ["calm", "mindfulness", "peace", "self-help", "acceptance", "clarity"],
    "anger":      ["justice", "philosophy", "truth", "purpose", "growth", "society"],
    "curiosity":  ["history", "science", "society", "evolution", "discovery", "thinking"],
    "calm":       ["wisdom", "spirituality", "philosophy", "nature", "presence", "simplicity"],
    "loneliness": ["connection", "empathy", "love", "coming-of-age", "identity", "belonging"],
    "hope":       ["dreams", "growth", "change", "purpose", "resilience", "inspiration"],
}


class EmotionModel:
    def __init__(self):
        self.lexicon = EMOTION_LEXICON
        self.theme_map = EMOTION_THEME_MAP

    def _preprocess(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text

    def detect(self, text: str) -> dict:
        """
        Returns a dict of emotion -> score (0.0 to 1.0).
        Scores are normalized so they sum to 1.0.
        """
        processed = self._preprocess(text)
        words = processed.split()
        scores = defaultdict(float)

        for emotion, keywords in self.lexicon.items():
            for kw in keywords:
                kw_words = kw.split()
                if len(kw_words) == 1:
                    if kw in words:
                        scores[emotion] += 1.0
                else:
                    if kw in processed:
                        scores[emotion] += 1.5   # multi-word phrases weighted higher

        # If no matches found, default to mild curiosity + calm
        if sum(scores.values()) == 0:
            scores["curiosity"] = 0.5
            scores["calm"] = 0.5
        
        # Normalize to sum to 1
        total = sum(scores.values())
        return {e: round(s / total, 4) for e, s in sorted(scores.items(), key=lambda x: -x[1])}

    def get_relevant_themes(self, emotion_scores: dict, top_n: int = 3) -> list:
        """
        Given emotion scores, return the most relevant book themes to target.
        """
        theme_weights = defaultdict(float)
        for emotion, score in emotion_scores.items():
            if emotion in self.theme_map:
                for theme in self.theme_map[emotion]:
                    theme_weights[theme] += score

        sorted_themes = sorted(theme_weights.items(), key=lambda x: -x[1])
        return [t[0] for t in sorted_themes[:top_n * 3]]   # return more for broader matching

    def dominant_emotion(self, emotion_scores: dict) -> str:
        if not emotion_scores:
            return "calm"
        return max(emotion_scores, key=emotion_scores.get)


# ── Optional: HuggingFace upgrade path ──────────────────────────────────────
# If you want to replace the lexicon approach with a neural model locally:
#
# from transformers import pipeline
# classifier = pipeline("text-classification",
#                        model="j-hartmann/emotion-english-distilroberta-base",
#                        return_all_scores=True)
#
# def detect_neural(text):
#     results = classifier(text)[0]
#     return {r['label'].lower(): round(r['score'], 4) for r in results}
