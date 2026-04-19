# 🌙 MoodShelf — Emotion-Aware Book Recommender

> *Tell me how you're feeling. I'll find your next book.*

MoodShelf is a hybrid ML recommendation system that suggests books based on your **current emotional state**, **reading history**, and **what similar readers enjoy** — all running locally, no internet required.

---

## 🧠 How It Works

```
User types: "I'm feeling anxious and overwhelmed lately..."
        ↓
[Emotion Model]       → {anxiety: 0.58, sadness: 0.24, ...}
        ↓
[Content Filter]      → Books with themes: calm, mindfulness, peace, healing
        ↓
[Collaborative Filter]→ Books rated highly by similar users (SVD)
        ↓
[Hybrid Scorer]       → Weighted blend → Top N recommendations
```

### The 3 ML Signals

| Signal | Model | Weight |
|--------|-------|--------|
| Emotion-to-theme matching | Keyword lexicon → TF-IDF cosine similarity | 40% |
| Content (reading history) | TF-IDF on themes + descriptions | 35% |
| Collaborative filtering | SVD matrix factorization | 25% |

Weights are tunable in the UI or in code.

---

## 📁 Project Structure

```
moodshelf/
├── data/
│   ├── books.csv           # Book catalog (30 books, expandable)
│   └── ratings.csv         # User ratings (15 users)
├── models/
│   ├── emotion_model.py    # Emotion detection from text
│   ├── content_based.py    # TF-IDF content similarity
│   ├── collaborative.py    # SVD collaborative filtering
│   └── hybrid.py           # Score fusion
├── recommender.py          # Main pipeline
├── app.py                  # Streamlit web UI
├── demo.py                 # CLI demo & interactive mode
└── requirements.txt
```

---

## 🚀 Setup & Running

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the CLI demo
```bash
python demo.py
```

### 3. Run the Streamlit web app
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

---

## 🎯 Extending the Project

### Add more books
Edit `data/books.csv` — just add rows with the same columns. The more books, the better the recommendations.

### Add real user data
Replace `data/ratings.csv` with actual user ratings. Any user×book rating matrix works.

### Upgrade to neural emotion detection
In `models/emotion_model.py`, uncomment the HuggingFace section and replace the `detect()` method:

```python
from transformers import pipeline
classifier = pipeline("text-classification",
                       model="j-hartmann/emotion-english-distilroberta-base",
                       return_all_scores=True)
```

### Add real book data
Fetch from Open Library API:
```python
import requests
r = requests.get("https://openlibrary.org/search.json?q=the+alchemist")
```

---

## 🔧 Tuning

| Parameter | Location | Effect |
|-----------|----------|--------|
| `emotion_weight` | `MoodShelf(emotion_weight=...)` | How much mood drives results |
| `content_weight` | `MoodShelf(content_weight=...)` | How much reading history matters |
| `collab_weight` | `MoodShelf(collab_weight=...)` | How much social signal matters |
| `n_components` | `CollaborativeModel(n_components=...)` | SVD latent factors |
| `EMOTION_THEME_MAP` | `emotion_model.py` | Customize emotion→theme mappings |

---

## 📊 What Makes This Different

Most recommenders use only collaborative filtering ("people who read X also read Y"). MoodShelf adds an **emotional context layer** that maps your current state of mind to book themes. This means:

- Two users with identical reading history get **different recommendations** if they're in different moods
- A user who loved action thrillers but is feeling anxious today gets **calming reads** instead
- The emotion layer is interpretable — you can see exactly why each book was recommended
