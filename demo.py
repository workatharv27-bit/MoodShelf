"""
demo.py
-------
Run this to test MoodShelf from the command line without Streamlit.

    python demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from recommender import MoodShelf


def run_demo():
    print("\n🌙 MoodShelf — Emotion-Aware Book Recommender\n")

    ms = MoodShelf(
        emotion_weight=0.40,
        content_weight=0.35,
        collab_weight=0.25,
    )
    ms.load_data()

    # ── Test cases ────────────────────────────────────────────────────────
    test_cases = [
        {
            "label": "Anxious & overwhelmed user",
            "mood": "I've been incredibly anxious lately, stressed from work and feeling overwhelmed. "
                    "I just need something calming and peaceful to read.",
            "user_id": "u3",
            "history": [4, 6],
        },
        {
            "label": "Curious & energetic user",
            "mood": "I'm really excited and curious about the world right now! "
                    "I want to learn something new, understand history or science.",
            "user_id": "u9",
            "history": [7, 14],
        },
        {
            "label": "Sad & lonely new user",
            "mood": "I've been feeling really lonely and sad. Missing connection with people. "
                    "I want something that makes me feel less alone.",
            "user_id": None,
            "history": [],
        },
        {
            "label": "Happy & adventurous",
            "mood": "I'm feeling great and happy today! Looking for something fun, "
                    "exciting with adventure and maybe some humor.",
            "user_id": "u2",
            "history": [3, 9],
        },
    ]

    for tc in test_cases:
        print(f"\n{'─'*60}")
        print(f"  TEST: {tc['label']}")
        print(f"  Mood: \"{tc['mood'][:80]}...\"")
        print(f"{'─'*60}")

        results = ms.recommend(
            mood_text=tc["mood"],
            user_id=tc["user_id"],
            reading_history=tc["history"],
            top_n=3,
        )
        ms.pretty_print(results)

    # ── Interactive mode ───────────────────────────────────────────────────
    print("\n\n" + "═" * 60)
    print("  🎯  Interactive Mode — Enter your own mood!")
    print("═" * 60)
    print("  (type 'quit' to exit)\n")

    while True:
        mood = input("  📝 How are you feeling? → ").strip()
        if mood.lower() in ("quit", "exit", "q"):
            break
        if not mood:
            continue

        results = ms.recommend(mood_text=mood, top_n=5)
        ms.pretty_print(results)


if __name__ == "__main__":
    run_demo()
