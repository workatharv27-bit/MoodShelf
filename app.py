"""
app.py
------
MoodShelf Streamlit UI — run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from recommender import MoodShelf

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MoodShelf",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Playfair Display', serif; }

    .main { background: #0f0e17; }
    .block-container { padding-top: 2rem; }

    .mood-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #e8c99a33;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .book-card {
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        border-left: 4px solid #e8c99a;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .score-pill {
        display: inline-block;
        background: #e8c99a22;
        border: 1px solid #e8c99a55;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        color: #e8c99a;
        margin-right: 6px;
    }
    .emotion-tag {
        display: inline-block;
        background: #7b2d8b33;
        border: 1px solid #7b2d8b88;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.82rem;
        color: #d4a8e0;
        margin: 3px;
    }
    .theme-tag {
        display: inline-block;
        background: #1e6b8833;
        border: 1px solid #1e6b8888;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.82rem;
        color: #7ecbdb;
        margin: 3px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load model (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    ms = MoodShelf()
    ms.load_data()
    return ms


ms = load_model()
all_books = ms.books_df


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌙 MoodShelf")
    st.caption("Emotion-aware book recommendations")
    st.divider()

    user_id = st.selectbox(
        "👤 Select user profile",
        options=["new_user"] + [f"u{i}" for i in range(1, 16)],
        help="Existing users get collaborative filtering from similar readers."
    )

    st.markdown("**📚 Mark books you've read**")
    all_titles = all_books[["book_id", "title"]].values.tolist()
    read_books = st.multiselect(
        "Already read",
        options=[b[0] for b in all_titles],
        format_func=lambda bid: next(t for i, t in all_titles if i == bid),
        default=[],
    )

    top_n = st.slider("How many recommendations?", 3, 10, 5)

    st.divider()

    with st.expander("⚙️ Tune weights"):
        w_emotion = st.slider("Emotion weight", 0.0, 1.0, 0.40, 0.05)
        w_content = st.slider("Content weight", 0.0, 1.0, 0.35, 0.05)
        w_collab = st.slider("Collaborative weight", 0.0, 1.0, 0.25, 0.05)

    ms.hybrid.w_emotion = w_emotion / (w_emotion + w_content + w_collab)
    ms.hybrid.w_content = w_content / (w_emotion + w_content + w_collab)
    ms.hybrid.w_collab  = w_collab  / (w_emotion + w_content + w_collab)


# ── Main content ─────────────────────────────────────────────────────────────
st.markdown("# 🌙 MoodShelf")
st.markdown("*Tell me how you're feeling. I'll find your next book.*")
st.divider()

mood_text = st.text_area(
    "📝 How are you feeling right now? Write freely.",
    height=120,
    placeholder="e.g. I've been feeling really anxious lately, overwhelmed with work. "
                "I need something calming but also want to feel hopeful again...",
)

recommend_btn = st.button("✨ Find My Books", type="primary", use_container_width=True)

if recommend_btn and mood_text.strip():
    with st.spinner("Reading your mood and searching the shelves..."):
        uid = None if user_id == "new_user" else user_id
        results = ms.recommend(
            mood_text=mood_text,
            user_id=uid,
            reading_history=read_books,
            top_n=top_n,
        )

    # Emotions detected
    st.markdown("### 🧠 Mood Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Detected emotions:**")
        emotion_html = " ".join(
            f'<span class="emotion-tag">{e} {s:.0%}</span>'
            for e, s in list(results["emotions"].items())[:5]
        )
        st.markdown(emotion_html, unsafe_allow_html=True)

    with col2:
        st.markdown("**Target themes:**")
        theme_html = " ".join(
            f'<span class="theme-tag">{t}</span>'
            for t in results["themes"]
        )
        st.markdown(theme_html, unsafe_allow_html=True)

    st.divider()

    # Recommendations
    st.markdown("### 📚 Your Recommendations")
    for i, (_, row) in enumerate(results["books"].iterrows()):
        exp = results["explanations"][i]
        st.markdown(f"""
        <div class="book-card">
            <h4 style="color:#e8c99a; margin:0 0 4px 0;">{i+1}. {row['title']}</h4>
            <p style="color:#aaa; margin:0 0 8px 0; font-size:0.9rem;">
                by {row['author']} &nbsp;|&nbsp; {row['genre']} &nbsp;|&nbsp; ★ {row['avg_rating']}
            </p>
            <p style="color:#ccc; font-size:0.88rem; margin-bottom:10px;">{row['description']}</p>
            <p style="color:#b8d4c8; font-size:0.85rem; font-style:italic;">{exp}</p>
            <div style="margin-top:8px;">
                <span class="score-pill">mood {row['score_emotion']:.2f}</span>
                <span class="score-pill">content {row['score_content']:.2f}</span>
                <span class="score-pill">social {row['score_collab']:.2f}</span>
                <span class="score-pill">★ final {row['score_final']:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif recommend_btn:
    st.warning("Please describe your mood above to get recommendations.")
