"""
app.py — MoodShelf Streamlit UI.  Run: streamlit run app.py
Requires trained models (run train.py first).
"""
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from recommender import MoodShelf

st.set_page_config(page_title="MoodShelf", page_icon="🌙", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
h1,h2,h3{font-family:'Playfair Display',serif;}
.book-card{background:linear-gradient(135deg,#1a1a2e,#0f3460);border-left:4px solid #e8c99a;
           border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem;}
.pill{display:inline-block;background:#e8c99a22;border:1px solid #e8c99a55;border-radius:20px;
      padding:2px 10px;font-size:.78rem;color:#e8c99a;margin-right:6px;}
.etag{display:inline-block;background:#7b2d8b33;border:1px solid #7b2d8b88;border-radius:20px;
      padding:3px 12px;font-size:.82rem;color:#d4a8e0;margin:3px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load():
    ms = MoodShelf()
    ms.load()
    return ms

ms = load()

with st.sidebar:
    st.markdown("## 🌙 MoodShelf")
    user_id = st.selectbox("User profile", ["new_user"] + [f"u{i}" for i in range(1,16)])
    books   = ms.books_df[["book_id","title"]].values.tolist()
    history = st.multiselect("Already read", [b[0] for b in books],
                              format_func=lambda bid: next(t for i,t in books if i==bid))
    top_n   = st.slider("Recommendations", 3, 10, 5)

st.markdown("# 🌙 MoodShelf")
st.markdown("*Tell me how you're feeling. I'll find your next book.*")

mood = st.text_area("How are you feeling right now?", height=120,
                     placeholder="e.g. I've been anxious and overwhelmed lately...")

if st.button("✨ Find My Books", type="primary", use_container_width=True) and mood.strip():
    with st.spinner("Analyzing mood and searching shelves..."):
        uid = None if user_id == "new_user" else user_id
        r = ms.recommend(mood, user_id=uid, reading_history=history, top_n=top_n)

    st.markdown("### 🧠 Detected Mood")
    tags = " ".join(f'<span class="etag">{e} {s:.0%}</span>'
                    for e, s in list(r["emotions"].items())[:5])
    st.markdown(tags, unsafe_allow_html=True)
    st.divider()

    st.markdown("### 📚 Recommendations")
    for i, (_, row) in enumerate(r["books"].iterrows()):
        st.markdown(f"""
        <div class="book-card">
            <h4 style="color:#e8c99a;margin:0 0 4px">{i+1}. {row['title']}</h4>
            <p style="color:#aaa;font-size:.9rem;margin:0 0 8px">
                {row['author']} &nbsp;|&nbsp; {row['genre']} &nbsp;|&nbsp; ★{row['avg_rating']}
            </p>
            <p style="color:#ccc;font-size:.88rem;margin-bottom:8px">{row['description']}</p>
            <p style="color:#b8d4c8;font-size:.85rem;font-style:italic">{r['explanations'][i]}</p>
            <div>
                <span class="pill">mood {row['score_emotion']:.2f}</span>
                <span class="pill">content {row['score_content']:.2f}</span>
                <span class="pill">social {row['score_collab']:.2f}</span>
                <span class="pill">final {row['score_final']:.2f}</span>
            </div>
        </div>""", unsafe_allow_html=True)
