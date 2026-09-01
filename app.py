import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import os

# ── PDF / DOCX parsing ────────────────────────────────────────────────────────
try:
    import pdfminer
    from pdfminer.high_level import extract_text as pdf_extract
    PDF_OK = True
except ImportError:
    PDF_OK = False

try:
    import docx
    DOCX_OK = True
except ImportError:
    DOCX_OK = False

# ── NLP ───────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download required NLTK data silently
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeIQ Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  – dark, editorial look
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:       #0d0f14;
    --surface:  #13161e;
    --border:   #1f2333;
    --accent:   #7c6fff;
    --accent2:  #ff6b6b;
    --gold:     #f5c842;
    --text:     #e8eaf0;
    --muted:    #6b7080;
    --green:    #4ade80;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

h1,h2,h3 { font-family: 'DM Serif Display', serif; }

.hero {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
}
.hero h1 {
    font-size: 3.2rem;
    background: linear-gradient(135deg, #7c6fff 0%, #ff6b6b 60%, #f5c842 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: .2rem;
}
.hero p { color: var(--muted); font-size: 1.05rem; margin-top: 0; }

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.rank-badge {
    display: inline-block;
    width: 32px; height: 32px;
    border-radius: 50%;
    font-family: 'DM Mono', monospace;
    font-size: .85rem;
    font-weight: 600;
    line-height: 32px;
    text-align: center;
    margin-right: .6rem;
}
.rank-1 { background: #f5c842; color: #000; }
.rank-2 { background: #c0c0c0; color: #000; }
.rank-3 { background: #cd7f32; color: #fff; }
.rank-n { background: var(--border); color: var(--muted); }

.score-bar-wrap {
    background: var(--border);
    border-radius: 99px;
    height: 8px;
    margin: .5rem 0 .3rem;
    overflow: hidden;
}
.score-bar {
    height: 100%;
    border-radius: 99px;
    transition: width .4s ease;
}
.pill {
    display: inline-block;
    padding: .18rem .7rem;
    border-radius: 99px;
    font-size: .75rem;
    font-weight: 500;
    margin: .15rem .1rem;
    background: #1a1d28;
    border: 1px solid var(--border);
    color: var(--accent);
}
.metric-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: var(--accent);
}
.metric-label { color: var(--muted); font-size: .8rem; margin-top: .1rem; }

/* Streamlit widget overrides */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 12px !important;
}
.stTextArea textarea {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: .85rem !important;
}
.stSlider [data-testid="stMarkdownContainer"] { color: var(--muted) !important; }
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7c6fff, #ff6b6b) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: .6rem 2rem !important;
    font-size: 1rem !important;
    width: 100%;
}
div[data-testid="stButton"] > button:hover { opacity: .88 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

SKILL_KEYWORDS = [
    "python","java","javascript","typescript","c++","c#","sql","r","scala","go",
    "react","angular","vue","node","django","flask","fastapi","spring","tensorflow",
    "pytorch","keras","sklearn","scikit","pandas","numpy","matplotlib","seaborn",
    "docker","kubernetes","aws","azure","gcp","git","linux","bash","hadoop","spark",
    "kafka","mongodb","mysql","postgresql","redis","elasticsearch","tableau","powerbi",
    "excel","nlp","machine learning","deep learning","computer vision","bert","gpt",
    "transformer","opencv","nltk","spacy","figma","html","css","rest","graphql",
    "agile","scrum","devops","ci/cd","mlflow","airflow","streamlit","flask",
    "data analysis","data science","data engineering","feature engineering",
]

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if not PDF_OK:
        return ""
    try:
        return pdf_extract(io.BytesIO(file_bytes))
    except Exception:
        return ""

def extract_text_from_docx(file_bytes: bytes) -> str:
    if not DOCX_OK:
        return ""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""

def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(data)
    elif name.endswith(".docx"):
        return extract_text_from_docx(data)
    elif name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    return ""

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s+#]", " ", text)
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)

def extract_skills(text: str) -> list:
    text_lower = text.lower()
    found = []
    for skill in SKILL_KEYWORDS:
        if skill in text_lower:
            found.append(skill.title())
    return list(dict.fromkeys(found))   # deduplicate preserving order

def extract_email(text: str) -> str:
    m = re.search(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", text, re.I)
    return m.group(0) if m else "—"

def extract_phone(text: str) -> str:
    m = re.search(r"(\+?\d[\d\s\-().]{8,}\d)", text)
    return m.group(0).strip() if m else "—"

def score_color(score: float) -> str:
    if score >= 0.70:
        return "#4ade80"
    elif score >= 0.45:
        return "#f5c842"
    else:
        return "#ff6b6b"

def rank_badge(i: int) -> str:
    cls = {0:"rank-1", 1:"rank-2", 2:"rank-3"}.get(i, "rank-n")
    label = {0:"🥇", 1:"🥈", 2:"🥉"}.get(i, str(i+1))
    return f'<span class="rank-badge {cls}">{label}</span>'

def tfidf_rank(jd_text: str, resumes: list[dict], top_n: int) -> pd.DataFrame:
    """
    resumes: list of {"name": ..., "raw": ..., "clean": ...}
    Returns ranked DataFrame.
    """
    docs = [preprocess(jd_text)] + [r["clean"] for r in resumes]
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=8000)
    tfidf_matrix = vec.fit_transform(docs)
    jd_vec = tfidf_matrix[0]
    resume_vecs = tfidf_matrix[1:]
    scores = cosine_similarity(jd_vec, resume_vecs)[0]

    rows = []
    for i, r in enumerate(resumes):
        rows.append({
            "Rank": i + 1,
            "Name": r["name"],
            "Match Score": round(float(scores[i]), 4),
            "Match %": round(float(scores[i]) * 100, 1),
            "Skills Found": extract_skills(r["raw"]),
            "Email": extract_email(r["raw"]),
            "Phone": extract_phone(r["raw"]),
            "raw_text": r["raw"],
        })

    df = pd.DataFrame(rows).sort_values("Match Score", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)
    return df.head(top_n)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    top_n = st.slider("Top candidates to show", 1, 20, 5)
    min_score = st.slider("Minimum match % threshold", 0, 100, 20)
    st.markdown("---")
    st.markdown("### 📋 How it works")
    st.markdown("""
1. Paste the **Job Description**
2. Upload **multiple resumes** (PDF / DOCX / TXT)
3. Click **Screen Resumes**
4. View ranked shortlist
""")
    st.markdown("---")
    st.markdown("""
<div style='color:#6b7080;font-size:.75rem'>
<b>Tech Stack</b><br>
TF-IDF · Cosine Similarity<br>
NLTK · Scikit-learn · Streamlit<br><br>
<i>AMGOI — AIML Dept · 2025-26</i>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>ResumeIQ</h1>
    <p>ML-Powered Resume Screener · TF-IDF + Cosine Similarity · Built for recruiters</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

col_jd, col_res = st.columns([1, 1], gap="large")

with col_jd:
    st.markdown("### 📄 Job Description")
    jd_input = st.text_area(
        "Paste the full JD here",
        height=320,
        placeholder="e.g.  We are looking for a Data Scientist with 3+ years of experience in Python, SQL, Machine Learning, and NLP...",
        label_visibility="collapsed",
    )

with col_res:
    st.markdown("### 📂 Upload Resumes")
    resume_files = st.file_uploader(
        "Drop PDF / DOCX / TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if resume_files:
        st.success(f"✅ {len(resume_files)} file(s) loaded")
        for f in resume_files:
            st.markdown(f"&nbsp;&nbsp;📎 `{f.name}`")

st.markdown("---")

run = st.button("🎯 Screen Resumes")

# ─────────────────────────────────────────────────────────────────────────────
# RUN SCREENING
# ─────────────────────────────────────────────────────────────────────────────
if run:
    if not jd_input.strip():
        st.error("⚠️ Please paste a Job Description first.")
        st.stop()
    if not resume_files:
        st.error("⚠️ Please upload at least one resume.")
        st.stop()

    with st.spinner("🔍 Parsing resumes and computing similarity scores…"):
        resumes = []
        failed = []
        for f in resume_files:
            raw = extract_text(f)
            if raw.strip():
                resumes.append({
                    "name": f.name,
                    "raw": raw,
                    "clean": preprocess(raw),
                })
            else:
                failed.append(f.name)

        if failed:
            st.warning(f"Could not extract text from: {', '.join(failed)}")

        if not resumes:
            st.error("No readable text found in any uploaded file.")
            st.stop()

        results = tfidf_rank(jd_input, resumes, top_n=top_n)
        results_filtered = results[results["Match %"] >= min_score]

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.markdown("## 📊 Screening Results")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-val">{len(resumes)}</div>
            <div class="metric-label">Resumes Screened</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-val">{len(results_filtered)}</div>
            <div class="metric-label">Qualified (≥{min_score}%)</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        top_score = results["Match %"].iloc[0] if len(results) else 0
        st.markdown(f"""<div class="metric-box">
            <div class="metric-val">{top_score}%</div>
            <div class="metric-label">Top Match Score</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        avg_score = round(results["Match %"].mean(), 1) if len(results) else 0
        st.markdown(f"""<div class="metric-box">
            <div class="metric-val">{avg_score}%</div>
            <div class="metric-label">Average Score</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Ranked cards ──────────────────────────────────────────────────────────
    st.markdown("### 🏆 Ranked Candidates")
    if results_filtered.empty:
        st.info(f"No candidates met the {min_score}% threshold. Try lowering it in the sidebar.")
    else:
        for i, row in results_filtered.iterrows():
            score = row["Match Score"]
            pct = row["Match %"]
            color = score_color(score)
            skills_html = "".join(f'<span class="pill">{s}</span>' for s in row["Skills Found"][:15])
            skills_html = skills_html if skills_html else '<span style="color:#6b7080">No common skills detected</span>'

            st.markdown(f"""
<div class="card">
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.5rem">
        <div>
            {rank_badge(int(row['Rank'])-1)}
            <span style="font-family:'DM Serif Display',serif;font-size:1.2rem">{row['Name']}</span>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:1.5rem;font-weight:700;color:{color}">
            {pct}%
        </div>
    </div>
    <div class="score-bar-wrap">
        <div class="score-bar" style="width:{pct}%;background:linear-gradient(90deg,{color}88,{color})"></div>
    </div>
    <div style="display:flex;gap:2rem;margin:.5rem 0;color:#6b7080;font-size:.82rem">
        <span>📧 {row['Email']}</span>
        <span>📞 {row['Phone']}</span>
        <span>🎯 Cosine: {score:.4f}</span>
    </div>
    <div style="margin-top:.5rem">{skills_html}</div>
</div>
""", unsafe_allow_html=True)

    # ── Expandable table ──────────────────────────────────────────────────────
    with st.expander("📋 View Full Data Table"):
        display_df = results[["Rank","Name","Match %","Email","Phone"]].copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Download CSV ──────────────────────────────────────────────────────────
    csv = results[["Rank","Name","Match %","Email","Phone"]].to_csv(index=False)
    st.download_button(
        "⬇️ Download Results as CSV",
        data=csv,
        file_name="screening_results.csv",
        mime="text/csv",
    )

    # ── Raw text preview ──────────────────────────────────────────────────────
    with st.expander("🔬 Preview Extracted Resume Text"):
        sel = st.selectbox("Select resume", [r["name"] for r in resumes])
        raw = next(r["raw"] for r in resumes if r["name"] == sel)
        st.text_area("Extracted Text", raw[:3000] + ("…" if len(raw) > 3000 else ""),
                     height=250, label_visibility="collapsed")
