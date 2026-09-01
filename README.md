# ResumeIQ — ML-Powered Resume Screener
**B.Tech. AIML Project | AMGOI, Vathar | AY 2025-26**
**Student:** Diksha Rahul Shinge | **Guide:** Prof. S.R. Pol

---

## 🚀 How to Run (Step-by-Step)

### Step 1 — Install Python
Make sure you have **Python 3.8 or higher** installed.
Download from: https://www.python.org/downloads/

---

### Step 2 — Install Dependencies
Open your terminal / command prompt, navigate to this folder, and run:

```bash
pip install -r requirements.txt
```

This installs:
- `streamlit` — Web UI framework
- `scikit-learn` — TF-IDF + Cosine Similarity
- `nltk` — Tokenization, Lemmatization, Stop-word removal
- `pdfminer.six` — PDF text extraction
- `python-docx` — DOCX text extraction
- `pandas`, `numpy` — Data handling

---

### Step 3 — Run the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## 🧠 How the ML Pipeline Works

```
Resume (PDF/DOCX) ──┐
                    ├──► Text Extraction ──► Preprocessing ──► TF-IDF Vectorization
Job Description ────┘        (PyPDFMiner)    (NLTK: tokenize,      (Scikit-learn)
                                              stop-words,                │
                                              lemmatize)                 ▼
                                                               Cosine Similarity Score
                                                                         │
                                                                         ▼
                                                               Ranked Candidate List
```

### Preprocessing Steps
1. **Lowercasing** — normalize all text
2. **Regex Cleaning** — remove special characters
3. **Tokenization** — split into words
4. **Stop-word Removal** — remove "is", "the", "at", etc.
5. **Lemmatization** — convert "programming" → "program"

### Scoring Formula
- TF-IDF converts JD + resumes into vectors
- Cosine Similarity = (JD · Resume) / (|JD| × |Resume|)
- Score range: 0.0 (no match) → 1.0 (perfect match)

---

## 📁 Project Structure

```
resume_screener/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 🖥️ Features

| Feature | Description |
|---|---|
| Multi-format parsing | PDF, DOCX, TXT resumes |
| NLP Preprocessing | Tokenize, Lemmatize, Stop-word removal |
| TF-IDF Vectorization | Bigram (1,2) n-grams, 8000 features |
| Cosine Similarity | Mathematical match scoring |
| Skill Extraction | 60+ technical skills detected |
| Contact Extraction | Email and phone via regex |
| Ranked Dashboard | Visual score bars, rank badges |
| CSV Export | Download results |
| Threshold Filter | Set minimum match % |

---

## 📚 References (from Synopsis)
1. Regilan et al., IEEE ICCC 2025
2. Antony et al., ICCIDS 2026
3. Liu, ISMLSC 2025
4. IEEE Xplore, Dec 2024
5. Deshmukh & Raut, Annals of Data Science, 2025

---

*Department of Artificial Intelligence and Machine Learning Engineering*
*Ashokrao Mane Group of Institutions, Vathar Tarf Vadgaon*
*Dr. Babasaheb Ambedkar Technological University, Lonere*
