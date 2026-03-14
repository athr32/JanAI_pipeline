# 🧠 JanAI — AI NLP Pipeline

> Civic Intelligence Platform for Local Government Leaders
> Converts citizen complaints (text or voice) into structured AI insights.

## What This Repo Does

This is the AI/NLP backend of JanAI. It takes a complaint (text or Hindi voice)
and returns: category, sentiment, priority score, and assigned department.

Supported Languages: English, Hindi
STT Engine: Groq Whisper API (free, fast)
NLP Models: facebook/bart-large-mnli + cardiffnlp/twitter-roberta

---

## Step 1 — Prerequisites

- Python 3.10 or 3.11  (NOT 3.12 — transformers has issues with 3.12)
- pip
- A free Groq API key  →  https://console.groq.com

---

## Step 2 — Clone the Repo

git clone https://github.com/YOUR_USERNAME/janai_nlp_pipeline.git
cd janai_nlp_pipeline

---

## Step 3 — Create Virtual Environment

# Windows
python -m venv janai_env
janai_env\Scripts\activate

# Mac / Linux
python -m venv janai_env
source janai_env/bin/activate

---

## Step 4 — Install Dependencies

pip install -r requirements.txt

This installs: streamlit, groq, transformers, torch, chromadb,
sentence-transformers, python-dotenv, streamlit-mic-recorder

---

## Step 5 — Create Your .env File

Create a file named exactly  .env  in the project root folder:

GROQ_API_KEY=gsk_your_key_here

Get your free key at: https://console.groq.com → API Keys → Create
Free tier: 7200 audio minutes/day + unlimited LLaMA calls

IMPORTANT: Never share this file. Never commit it to GitHub.

---

## Step 6 — Run the Streamlit Demo

streamlit run app.py

Opens at: http://localhost:8501
Three tabs: Text | Live Voice | Upload Audio

---

## For Backend Developers — FastAPI Integration

Import and call process_complaint() from pipeline/ai_pipeline.py:

from pipeline.ai_pipeline import process_complaint

result = process_complaint(
    input_type          = 'text',          # 'text' or 'voice'
    content             = 'Road is broken near station',
    location            = 'MG Road, Nashik',
    population_affected = 500
)

# result is a dict with these keys:
print(result['complaint_id'])       # unique ID
print(result['category'])           # 'roads and infrastructure'
print(result['priority_label'])     # 'HIGH'
print(result['priority_score'])     # 38.5
print(result['sentiment'])          # 'NEGATIVE'
print(result['assigned_department'])# 'PWD - Public Works Department'
print(result['status'])             # 'pending'

---

## Priority Score Logic

Score = (Urgency x Severity) + Recurrence + Population Impact + Sentiment

CRITICAL  >= 50   →  Immediate escalation
HIGH      >= 30   →  Resolve within 48 hours
MEDIUM    >= 15   →  Resolve within 1 week
LOW       < 15    →  Scheduled maintenance

---

## Department Mapping

roads and infrastructure  →  PWD — Public Works Department
water supply              →  Water Supply Board
sanitation and garbage    →  Municipal Sanitation Department
electricity               →  State Electricity Board
healthcare                →  District Health Office
education                 →  District Education Office
law and order             →  Police Department
public transport          →  Transport Authority

---

## Project Structure

janai_nlp_pipeline/
├── nlp/
│   ├── classifier.py       # BART zero-shot classification
│   ├── sentiment.py        # RoBERTa sentiment analysis
│   └── __init__.py
├── pipeline/
│   ├── ai_pipeline.py      # Main entry point for FastAPI
│   └── __init__.py
├── priority_engine/
│   ├── scorer.py           # Priority scoring formula
│   └── __init__.py
├── vector_db/
│   ├── store.py            # ChromaDB storage + recurrence
│   └── __init__.py
├── app.py                  # Streamlit demo UI
├── requirements.txt        # pip dependencies
├── README.md
└── .gitignore

---

## Common Errors

Translation not working
  → Check sidebar shows green Groq key status
  → Verify .env has no spaces: GROQ_API_KEY=gsk_xxx

Module not found errors
  → Make sure you are inside the janai_nlp_pipeline/ folder
  → Run: cd janai_nlp_pipeline then streamlit run app.py

Hindi classified wrongly
  → Check the green translation box in results
  → If translation is wrong, edit the text manually and re-analyze

Sidebar shows red GROQ_API_KEY missing
  → Create .env file in the same folder as app.py
  → File must be named exactly .env (not env.txt or .env.txt)

---

## Built By
JanAI Team — Hackathon 2026
NLP Pipeline: [Your Name]
Backend/FastAPI: [Teammate Name]
