# ============================================================
#  JanAI — NLP Issue Classifier
#  Uses zero-shot classification (no training data needed)
# ============================================================

from transformers import pipeline
import re

# ── Load Model ───────────────────────────────────────────────
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# ── Governance Categories ────────────────────────────────────
CATEGORIES = [
    "roads and infrastructure",
    "water supply",
    "sanitation and garbage",
    "electricity",
    "healthcare",
    "education",
    "law and order",
    "public transport"
]

HIGH_URGENCY_KEYWORDS = [
    "urgent", "emergency", "accident", "fire",
    "flood", "death", "hospital", "bleeding",
    "collapsed", "critical", "danger", "immediate"
]

# ── Helper ───────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase, strip punctuation and extra spaces."""
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# ── Main Function ────────────────────────────────────────────
def classify_issue(complaint: str) -> dict:
    """
    Classify a citizen complaint into a governance category.

    Returns
    -------
    dict with keys:
        original_text, cleaned_text, category, confidence, severity
    """
    cleaned = clean_text(complaint)
    result  = classifier(cleaned, CATEGORIES)

    top_category = result["labels"][0]
    confidence   = round(result["scores"][0], 3)
    severity     = "HIGH" if any(
        word in cleaned for word in HIGH_URGENCY_KEYWORDS
    ) else "MEDIUM"

    return {
        "original_text": complaint,
        "cleaned_text":  cleaned,
        "category":      top_category,
        "confidence":    confidence,
        "severity":      severity
    }


# ── Quick Test ───────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "There is a huge pothole on MG Road causing accidents",
        "No water supply in our area for 3 days",
        "Garbage not collected since last week, smells terrible",
        "Street lights not working near the school"
    ]
    for s in samples:
        r = classify_issue(s)
        print(f"\nComplaint : {r['original_text']}")
        print(f"Category  : {r['category']}")
        print(f"Confidence: {r['confidence']}")
        print(f"Severity  : {r['severity']}")
