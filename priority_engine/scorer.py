# ============================================================
#  JanAI — Priority Scoring Engine
#
#  Formula:
#  Score = (Urgency × Severity)
#        + (Recurrence × Frequency Weight)
#        + (Population Impact)
#        + (Negative Sentiment Intensity)
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nlp.classifier  import classify_issue
from nlp.sentiment   import analyze_sentiment
from vector_db.store import get_recurrence_count


# ── Weights & Thresholds ─────────────────────────────────────
SEVERITY_WEIGHTS = {
    "HIGH":   10,
    "MEDIUM":  5,
    "LOW":     2
}

FREQUENCY_WEIGHT  = 2     # multiplier per recurring complaint
POPULATION_SCALE  = 0.01  # score per person affected
SENTIMENT_SCALE   = 10    # amplifier for negative sentiment

PRIORITY_THRESHOLDS = {
    "CRITICAL": 50,
    "HIGH":     30,
    "MEDIUM":   15,
    "LOW":       0
}


# ── Main Scoring Function ─────────────────────────────────────
def compute_priority_score(
    complaint:           str,
    population_affected: int = 1
) -> dict:
    """
    Compute priority score for a citizen complaint.

    Parameters
    ----------
    complaint            : raw complaint text
    population_affected  : estimated number of people impacted

    Returns
    -------
    dict with score, label, breakdown, and all metadata
    """

    # ── Step 1: NLP + Sentiment ───────────────────────────────
    nlp       = classify_issue(complaint)
    sentiment = analyze_sentiment(complaint)
    recurrence = get_recurrence_count(complaint)

    # ── Step 2: Individual Scores ─────────────────────────────
    severity_weight   = SEVERITY_WEIGHTS.get(nlp["severity"], 2)
    urgency_score     = nlp["confidence"] * 10           # 0–10
    recurrence_score  = recurrence * FREQUENCY_WEIGHT
    population_score  = population_affected * POPULATION_SCALE
    sentiment_score   = sentiment["negative_intensity"] * SENTIMENT_SCALE

    # ── Step 3: Weighted Total ────────────────────────────────
    total = round(
        (urgency_score * severity_weight) +
        recurrence_score                  +
        population_score                  +
        sentiment_score,
        2
    )

    # ── Step 4: Priority Label ────────────────────────────────
    if   total >= PRIORITY_THRESHOLDS["CRITICAL"]:
        priority_label = "CRITICAL"
    elif total >= PRIORITY_THRESHOLDS["HIGH"]:
        priority_label = "HIGH"
    elif total >= PRIORITY_THRESHOLDS["MEDIUM"]:
        priority_label = "MEDIUM"
    else:
        priority_label = "LOW"

    return {
        "complaint":           complaint,
        "category":            nlp["category"],
        "severity":            nlp["severity"],
        "sentiment":           sentiment["sentiment"],
        "recurrence_count":    recurrence,
        "population_affected": population_affected,
        "score_breakdown": {
            "urgency_x_severity": round(urgency_score * severity_weight, 2),
            "recurrence":         recurrence_score,
            "population":         round(population_score, 2),
            "sentiment":          round(sentiment_score, 2)
        },
        "total_score":    total,
        "priority_label": priority_label
    }


# ── Quick Test ───────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        ("Massive pothole on MG Road causing accidents daily",         500),
        ("Street light not working near park",                         50),
        ("Water supply completely cut off — emergency situation!",     1000),
        ("Minor crack on footpath near market",                        20),
    ]

    for complaint, population in test_cases:
        r = compute_priority_score(complaint, population)
        print(f"\n{'─'*55}")
        print(f"Complaint : {r['complaint']}")
        print(f"Category  : {r['category']}")
        print(f"Priority  : {r['priority_label']}  (score: {r['total_score']})")
        print(f"Breakdown : {r['score_breakdown']}")
