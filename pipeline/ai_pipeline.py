# ============================================================
#  JanAI — Master AI Pipeline
#  Single entry point for all complaint processing
#
#  Flow:
#  Input (text/voice/image)
#    → Pre-processing (STT / clean)
#    → NLP Classification
#    → Sentiment Analysis
#    → Priority Scoring
#    → Vector DB Storage
#    → Structured Record Output
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nlp.classifier        import classify_issue
from nlp.sentiment         import analyze_sentiment
from whisper.transcriber   import transcribe_audio
from vector_db.store       import add_complaint, find_similar_complaints
from priority_engine.scorer import compute_priority_score
from datetime              import datetime


# ── Department Mapping ────────────────────────────────────────
DEPARTMENT_MAP = {
    "roads and infrastructure": "PWD — Public Works Department",
    "water supply":             "Water Supply Board",
    "sanitation and garbage":   "Municipal Sanitation Department",
    "electricity":              "State Electricity Board",
    "healthcare":               "District Health Office",
    "education":                "District Education Office",
    "law and order":            "Police Department",
    "public transport":         "Transport Authority"
}


def map_to_department(category: str) -> str:
    return DEPARTMENT_MAP.get(category, "General Administration")


# ── Master Pipeline ───────────────────────────────────────────
def process_complaint(
    input_type:          str,
    content:             str,
    location:            str = "Unknown",
    population_affected: int = 1
) -> dict:
    """
    Process any type of citizen complaint end-to-end.

    Parameters
    ----------
    input_type          : 'text' | 'voice' | 'image'
    content             : complaint text OR path to audio/image file
    location            : location string (e.g. 'MG Road, Pune')
    population_affected : estimated people impacted (for scoring)

    Returns
    -------
    Fully structured complaint record dict
    """

    print(f"\n{'='*55}")
    print(f"[JanAI Pipeline] Processing '{input_type}' complaint...")
    language = "en"

    # ── Step 1: Input Processing ──────────────────────────────
    print("[1/5] Input processing...")

    if input_type == "voice":
        stt_result = transcribe_audio(content)
        text       = stt_result["transcript"]
        language   = stt_result["language"]
        print(f"      Transcribed: '{text[:60]}...'")

    elif input_type == "text":
        text = content

    elif input_type == "image":
        # Image caption / OCR result passed as content
        # (OpenCV module handled by teammate, result passed here)
        text = content

    else:
        raise ValueError(f"Unknown input_type: '{input_type}'. Use text/voice/image.")

    # ── Step 2: NLP Classification ────────────────────────────
    print("[2/5] Classifying issue...")
    nlp = classify_issue(text)
    print(f"      Category: {nlp['category']}  |  Severity: {nlp['severity']}")

    # ── Step 3: Sentiment Analysis ────────────────────────────
    print("[3/5] Analyzing sentiment...")
    sentiment = analyze_sentiment(text)
    print(f"      Sentiment: {sentiment['sentiment']}  |  Intensity: {sentiment['negative_intensity']}")

    # ── Step 4: Priority Scoring ──────────────────────────────
    print("[4/5] Computing priority score...")
    priority = compute_priority_score(text, population_affected)
    print(f"      Score: {priority['total_score']}  |  Label: {priority['priority_label']}")

    # ── Step 5: Store in Vector DB ────────────────────────────
    print("[5/5] Storing in vector database...")
    complaint_id = add_complaint(
        text     = text,
        category = nlp["category"],
        severity = nlp["severity"],
        location = location
    )

    # ── Build Final Record ────────────────────────────────────
    record = {
        "complaint_id":        complaint_id,
        "timestamp":           datetime.now().isoformat(),
        "input_type":          input_type,
        "original_text":       text,
        "language":            language,
        "location":            location,
        "category":            nlp["category"],
        "severity":            nlp["severity"],
        "sentiment":           sentiment["sentiment"],
        "priority_label":      priority["priority_label"],
        "priority_score":      priority["total_score"],
        "score_breakdown":     priority["score_breakdown"],
        "recurrence_count":    priority["recurrence_count"],
        "population_affected": population_affected,
        "status":              "OPEN",
        "assigned_department": map_to_department(nlp["category"])
    }

    print(f"\n✅ Complaint processed!")
    print(f"   ID         : {complaint_id[:8]}...")
    print(f"   Category   : {record['category']}")
    print(f"   Priority   : {record['priority_label']}  (score: {record['priority_score']})")
    print(f"   Department : {record['assigned_department']}")
    print(f"{'='*55}\n")

    return record


# ── Quick Test ───────────────────────────────────────────────
if __name__ == "__main__":

    # Test 1: Text complaint
    r1 = process_complaint(
        input_type          = "text",
        content             = "Massive pothole on MG Road causing accidents daily",
        location            = "MG Road, Pune",
        population_affected = 500
    )

    # Test 2: Another text complaint (different category)
    r2 = process_complaint(
        input_type          = "text",
        content             = "No water supply in Sector 5 for 3 days straight!",
        location            = "Sector 5, Nashik",
        population_affected = 200
    )

    # Test 3: Voice complaint (pass audio file path)
    # r3 = process_complaint(
    #     input_type = "voice",
    #     content    = "path/to/complaint.ogg",
    #     location   = "Shivaji Nagar"
    # )

    print("\nSample Output Record:")
    for k, v in r1.items():
        print(f"  {k:25}: {v}")
