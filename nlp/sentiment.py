# ============================================================
#  JanAI — Sentiment Analyzer
#  Uses RoBERTa fine-tuned on Twitter/social media text
# ============================================================

from transformers import pipeline

# ── Load Model ───────────────────────────────────────────────
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)


# ── Main Function ────────────────────────────────────────────
def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment of a complaint or social media post.

    Labels  : POSITIVE, NEGATIVE, NEUTRAL
    Returns : label, confidence, and negative_intensity
              (negative_intensity is used in priority scoring)
    """
    result = sentiment_model(text[:512])[0]   # model max = 512 tokens

    label = result["label"].upper()
    score = round(result["score"], 3)

    # Intensity weight — high negative = higher priority
    intensity_map = {
        "NEGATIVE": score,
        "NEUTRAL":  round(score * 0.3, 3),
        "POSITIVE": round(score * 0.1, 3)
    }

    return {
        "text":               text,
        "sentiment":          label,
        "confidence":         score,
        "negative_intensity": intensity_map.get(label, 0.0)
    }


# ── Quick Test ───────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "This is absolutely terrible, nobody cares about us!",
        "The road was fixed yesterday, thank you so much!",
        "Water supply is irregular sometimes in our area."
    ]
    for t in samples:
        r = analyze_sentiment(t)
        print(f"\nText      : {r['text']}")
        print(f"Sentiment : {r['sentiment']}")
        print(f"Confidence: {r['confidence']}")
        print(f"Intensity : {r['negative_intensity']}")
