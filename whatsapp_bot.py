# ================================================================
#
#   JanAI — WhatsApp Complaint Bot
#   ================================
#   Citizens send complaints via WhatsApp.
#   Bot analyzes them using the full AI pipeline and replies
#   with category, priority, complaint ID and department.
#
#   HOW TO RUN:
#   -----------
#   1. pip install flask twilio
#   2. ngrok http 5000             (in a separate terminal)
#   3. Copy the ngrok URL → paste in Twilio sandbox webhook
#   4. python whatsapp_bot.py
#
#   WEBHOOK URL TO SET IN TWILIO:
#   https://xxxx-xx-xx.ngrok.io/webhook
#
# ================================================================

from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import os
import sys

# ── Load .env (GROQ_API_KEY, TWILIO keys) ────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

# ── Add project root to path so nlp/, vector_db/ etc. import ─────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Twilio credentials from .env ──────────────────────────────────
TWILIO_ACCOUNT_SID     = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN      = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_NUMBER = os.environ.get("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")

app = Flask(__name__)


# ================================================================
#  TRANSLATION FUNCTION
#  Same logic as app.py — translate Hindi to English before NLP
# ================================================================

def translate_to_english(text: str) -> str:
    """
    Translate Hindi/Marathi complaint to English using Groq LLaMA.
    Returns original text unchanged if translation fails.
    """
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii < len(text) * 0.2:
        return text   # already English

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return text

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model    = "llama-3.1-8b-instant",
            messages = [{
                "role": "user",
                "content": (
                    f"Translate this Hindi/Marathi text to English. "
                    f"Reply with ONLY the English translation, nothing else.\n\n"
                    f"Text: {text}"
                )
            }],
            temperature = 0.0,
            max_tokens  = 300,
        )
        translated = response.choices[0].message.content.strip()
        non_ascii_result = sum(1 for c in translated if ord(c) > 127)
        if non_ascii_result > len(translated) * 0.3:
            return text
        return translated if translated else text
    except Exception:
        return text


# ================================================================
#  CORE PIPELINE FUNCTION
#  Runs the full AI pipeline on the complaint text
# ================================================================

def analyze_complaint(complaint_text: str, location: str = "Unknown", population: int = 100) -> dict:
    """
    Run the full JanAI NLP pipeline on a complaint.

    Steps:
      1. Detect language — translate Hindi to English if needed
      2. NLP classification (BART) — category + severity
      3. Sentiment analysis (RoBERTa) — negative intensity
      4. Priority scoring — final score + label
      5. Save to Vector DB — get complaint ID + recurrence count

    Returns a dict with all results, or error info.
    """
    try:
        from nlp.classifier         import classify_issue
        from nlp.sentiment          import analyze_sentiment
        from vector_db.store        import add_complaint
        from priority_engine.scorer import compute_priority_score

        # Step 1 — Translate if Hindi/Marathi
        english_text     = translate_to_english(complaint_text)
        was_translated   = english_text != complaint_text

        # Step 2 — NLP Classification
        nlp_result       = classify_issue(english_text)

        # Step 3 — Sentiment Analysis
        sentiment_result = analyze_sentiment(english_text)

        # Step 4 — Priority Scoring
        priority_result  = compute_priority_score(english_text, population)

        # Step 5 — Save to Vector DB (stores original language text)
        complaint_id = add_complaint(
            complaint_text,              # original Hindi preserved in DB
            nlp_result["category"],
            nlp_result["severity"],
            location
        )

        return {
            "success":        True,
            "complaint_id":   complaint_id,
            "english_text":   english_text,
            "was_translated": was_translated,
            "category":       nlp_result["category"],
            "severity":       nlp_result["severity"],
            "confidence":     nlp_result["confidence"],
            "sentiment":      sentiment_result["sentiment"],
            "intensity":      sentiment_result["negative_intensity"],
            "priority_score": priority_result["total_score"],
            "priority_label": priority_result["priority_label"],
            "recurrence":     priority_result["recurrence_count"],
        }

    except Exception as e:
        return {
            "success": False,
            "error":   str(e)
        }


# ================================================================
#  MESSAGE FORMATTER
#  Builds the WhatsApp reply message from pipeline results
# ================================================================

DEPT_MAP = {
    "roads and infrastructure": "PWD — Public Works Dept",
    "water supply":             "Water Supply Board",
    "sanitation and garbage":   "Municipal Sanitation Dept",
    "electricity":              "State Electricity Board",
    "healthcare":               "District Health Office",
    "education":                "District Education Office",
    "law and order":            "Police Department",
    "public transport":         "Transport Authority",
}

PRIORITY_EMOJI = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
SEVERITY_EMOJI = {"HIGH": "⚠️", "MEDIUM": "📋", "LOW": "ℹ️"}


def build_reply(result: dict, original_text: str) -> str:
    """Build a clean WhatsApp reply from the pipeline result."""

    if not result["success"]:
        return (
            "❌ *JanAI Error*\n\n"
            f"Could not process your complaint:\n_{result['error']}_\n\n"
            "Please try again or type *help*."
        )

    label    = result["priority_label"]
    score    = result["priority_score"]
    category = result["category"].title()
    dept     = DEPT_MAP.get(result["category"], "General Administration")
    # Show only first 20 chars of complaint ID — enough to be unique
    short_id = result["complaint_id"][:20]

    p_emoji = PRIORITY_EMOJI.get(label, "🔵")
    s_emoji = SEVERITY_EMOJI.get(result["severity"], "📋")

    # Show translation if complaint was in Hindi
    translation_line = ""
    if result["was_translated"]:
        translation_line = f"\n🌐 _Translated: {result['english_text']}_\n"

    reply = (
        f"✅ *JanAI — Complaint Registered*\n"
        f"{'─' * 28}\n"
        f"\n"
        f"📝 *Your Complaint:*\n"
        f"_{original_text}_\n"
        f"{translation_line}\n"
        f"{'─' * 28}\n"
        f"\n"
        f"📊 *AI Analysis*\n"
        f"\n"
        f"🏷️ *Category:*  {category}\n"
        f"   Confidence: {int(result['confidence'] * 100)}%\n"
        f"\n"
        f"{p_emoji} *Priority:*  {label}  (Score: {score})\n"
        f"{s_emoji} *Severity:*  {result['severity']}\n"
        f"😠 *Sentiment:*  {result['sentiment']}  (Intensity: {result['intensity']:.2f})\n"
        f"🔁 *Recurrence:*  Reported {result['recurrence']}x before\n"
        f"\n"
        f"{'─' * 28}\n"
        f"\n"
        f"🏛️ *Forwarded To:*\n"
        f"   {dept}\n"
        f"\n"
        f"🆔 *Complaint ID:*\n"
        f"   `{short_id}`\n"
        f"   ✅ _Saved to database_\n"
        f"\n"
        f"{'─' * 28}\n"
        f"_JanAI Civic Intelligence · 2026_"
    )
    return reply


def build_help_message() -> str:
    """Reply when user sends 'help' or a greeting."""
    return (
        "🧠 *Welcome to JanAI!*\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "\n"
        "I am an AI civic complaint system.\n"
        "Send me your complaint in *English or Hindi*\n"
        "and I will:\n"
        "\n"
        "✅ Identify the category\n"
        "✅ Calculate priority score\n"
        "✅ Assign to the right department\n"
        "✅ Save with a unique Complaint ID\n"
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "*Examples:*\n"
        "\n"
        "🇬🇧 _There is a big pothole on MG Road_\n"
        "🇮🇳 _हमारे यहां पानी नहीं आ रहा है_\n"
        "\n"
        "Just type your complaint and send! 👆"
    )


# ================================================================
#  FLASK WEBHOOK
#  Twilio calls this URL every time a WhatsApp message arrives
# ================================================================

@app.route("/webhook", methods=["POST"])
def webhook():
    """
    Main webhook — called by Twilio for every incoming WhatsApp message.

    Flow:
      1. Extract sender + message text from Twilio POST data
      2. If help/greeting → send help message
      3. Otherwise → run full AI pipeline
      4. Build reply and send back via TwiML
    """
    incoming_msg = request.values.get("Body", "").strip()
    sender       = request.values.get("From", "")

    print(f"\n{'='*50}")
    print(f"📱 From:    {sender}")
    print(f"💬 Message: {incoming_msg}")
    print(f"{'='*50}")

    resp = MessagingResponse()
    msg  = resp.message()

    # ── Empty message ─────────────────────────────────────────────
    if not incoming_msg:
        msg.body("⚠️ Please send a complaint. Type *help* for instructions.")
        return str(resp)

    # ── Help / greeting ───────────────────────────────────────────
    greetings = ["help", "hi", "hello", "helo", "hey", "start",
                 "menu", "नमस्ते", "हेलो", "hai"]
    if incoming_msg.lower() in greetings:
        msg.body(build_help_message())
        return str(resp)

    # ── Extract location hint from message ────────────────────────
    location = "Unknown Location"
    for keyword in [" in ", " at ", " near ", " on "]:
        if keyword in incoming_msg.lower():
            parts = incoming_msg.lower().split(keyword, 1)
            if len(parts) > 1:
                loc_words = parts[1].split()[:4]
                location  = " ".join(loc_words).title()
            break

    # ── Send acknowledgment via REST (pipeline takes ~3-5 sec) ────
    ack_sent = False
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
        try:
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            client.messages.create(
                from_ = TWILIO_WHATSAPP_NUMBER,
                to    = sender,
                body  = "⏳ *JanAI* is analyzing your complaint...\n_Please wait 3-5 seconds._"
            )
            ack_sent = True
        except Exception as e:
            print(f"⚠️ Ack failed: {e}")

    # ── Run the AI pipeline ───────────────────────────────────────
    print(f"🔄 Analyzing: {incoming_msg[:60]}...")
    result = analyze_complaint(
        complaint_text = incoming_msg,
        location       = location,
        population     = 100
    )

    reply_text = build_reply(result, incoming_msg)
    print(f"✅ Done. Priority: {result.get('priority_label', 'ERROR')}")
    print(f"🆔 ID: {result.get('complaint_id', 'N/A')}")

    # ── Send reply ────────────────────────────────────────────────
    if ack_sent:
        # Already sent ack via REST — send reply via REST too
        try:
            client.messages.create(
                from_ = TWILIO_WHATSAPP_NUMBER,
                to    = sender,
                body  = reply_text
            )
            msg.body("")   # empty TwiML since we sent via REST
        except Exception as e:
            print(f"⚠️ REST reply failed: {e}")
            msg.body(reply_text)
    else:
        msg.body(reply_text)

    return str(resp)


@app.route("/", methods=["GET"])
def home():
    """Health check page — open in browser to verify server is running."""
    return """
    <html>
    <body style="font-family:monospace;background:#050A1A;color:#00D4FF;padding:40px;">
      <h1>🧠 JanAI WhatsApp Bot</h1>
      <p style="color:#00FF9D;">✅ Server is running</p>
      <p style="color:#E2EAF4;">Webhook: <strong>/webhook</strong></p>
      <p style="color:#6B8CAE;">
        Set your full ngrok URL + /webhook in the Twilio sandbox settings.
      </p>
    </body>
    </html>
    """


# ================================================================
#  STARTUP CHECKS + RUN
# ================================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🧠  JanAI WhatsApp Bot")
    print("=" * 50)

    # Check all required environment variables
    checks = {
        "GROQ_API_KEY":           os.environ.get("GROQ_API_KEY", ""),
        "TWILIO_ACCOUNT_SID":     os.environ.get("TWILIO_ACCOUNT_SID", ""),
        "TWILIO_AUTH_TOKEN":      os.environ.get("TWILIO_AUTH_TOKEN", ""),
        "TWILIO_WHATSAPP_NUMBER": os.environ.get("TWILIO_WHATSAPP_NUMBER", ""),
    }

    all_ok = True
    for key, val in checks.items():
        if val:
            print(f"  ✅  {key}: ...{val[-6:]}")
        else:
            print(f"  ❌  {key}: MISSING — add to .env file")
            all_ok = False

    if not all_ok:
        print("\n⚠️  Some keys missing. Check your .env file.")
    else:
        print("\n✅  All keys loaded. Bot ready!")

    print("\n📡  Flask running at:  http://localhost:5000")
    print("🔗  Webhook endpoint:  http://localhost:5000/webhook")
    print("⚡  ngrok command:     ngrok http 5000")
    print("=" * 50 + "\n")

    app.run(debug=True, port=5000)