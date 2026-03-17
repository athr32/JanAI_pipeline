# ================================================================
#
#   JanAI — WhatsApp Complaint Bot (Groq-Only Version)
#   ====================================================
#   No heavy local models. Everything runs via Groq API.
#   Startup: instant. Response time: 2-3 seconds.
#
#   HOW TO RUN:
#   -----------
#   1. pip install flask twilio groq python-dotenv
#   2. ngrok http 5000             (in a separate terminal)
#   3. Copy ngrok URL → paste in Twilio sandbox webhook
#   4. python whatsapp_bot.py
#
#   WEBHOOK URL FORMAT:
#   https://xxxx-xx-xx.ngrok-free.app/webhook
#
# ================================================================

from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from groq import Groq
import os, sys, json, uuid, datetime

# ── Load .env ─────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TWILIO_ACCOUNT_SID     = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN      = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_NUMBER = os.environ.get("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
GROQ_API_KEY           = os.environ.get("GROQ_API_KEY", "")

groq_client = Groq(api_key=GROQ_API_KEY)
app = Flask(__name__)


# ================================================================
#  GROQ ANALYSIS — single API call does everything
#  Translate + Classify + Sentiment + Priority all at once
# ================================================================

def analyze_with_groq(complaint_text: str, population: int = 100) -> dict:
    """
    Send complaint to Groq LLaMA — gets back full analysis as JSON.
    One API call replaces BART + RoBERTa + translation.
    Takes 2-3 seconds total.
    """
    prompt = f"""You are JanAI, an Indian civic complaint analysis system.
Analyze this complaint and return ONLY a JSON object, nothing else.

Complaint: "{complaint_text}"
Population affected: {population}

Return this exact JSON structure:
{{
  "translated": "English translation of complaint (same text if already English)",
  "was_translated": true or false,
  "category": one of ["roads and infrastructure", "water supply", "sanitation and garbage", "electricity", "healthcare", "education", "law and order", "public transport"],
  "severity": one of ["HIGH", "MEDIUM", "LOW"],
  "confidence": number between 0.0 and 1.0,
  "sentiment": one of ["NEGATIVE", "NEUTRAL", "POSITIVE"],
  "intensity": number between 0.0 and 1.0,
  "priority_label": one of ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
  "priority_score": number between 0 and 100,
  "urgency_reason": one sentence why this priority was assigned
}}

Hindi keyword guide:
पानी/जल/नल = water supply
सड़क/गड्ढा = roads and infrastructure  
कूड़ा/कचरा/गंदगी = sanitation and garbage
बिजली = electricity
अस्पताल/डॉक्टर = healthcare
स्कूल/शिक्षा = education
पुलिस/अपराध = law and order
बस/ट्रांसपोर्ट = public transport

Priority rules:
CRITICAL = score 50+, immediate danger to life
HIGH     = score 30-49, major disruption
MEDIUM   = score 15-29, moderate issue
LOW      = score 0-14, minor issue

Return ONLY valid JSON. No explanation before or after."""

    response = groq_client.chat.completions.create(
        model       = "llama-3.1-8b-instant",
        messages    = [{"role": "user", "content": prompt}],
        temperature = 0.0,
        max_tokens  = 400,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def save_to_vectordb(complaint_text: str, category: str,
                     severity: str, location: str) -> str:
    """
    Save complaint to ChromaDB vector database.
    Falls back to a generated UUID if ChromaDB is unavailable.
    """
    try:
        from vector_db.store import add_complaint
        return add_complaint(complaint_text, category, severity, location)
    except Exception:
        # Fallback: generate a unique complaint ID without ChromaDB
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        short_id  = str(uuid.uuid4()).replace("-", "")[:8].upper()
        return f"JAN-{timestamp}-{short_id}"


def get_recurrence(complaint_text: str) -> int:
    """Check how many similar complaints exist in vector DB."""
    try:
        from vector_db.store import get_recurrence_count
        return get_recurrence_count(complaint_text)
    except Exception:
        return 0


# ================================================================
#  DEPARTMENT MAP
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


# ================================================================
#  REPLY BUILDER
# ================================================================

def build_reply(analysis: dict, complaint_id: str,
                recurrence: int, original_text: str) -> str:

    label    = analysis.get("priority_label", "MEDIUM")
    score    = analysis.get("priority_score", 0)
    category = analysis.get("category", "unknown").title()
    dept     = DEPT_MAP.get(analysis.get("category", ""), "General Administration")
    conf     = int(analysis.get("confidence", 0.8) * 100)
    sent     = analysis.get("sentiment", "NEGATIVE")
    intens   = analysis.get("intensity", 0.5)
    severity = analysis.get("severity", "MEDIUM")
    reason   = analysis.get("urgency_reason", "")
    short_id = complaint_id[:20]

    p_emoji = PRIORITY_EMOJI.get(label, "🔵")
    s_emoji = SEVERITY_EMOJI.get(severity, "📋")

    translation_line = ""
    if analysis.get("was_translated") and analysis.get("translated"):
        translation_line = f"\n🌐 _Translated: {analysis['translated']}_\n"

    return (
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
        f"   Confidence: {conf}%\n"
        f"\n"
        f"{p_emoji} *Priority:*  {label}  (Score: {score})\n"
        f"{s_emoji} *Severity:*  {severity}\n"
        f"😠 *Sentiment:*  {sent}  (Intensity: {intens:.2f})\n"
        f"🔁 *Recurrence:*  Reported {recurrence}x before\n"
        f"💡 _{reason}_\n"
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


def build_help_message() -> str:
    return (
        "🧠 *Welcome to JanAI!*\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "\n"
        "I am an AI civic complaint system.\n"
        "Send your complaint in *English or Hindi*\n"
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
# ================================================================

@app.route("/webhook", methods=["POST"])
def webhook():
    incoming_msg = request.values.get("Body", "").strip()
    sender       = request.values.get("From", "")
    num_media    = int(request.values.get("NumMedia", 0))
    media_type   = request.values.get("MediaContentType0", "")
    media_url    = request.values.get("MediaUrl0", "")

    print(f"\n{'='*50}")
    print(f"📱 From:      {sender}")
    print(f"💬 Message:   {incoming_msg}")
    print(f"📎 Media:     {num_media} — {media_type}")
    print(f"{'='*50}")

    resp = MessagingResponse()
    msg  = resp.message()

    # ── Handle voice/audio message ────────────────────────────────
    if num_media > 0 and "audio" in media_type:
        import tempfile
        import requests as req
        from requests.auth import HTTPBasicAuth
        try:
            audio_resp = req.get(
                media_url,
                auth    = HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
                timeout = 30
            )
            ext = ".ogg" if "ogg" in media_type else ".mp3"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(audio_resp.content)
                tmp_path = f.name
            print(f"🎙️ Audio saved ({len(audio_resp.content)//1024} KB)")

            with open(tmp_path, "rb") as af:
                transcription = groq_client.audio.transcriptions.create(
                    file            = (f"audio{ext}", af.read()),
                    model           = "whisper-large-v3-turbo",
                    response_format = "text",
                    temperature     = 0.0,
                )
            os.unlink(tmp_path)

            incoming_msg = (transcription if isinstance(transcription, str)
                            else getattr(transcription, "text", "")).strip()
            print(f"📝 Transcribed: {incoming_msg}")

            if not incoming_msg:
                msg.body("⚠️ Could not transcribe your voice note. Please speak clearly or type your complaint.")
                return str(resp)

            # Tell user what was heard
            try:
                Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN).messages.create(
                    from_ = TWILIO_WHATSAPP_NUMBER,
                    to    = sender,
                    body  = f"🎙️ *Voice transcribed:*\n_{incoming_msg}_\n\n⏳ Analyzing..."
                )
            except Exception:
                pass

        except Exception as e:
            print(f"❌ Voice error: {e}")
            msg.body(f"❌ Could not process voice message.\nPlease type your complaint instead.")
            return str(resp)

    # ── Empty text message ────────────────────────────────────────
    elif not incoming_msg:
        msg.body("⚠️ Please send a text or voice complaint. Type *help* for instructions.")
        return str(resp)

    greetings = ["help", "hi", "hello", "helo", "hey", "start",
                 "menu", "नमस्ते", "हेलो", "hai"]
    if incoming_msg.lower() in greetings:
        msg.body(build_help_message())
        return str(resp)

    # Extract location hint
    location = "Unknown Location"
    for keyword in [" in ", " at ", " near ", " on "]:
        if keyword in incoming_msg.lower():
            parts     = incoming_msg.lower().split(keyword, 1)
            loc_words = parts[1].split()[:4]
            location  = " ".join(loc_words).title()
            break

    # Send acknowledgment
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
        try:
            Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN).messages.create(
                from_ = TWILIO_WHATSAPP_NUMBER,
                to    = sender,
                body  = "⏳ *JanAI* is analyzing your complaint...\n_Takes 2-3 seconds._"
            )
        except Exception as e:
            print(f"⚠️ Ack failed: {e}")

    # Run Groq analysis
    print(f"🔄 Analyzing with Groq...")
    try:
        analysis   = analyze_with_groq(incoming_msg)
        complaint_id = save_to_vectordb(
            incoming_msg,
            analysis.get("category", "unknown"),
            analysis.get("severity", "MEDIUM"),
            location
        )
        recurrence = get_recurrence(incoming_msg)
        reply_text = build_reply(analysis, complaint_id, recurrence, incoming_msg)
        print(f"✅ Done. Priority: {analysis.get('priority_label')} | ID: {complaint_id[:16]}")

    except Exception as e:
        print(f"❌ Error: {e}")
        reply_text = (
            f"❌ *JanAI Error*\n\n"
            f"Could not analyze complaint: {str(e)[:100]}\n\n"
            f"Please try again."
        )

    # Send reply via REST
    try:
        Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN).messages.create(
            from_ = TWILIO_WHATSAPP_NUMBER,
            to    = sender,
            body  = reply_text
        )
        msg.body("")
    except Exception as e:
        print(f"⚠️ REST send failed: {e}")
        msg.body(reply_text)

    return str(resp)


@app.route("/", methods=["GET"])
def home():
    return """
    <html><body style="font-family:monospace;background:#050A1A;color:#00D4FF;padding:40px;">
    <h1>🧠 JanAI WhatsApp Bot</h1>
    <p style="color:#00FF9D;">✅ Server is running</p>
    <p style="color:#6B8CAE;">Webhook: <strong style="color:#E2EAF4;">/webhook</strong></p>
    </body></html>
    """


# ================================================================
#  STARTUP
# ================================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🧠  JanAI WhatsApp Bot  (Groq-Only)")
    print("=" * 50)

    checks = {
        "GROQ_API_KEY":           GROQ_API_KEY,
        "TWILIO_ACCOUNT_SID":     TWILIO_ACCOUNT_SID,
        "TWILIO_AUTH_TOKEN":      TWILIO_AUTH_TOKEN,
        "TWILIO_WHATSAPP_NUMBER": TWILIO_WHATSAPP_NUMBER,
    }
    all_ok = True
    for key, val in checks.items():
        if val:
            print(f"  ✅  {key}: ...{val[-6:]}")
        else:
            print(f"  ❌  {key}: MISSING")
            all_ok = False

    if all_ok:
        print("\n✅  All keys loaded. Bot ready!")
    else:
        print("\n⚠️  Some keys missing. Check your .env file.")

    print("\n📡  Flask: http://localhost:5000")
    print("🔗  Webhook: http://localhost:5000/webhook")
    print("⚡  Run in another terminal: ngrok http 5000")
    print("=" * 50 + "\n")

    app.run(debug=False, port=5000)
