# ============================================================
#  JanAI — Streamlit Test Interface v3 (Voice Fixed)
#  Run: streamlit run app.py
#  pip install streamlit streamlit-mic-recorder openai-whisper
# ============================================================

import streamlit as st
import sys
import os
import time
import tempfile

# ── MUST be first Streamlit call ─────────────────────────────
st.set_page_config(
    page_title="JanAI — Pipeline Tester",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Initialize ALL session state keys at the very top ────────
defaults = {
    "prefill_text":        "There is a massive pothole on MG Road near the bus stop causing accidents daily",
    "voice_transcript":    "",
    "voice_lang":          "hi",
    "run_voice_pipeline":  False,
    "upload_transcript":   "",
    "upload_lang":         "hi",
    "run_upload_pipeline": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Add project root to path so modules import correctly ─────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Load .env file (GROQ_API_KEY, BHASHINI keys etc.) ────────
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #050A1A; color: #E2EAF4; }

.janai-header {
    background: linear-gradient(135deg, #0A1628 0%, #0D1F3C 100%);
    border: 1px solid #1A3A6B; border-radius: 16px;
    padding: 28px 40px; margin-bottom: 24px;
    position: relative; overflow: hidden;
}
.janai-header::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:220px; height:220px;
    background:radial-gradient(circle,rgba(0,212,255,0.10) 0%,transparent 70%);
    border-radius:50%;
}
.janai-title {
    font-family:'Space Mono',monospace; font-size:2.6rem;
    font-weight:700; color:#00D4FF; margin:0; letter-spacing:-1px;
}
.janai-subtitle { color:#6B8CAE; font-size:0.95rem; margin-top:6px; font-weight:300; }

.stTabs [data-baseweb="tab-list"] {
    background:#0A1628; border-radius:10px; padding:4px; gap:4px; border:1px solid #1A3A6B;
}
.stTabs [data-baseweb="tab"] {
    background:transparent !important; color:#6B8CAE !important;
    border-radius:8px !important; font-family:'Space Mono',monospace !important;
    font-size:0.78rem !important; font-weight:700 !important;
    letter-spacing:1px; padding:10px 18px !important;
}
.stTabs [aria-selected="true"] {
    background:#0D1F3C !important; color:#00D4FF !important; border:1px solid #1A3A6B !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top:20px; }

.result-card {
    background:#0D1F3C; border:1px solid #1A3A6B;
    border-radius:12px; padding:20px 24px; margin-bottom:16px;
}
.result-card-title {
    font-family:'Space Mono',monospace; font-size:0.68rem; color:#6B8CAE;
    text-transform:uppercase; letter-spacing:2px; margin-bottom:8px;
}
.result-card-value { font-size:1.1rem; color:#E2EAF4; font-weight:500; }

.voice-card {
    background:linear-gradient(135deg,#0D1F3C 0%,#0A1A30 100%);
    border:1px solid #7B2FBE44; border-radius:14px;
    padding:24px; margin-bottom:16px;
}
.voice-title {
    font-family:'Space Mono',monospace; font-size:0.72rem; color:#7B2FBE;
    text-transform:uppercase; letter-spacing:2px; margin-bottom:12px;
}
.transcript-box {
    background:#050A1A; border:1px solid #7B2FBE55;
    border-radius:10px; padding:16px 18px; margin-top:10px;
    font-size:0.95rem; color:#E2EAF4; min-height:60px; line-height:1.6;
}
.transcript-label {
    font-family:'Space Mono',monospace; font-size:0.65rem; color:#7B2FBE;
    text-transform:uppercase; letter-spacing:2px; margin-bottom:6px;
}
.lang-badge {
    display:inline-block; background:#7B2FBE22; color:#B47FE8;
    border:1px solid #7B2FBE44; border-radius:12px;
    padding:2px 10px; font-size:0.72rem; font-family:'Space Mono',monospace; margin-left:8px;
}
.badge-critical { background:#FF3E3E22;color:#FF3E3E;border:1px solid #FF3E3E55;padding:4px 14px;border-radius:20px;font-weight:700;font-size:0.9rem; }
.badge-high     { background:#FF8C0022;color:#FF8C00;border:1px solid #FF8C0055;padding:4px 14px;border-radius:20px;font-weight:700;font-size:0.9rem; }
.badge-medium   { background:#FFD70022;color:#FFD700;border:1px solid #FFD70055;padding:4px 14px;border-radius:20px;font-weight:700;font-size:0.9rem; }
.badge-low      { background:#00FF9D22;color:#00FF9D;border:1px solid #00FF9D55;padding:4px 14px;border-radius:20px;font-weight:700;font-size:0.9rem; }

.score-bar-bg { background:#1A3A6B;border-radius:8px;height:12px;margin-top:8px;overflow:hidden; }
.score-bar-fill { height:100%;border-radius:8px; }
.sentiment-negative { color:#FF6B35;font-weight:600; }
.sentiment-positive { color:#00FF9D;font-weight:600; }
.sentiment-neutral  { color:#FFD700;font-weight:600; }

.step-item {
    display:flex;align-items:center;gap:12px;
    padding:9px 0;border-bottom:1px solid #1A3A6B;font-size:0.88rem;
}
.step-dot { width:10px;height:10px;border-radius:50%;flex-shrink:0; }

section[data-testid="stSidebar"] {
    background-color:#0A1628 !important; border-right:1px solid #1A3A6B;
}
.stTextArea textarea, .stTextInput input {
    background-color:#0D1F3C !important; border:1px solid #1A3A6B !important;
    color:#E2EAF4 !important; border-radius:8px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color:#00D4FF !important;
    box-shadow:0 0 0 2px rgba(0,212,255,0.12) !important;
}
.stButton > button {
    background:linear-gradient(135deg,#00D4FF,#0099CC) !important;
    color:#050A1A !important; font-family:'Space Mono',monospace !important;
    font-weight:700 !important; font-size:0.88rem !important;
    border:none !important; border-radius:10px !important;
    padding:12px 24px !important; width:100% !important; letter-spacing:1px;
}
.stButton > button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 8px 24px rgba(0,212,255,0.25) !important;
}
.big-score { font-family:'Space Mono',monospace;font-size:3.2rem;font-weight:700;line-height:1; }
hr { border-color:#1A3A6B !important; }
#MainMenu, footer, header { visibility:hidden; }
.stSelectbox > div > div {
    background:#0D1F3C !important; border:1px solid #1A3A6B !important;
    color:#E2EAF4 !important; border-radius:8px !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS  — defined BEFORE they are called
# ══════════════════════════════════════════════════════════════

def get_initial_prompt(lang: str) -> str:
    """Prime Whisper with civic complaint context for Indian languages."""
    prompts = {
        "hi": "यह एक नागरिक शिकायत है। सड़क, पानी, बिजली, कचरा, स्वास्थ्य, स्कूल।",
        "mr": "ही एक नागरिक तक्रार आहे। रस्ता, पाणी, वीज, कचरा, आरोग्य, शाळा.",
        "ta": "இது ஒரு குடிமக்கள் புகார். சாலை, தண்ணீர், மின்சாரம், குப்பை.",
        "te": "ఇది పౌర ఫిర్యాదు. రోడ్డు, నీరు, విద్యుత్, చెత్త, ఆరోగ్యం.",
        "bn": "এটি একটি নাগরিক অভিযোগ। রাস্তা, জল, বিদ্যুৎ, আবর্জনা।",
        "gu": "આ એક નાગરિક ફરિયાદ છે. રસ્તો, પાણી, વીજળી, કચરો.",
        "kn": "ಇದು ಒಂದು ನಾಗರಿಕ ದೂರು. ರಸ್ತೆ, ನೀರು, ವಿದ್ಯುತ್, ತ್ಯಾಜ್ಯ.",
        "pa": "ਇਹ ਇੱਕ ਨਾਗਰਿਕ ਸ਼ਿਕਾਇਤ ਹੈ। ਸੜਕ, ਪਾਣੀ, ਬਿਜਲੀ, ਕੂੜਾ.",
        "en": "This is a citizen complaint about road, water, electricity, garbage, or healthcare issues.",
    }
    return prompts.get(lang, prompts["en"])


def transcribe_with_groq(file_path: str, lang: str) -> dict:
    """
    Groq Whisper API — fastest option, free, cloud-based.
    25x faster than local Whisper. Uses whisper-large-v3-turbo model.
    Needs: pip install groq  +  GROQ_API_KEY in .env
    """
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError("Groq not installed. Run: pip install groq")

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. "
            "Get free key at https://console.groq.com "
            "and add GROQ_API_KEY=gsk_... to your .env file"
        )

    client = Groq(api_key=api_key)

    # Groq needs file size < 25MB — check first
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > 24:
        raise RuntimeError(f"Audio file too large ({file_size_mb:.1f}MB). Groq limit is 25MB.")

    # Groq language code map — must be ISO 639-1 two-letter codes
    lang_map = {
        "hi": "hi",   # Hindi
        "mr": "mr",   # Marathi
        "ta": "ta",   # Tamil
        "te": "te",   # Telugu
        "bn": "bn",   # Bengali
        "gu": "gu",   # Gujarati
        "kn": "kn",   # Kannada
        "pa": "pa",   # Punjabi
        "en": "en",   # English
    }
    groq_lang = lang_map.get(lang, lang)

    with open(file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    response = client.audio.transcriptions.create(
        file            = (os.path.basename(file_path), audio_bytes),
        model           = "whisper-large-v3-turbo",
        language        = groq_lang,
        prompt          = get_initial_prompt(lang),
        response_format = "text",   # "text" is more reliable than verbose_json for Indian langs
        temperature     = 0.0,
    )

    # When format="text", response IS the transcript string directly
    if isinstance(response, str):
        transcript = response.strip()
    else:
        transcript = getattr(response, "text", str(response)).strip()

    if not transcript:
        raise RuntimeError("Groq returned empty transcript. Try speaking more clearly.")

    return {
        "transcript": transcript,
        "language":   groq_lang,
        "source":     "Groq Whisper large-v3-turbo ⚡"
    }


def transcribe_with_local_whisper(file_path: str, lang: str) -> dict:
    """Local Whisper fallback — slower but works offline."""
    import site, sys

    for key in list(sys.modules.keys()):
        if "whisper" in key:
            del sys.modules[key]

    whisper_root = None
    all_site = site.getsitepackages()
    try:
        all_site.append(site.getusersitepackages())
    except Exception:
        pass

    for sp in all_site:
        candidate = os.path.join(sp, "whisper", "__init__.py")
        if os.path.exists(candidate):
            whisper_root = os.path.dirname(os.path.dirname(candidate))
            break

    if whisper_root is None:
        raise RuntimeError("openai-whisper not found. Run: pip install openai-whisper")

    original_path = sys.path[:]
    sys.path = [whisper_root] + [p for p in sys.path if p != whisper_root]

    try:
        import whisper as _w
        model  = _w.load_model("small")   # small for speed; change to medium for accuracy
        result = model.transcribe(
            file_path,
            language                   = lang if lang != "en" else None,
            beam_size                  = 5,
            best_of                    = 5,
            temperature                = 0.0,
            condition_on_previous_text = True,
            initial_prompt             = get_initial_prompt(lang),
            fp16                       = False,
            verbose                    = False,
        )
        return {
            "transcript": result["text"].strip(),
            "language":   result.get("language", lang),
            "source":     "Whisper local (small)"
        }
    except Exception as e:
        raise RuntimeError(f"Local Whisper failed: {e}")
    finally:
        sys.path = original_path


def transcribe_audio_file(file_path: str, lang: str, engine: str) -> dict:
    """
    Route audio to the correct STT engine.
    Priority: Groq API (fastest) → Bhashini → Sarvam → Local Whisper (fallback)
    """
    # ── Groq API — fastest, free, recommended ────────────────
    if engine == "Groq Whisper API ⚡ (Recommended)":
        return transcribe_with_groq(file_path, lang)

    # ── Bhashini ──────────────────────────────────────────────
    elif engine == "Bhashini (Indian Gov)":
        try:
            from stt.transcriber import transcribe_with_bhashini
            return transcribe_with_bhashini(file_path, lang)
        except Exception as e:
            st.warning(f"Bhashini failed ({e}). Falling back to Groq...")
            return transcribe_with_groq(file_path, lang)

    # ── Sarvam ────────────────────────────────────────────────
    elif engine == "Sarvam AI":
        try:
            from stt.sarvam_stt import transcribe_with_sarvam
            return transcribe_with_sarvam(file_path, f"{lang}-IN")
        except Exception as e:
            st.warning(f"Sarvam failed ({e}). Falling back to Groq...")
            return transcribe_with_groq(file_path, lang)

    # ── Local Whisper — offline fallback ─────────────────────
    else:
        return transcribe_with_local_whisper(file_path, lang)




def classify_with_groq(text: str) -> dict:
    """
    Use Groq LLaMA to classify complaint category and severity.
    Works directly on Hindi/Marathi/any language — no translation needed.
    Much more accurate than BART for Indian language complaints.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return None  # fallback to BART

    try:
        from groq import Groq
        import json as _json

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model    = "llama3-8b-8192",
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an AI classifier for Indian civic complaints. "
                        "Analyze the complaint and return ONLY a JSON object with these exact fields:\n"
                        "{\n"
                        '  "category": one of ["roads and infrastructure", "water supply", '
                        '"sanitation and garbage", "electricity", "healthcare", "education", '
                        '"law and order", "public transport"],\n'
                        '  "severity": one of ["HIGH", "MEDIUM", "LOW"],\n'
                        '  "confidence": a number between 0.0 and 1.0,\n'
                        '  "reason": one sentence explaining why\n'
                        "}\n\n"
                        "STRICT Hindi/Marathi keyword mapping — always follow this:\n\n"
                        "WATER SUPPLY: पानी, जल, नल, पाइप, टंकी, पानी नहीं, पानी की समस्या, "
                        "जलापूर्ति, पानी बंद, पेयजल, पाणी (Marathi)\n\n"
                        "SANITATION AND GARBAGE: कूड़ा, कूड़े, कचरा, गंदगी, सफाई, नाली, "
                        "सीवर, बदबू, मैला, झाड़ू, कूड़ेदान, डस्टबिन, साफ नहीं, "
                        "कचरा नहीं उठाया, कचरागाड़ी, स्वच्छता\n\n"
                        "ROADS AND INFRASTRUCTURE: सड़क, गड्ढा, रास्ता, पुल, टूटी सड़क, "
                        "खड्डा, निर्माण, फुटपाथ\n\n"
                        "ELECTRICITY: बिजली, लाइट, अंधेरा, बत्ती, ट्रांसफार्मर, करंट, "
                        "बिजली गुल, स्ट्रीट लाइट\n\n"
                        "HEALTHCARE: अस्पताल, डॉक्टर, स्वास्थ्य, दवाई, नर्स, इलाज, "
                        "चिकित्सा, क्लिनिक\n\n"
                        "EDUCATION: स्कूल, शिक्षा, शिक्षक, विद्यालय, कॉलेज, पढ़ाई\n\n"
                        "LAW AND ORDER: पुलिस, अपराध, चोरी, लड़ाई, असुरक्षा, डर\n\n"
                        "PUBLIC TRANSPORT: बस, रिक्शा, ट्रांसपोर्ट, ऑटो, टैक्सी\n\n"
                        "IMPORTANT: कूड़ा/कूड़े/कचरा ALWAYS = sanitation and garbage. "
                        "पानी/जल ALWAYS = water supply. "
                        "Never classify these as roads.\n\n"
                        "Return ONLY valid JSON. No explanation before or after."
                    )
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature = 0.0,
            max_tokens  = 256,
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = _json.loads(raw)

        # Validate required fields
        if "category" not in result or "severity" not in result:
            return None

        return {
            "category":   result["category"],
            "severity":   result.get("severity", "MEDIUM"),
            "confidence": float(result.get("confidence", 0.85)),
            "reason":     result.get("reason", ""),
            "source":     "Groq LLaMA"
        }

    except Exception as e:
        return None  # silent fallback to BART


def analyze_sentiment_with_groq(text: str) -> dict:
    """
    Use Groq LLaMA for sentiment analysis.
    Works on any language — much better than RoBERTa for Hindi.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return None

    try:
        from groq import Groq
        import json as _json

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model    = "llama3-8b-8192",
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Analyze the sentiment of this civic complaint. "
                        "Return ONLY a JSON object:\n"
                        "{\n"
                        '  "sentiment": one of ["NEGATIVE", "NEUTRAL", "POSITIVE"],\n'
                        '  "negative_intensity": number 0.0 to 1.0 (how urgent/angry),\n'
                        '  "urgency": one of ["CRITICAL", "HIGH", "MEDIUM", "LOW"]\n'
                        "}\n"
                        "Return ONLY JSON. No other text."
                    )
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature = 0.0,
            max_tokens  = 128,
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = _json.loads(raw)

        return {
            "sentiment":          result.get("sentiment", "NEGATIVE"),
            "negative_intensity": float(result.get("negative_intensity", 0.7)),
            "urgency":            result.get("urgency", "MEDIUM"),
            "source":             "Groq LLaMA"
        }

    except Exception:
        return None


def translate_to_english_simple(text: str) -> str:
    """
    Translate any Indian language text to English using Groq.
    Raises a visible error if translation fails so we can debug.
    """
    # Load .env file explicitly every time
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except Exception:
        pass

    api_key = os.environ.get("GROQ_API_KEY", "").strip()

    if not api_key:
        st.error("❌ GROQ_API_KEY not found in .env file — translation skipped.")
        return text

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model    = "llama-3.1-8b-instant",   # most reliable model
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Translate this Hindi/Marathi text to English. "
                        f"Reply with ONLY the English translation, nothing else.\n\n"
                        f"Text: {text}"
                    )
                }
            ],
            temperature = 0.0,
            max_tokens  = 300,
        )
        translated = response.choices[0].message.content.strip()

        # Verify it actually translated (result should be mostly ASCII)
        non_ascii = sum(1 for c in translated if ord(c) > 127)
        if non_ascii > len(translated) * 0.3:
            # Still non-English — translation failed
            st.warning(f"⚠️ Groq returned non-English text. Raw response: {translated[:100]}")
            return text

        return translated if translated else text

    except Exception as e:
        st.error(f"❌ Translation error: {e}")
        return text


def run_pipeline(complaint_text: str, population: int,
                 location: str, progress_placeholder) -> tuple:
    """
    Run the full JanAI AI pipeline.
    Returns (nlp_result, sentiment_result, priority_result, complaint_id)
    """
    # ── Step 1: Detect if non-English and translate ─────────────
    # Check if text has non-ASCII chars (Hindi/Marathi/etc)
    non_ascii = sum(1 for c in complaint_text if ord(c) > 127)
    is_non_english = non_ascii > len(complaint_text) * 0.2

    all_steps = [
        ("Input Processing",   "wait"),
        ("Translating to English", "wait"),
        ("NLP Classification", "wait"),
        ("Sentiment Analysis", "wait"),
        ("Priority Scoring",   "wait"),
        ("Vector DB Storage",  "wait"),
    ]

    def show(steps):
        colors = {"done":"#00FF9D","running":"#00D4FF","wait":"#1A3A6B"}
        icons  = {"done":"✓","running":"◉","wait":"○"}
        html = ""
        for lbl, st_ in steps:
            c    = colors[st_]
            glow = f"box-shadow:0 0 8px {c};" if st_ == "running" else ""
            html += f"""
            <div class="step-item">
              <div class="step-dot" style="background:{c};{glow}"></div>
              <span style="color:{'#E2EAF4' if st_!='wait' else '#6B8CAE'}">{icons[st_]} {lbl}</span>
            </div>"""
        progress_placeholder.markdown(
            f'<div class="result-card">{html}</div>',
            unsafe_allow_html=True
        )

    from nlp.classifier         import classify_issue
    from nlp.sentiment          import analyze_sentiment
    from vector_db.store        import add_complaint
    from priority_engine.scorer import compute_priority_score

    # Step 1
    all_steps[0] = ("Input Processing", "running"); show(all_steps); time.sleep(0.2)
    all_steps[0] = ("Input Processing", "done")

    # Step 2 — Translate to English if needed
    all_steps[1] = ("Translating to English", "running"); show(all_steps)
    if is_non_english:
        english_text = translate_to_english_simple(complaint_text)
    else:
        english_text = complaint_text
    all_steps[1] = ("Translating to English", "done")

    # ── New show() that ALWAYS keeps translation box visible ──
    def show_with_translation(steps):
        colors = {"done":"#00FF9D","running":"#00D4FF","wait":"#1A3A6B"}
        icons  = {"done":"✓","running":"◉","wait":"○"}
        html = ""
        for lbl, st_ in steps:
            c    = colors[st_]
            glow = f"box-shadow:0 0 8px {c};" if st_ == "running" else ""
            html += f"""
            <div class="step-item">
              <div class="step-dot" style="background:{c};{glow}"></div>
              <span style="color:{'#E2EAF4' if st_!='wait' else '#6B8CAE'}">{icons[st_]} {lbl}</span>
            </div>"""
        # Always append translation box if non-English
        translation_box = ""
        if is_non_english:
            translation_box = f"""
            <div style="margin-top:14px; padding:14px 16px;
                        background:#050A1A; border:2px solid #00FF9D55;
                        border-radius:10px;">
              <div style="font-size:0.65rem; color:#00FF9D; font-family:monospace;
                          letter-spacing:2px; text-transform:uppercase; margin-bottom:8px;">
                🌐 English Translation (sent to NLP)
              </div>
              <div style="display:grid; grid-template-columns:1fr auto 1fr; gap:10px; align-items:start;">
                <div>
                  <div style="font-size:0.7rem; color:#6B8CAE; margin-bottom:4px;">Original</div>
                  <div style="font-size:0.88rem; color:#B8D4F0;">{complaint_text}</div>
                </div>
                <div style="color:#00FF9D; font-size:1.2rem; padding-top:16px;">→</div>
                <div>
                  <div style="font-size:0.7rem; color:#00FF9D; margin-bottom:4px;">Translated ✓</div>
                  <div style="font-size:0.95rem; color:#E2EAF4; font-weight:600;">{english_text}</div>
                </div>
              </div>
            </div>"""
        progress_placeholder.markdown(
            f'<div class="result-card">{html}{translation_box}</div>',
            unsafe_allow_html=True
        )

    show_with_translation(all_steps)

    # Step 3 — NLP on English text ALWAYS
    all_steps[2] = ("NLP Classification", "running"); show_with_translation(all_steps); time.sleep(0.2)
    nlp_result = classify_issue(english_text)
    all_steps[2] = ("NLP Classification", "done")

    # Step 4 — Sentiment on English text ALWAYS
    all_steps[3] = ("Sentiment Analysis", "running"); show_with_translation(all_steps); time.sleep(0.2)
    sentiment_result = analyze_sentiment(english_text)
    all_steps[3] = ("Sentiment Analysis", "done")

    # Step 5 — Priority on English text
    all_steps[4] = ("Priority Scoring", "running"); show_with_translation(all_steps); time.sleep(0.2)
    priority_result = compute_priority_score(english_text, population)
    all_steps[4] = ("Priority Scoring", "done")

    # Step 6 — Store original text in DB
    all_steps[5] = ("Vector DB Storage", "running"); show_with_translation(all_steps); time.sleep(0.2)
    complaint_id = add_complaint(
        complaint_text,
        nlp_result["category"],
        nlp_result["severity"],
        location
    )
    all_steps[5] = ("Vector DB Storage", "done"); show_with_translation(all_steps)

    return nlp_result, sentiment_result, priority_result, complaint_id, complaint_text, english_text


DEPT_MAP = {
    "roads and infrastructure": "PWD — Public Works Department",
    "water supply":             "Water Supply Board",
    "sanitation and garbage":   "Municipal Sanitation Department",
    "electricity":              "State Electricity Board",
    "healthcare":               "District Health Office",
    "education":                "District Education Office",
    "law and order":            "Police Department",
    "public transport":         "Transport Authority",
}

def show_results(nlp_r, sent_r, pri_r, cid, original_text="", translated_text=""):
    """Render all result cards after pipeline completes."""

    # ── Translation card — shown prominently so user can verify ──
    if translated_text and translated_text != original_text:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0A1F0A,#0D2A0D);
                    border:1px solid #00FF9D55; border-radius:12px;
                    padding:18px 22px; margin-bottom:16px;">
          <div style="font-family:'Space Mono',monospace; font-size:0.68rem;
                      color:#00FF9D; text-transform:uppercase; letter-spacing:2px;
                      margin-bottom:12px;">
            🌐 Translation Used for NLP Analysis
          </div>
          <div style="display:grid; grid-template-columns:1fr auto 1fr; gap:12px; align-items:center;">
            <div style="background:#050A1A; border:1px solid #1A3A6B;
                        border-radius:8px; padding:12px 14px;">
              <div style="font-size:0.7rem; color:#6B8CAE; margin-bottom:6px;">
                🗣 Original (Hindi/Other)
              </div>
              <div style="font-size:0.95rem; color:#B8D4F0; line-height:1.5;">
                {original_text}
              </div>
            </div>
            <div style="font-size:1.4rem; color:#00FF9D; text-align:center;">→</div>
            <div style="background:#050A1A; border:1px solid #00FF9D44;
                        border-radius:8px; padding:12px 14px;">
              <div style="font-size:0.7rem; color:#00FF9D; margin-bottom:6px;">
                🔤 Translated to English
              </div>
              <div style="font-size:0.95rem; color:#E2EAF4; line-height:1.5; font-weight:500;">
                {translated_text}
              </div>
            </div>
          </div>
          <div style="margin-top:10px; font-size:0.78rem; color:#6B8CAE;">
            ⚠️ If the translation looks wrong, go back and edit the transcript before analyzing.
          </div>
        </div>
        """, unsafe_allow_html=True)

    score     = pri_r["total_score"]
    label     = pri_r["priority_label"]
    bar_pct   = min(int(score), 100)
    bar_color = {"CRITICAL":"#FF3E3E","HIGH":"#FF8C00",
                 "MEDIUM":"#FFD700","LOW":"#00FF9D"}.get(label, "#00D4FF")

    # ── Priority score card ───────────────────────────────────
    st.markdown(f"""
    <div class="result-card">
      <div class="result-card-title">Priority Score</div>
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:8px;">
        <span class="big-score" style="color:{bar_color}">{score}</span>
        <span class="badge-{label.lower()}">{label}</span>
      </div>
      <div class="score-bar-bg">
        <div class="score-bar-fill" style="width:{bar_pct}%;background:{bar_color};"></div>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:6px;">
        <span style="font-size:0.75rem;color:#6B8CAE;">0 — LOW</span>
        <span style="font-size:0.75rem;color:#6B8CAE;">50+ — CRITICAL</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Three detail cards ────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="result-card">
          <div class="result-card-title">Category</div>
          <div class="result-card-value" style="color:#00D4FF">{nlp_r['category'].title()}</div>
          <div style="font-size:0.8rem;color:#6B8CAE;margin-top:4px;">Confidence: {int(nlp_r['confidence']*100)}%</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        s     = sent_r["sentiment"]
        emoji = {"NEGATIVE":"😠","POSITIVE":"😊","NEUTRAL":"😐"}.get(s,"😐")
        st.markdown(f"""
        <div class="result-card">
          <div class="result-card-title">Sentiment</div>
          <div class="result-card-value sentiment-{s.lower()}">{emoji} {s}</div>
          <div style="font-size:0.8rem;color:#6B8CAE;margin-top:4px;">Intensity: {sent_r['negative_intensity']:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        sc = {"HIGH":"#FF3E3E","MEDIUM":"#FFD700","LOW":"#00FF9D"}.get(nlp_r["severity"],"#00D4FF")
        st.markdown(f"""
        <div class="result-card">
          <div class="result-card-title">Severity</div>
          <div class="result-card-value" style="color:{sc}">{nlp_r['severity']}</div>
          <div style="font-size:0.8rem;color:#6B8CAE;margin-top:4px;">Recurrence: {pri_r['recurrence_count']}x</div>
        </div>""", unsafe_allow_html=True)

    # ── Breakdown card ────────────────────────────────────────
    bd = pri_r["score_breakdown"]
    st.markdown(f"""
    <div class="result-card">
      <div class="result-card-title">Score Breakdown</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px;">
        <div><div style="color:#6B8CAE;font-size:0.8rem;">Urgency × Severity</div>
             <div style="color:#00D4FF;font-weight:600;">{bd['urgency_x_severity']}</div></div>
        <div><div style="color:#6B8CAE;font-size:0.8rem;">Recurrence</div>
             <div style="color:#7B2FBE;font-weight:600;">{bd['recurrence']}</div></div>
        <div><div style="color:#6B8CAE;font-size:0.8rem;">Population Impact</div>
             <div style="color:#00FF9D;font-weight:600;">{bd['population']}</div></div>
        <div><div style="color:#6B8CAE;font-size:0.8rem;">Sentiment</div>
             <div style="color:#FF6B35;font-weight:600;">{bd['sentiment']}</div></div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Department + ID card ──────────────────────────────────
    dept = DEPT_MAP.get(nlp_r["category"], "General Administration")
    st.markdown(f"""
    <div class="result-card">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
          <div class="result-card-title">Assigned Department</div>
          <div class="result-card-value">🏛 {dept}</div>
        </div>
        <div style="text-align:right;">
          <div class="result-card-title">Complaint ID</div>
          <div style="font-family:'Space Mono',monospace;color:#6B8CAE;font-size:0.8rem;">{cid[:20]}...</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


def render_empty_result(icon, msg):
    st.markdown(f"""
    <div class="result-card" style="text-align:center;padding:48px 24px;">
      <div style="font-size:3rem;margin-bottom:16px;">{icon}</div>
      <div style="color:#6B8CAE;">{msg}</div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="janai-header">
  <p class="janai-title">🧠 JanAI</p>
  <p class="janai-subtitle">AI Pipeline Tester — Text · Live Voice · Upload Audio</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")
    population = st.slider("👥 Population Affected", 1, 5000, 100, 10)
    location   = st.text_input("📍 Location", value="MG Road, Pune")

    st.markdown("---")
    st.markdown("### 🎙️ STT Engine")
    stt_engine = st.selectbox("Engine", [
        "Groq Whisper API ⚡ (Recommended)"
        # "Bhashini (Indian Gov)",
        # "Sarvam AI",
        # "Whisper (Offline)",
    ])
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key:
        st.success(f"✅ Groq key loaded: ...{groq_key[-6:]}")
    else:
        st.error("❌ GROQ_API_KEY missing from .env")
        st.caption("Add GROQ_API_KEY=gsk_xxx to your .env file")
    voice_language = st.selectbox("Language", ["Hindi (hi)","English (en)"
        # , "Marathi (mr)", 
        # "Tamil (ta)", "Telugu (te)", "Bengali (bn)",
        # "Gujarati (gu)", "Kannada (kn)"
    ])
    lang_code = voice_language.split("(")[1].replace(")", "").strip()

    st.markdown("---")
    st.markdown("### 🧪 Sample Complaints")
    samples = {
        "🛣️ Road":       "There is a massive pothole on MG Road near the bus stop causing accidents daily",
        "💧 Water":      "No water supply in our area for 3 days straight, people are suffering",
        "🗑️ Garbage":    "Garbage not collected since last week, entire street smells terrible",
        "⚡ Electricity": "Street lights not working near the school for 2 weeks",
        "🚨 Emergency":  "Emergency! Bridge wall collapsed near highway, danger to lives immediately",
        "🏥 Healthcare": "No doctor available at primary health centre since Monday",
    }
    for lbl, txt in samples.items():
        if st.button(lbl, key=f"s_{lbl}"):
            st.session_state["prefill_text"] = txt

    st.markdown("---")
    st.markdown("### 📦 Module Status")
    for name, mod, fn in [
        ("NLP Classifier",  "nlp.classifier",          "classify_issue"),
        ("Sentiment",       "nlp.sentiment",            "analyze_sentiment"),
        ("Vector DB",       "vector_db.store",          "add_complaint"),
        ("Priority Engine", "priority_engine.scorer",   "compute_priority_score"),
    ]:
        try:
            m = __import__(mod, fromlist=[fn]); getattr(m, fn)
            st.markdown(f"✅ **{name}**")
        except:
            st.markdown(f"❌ **{name}**")


# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab_text, tab_voice, tab_upload = st.tabs([
    "📝  TEXT COMPLAINT",
    "🎙️  LIVE VOICE",
    "📁  UPLOAD AUDIO",
])


# ────────────────────────────────────────────────────────────
#  TAB 1 — TEXT
# ────────────────────────────────────────────────────────────
with tab_text:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("#### 📝 Type Your Complaint")
        complaint_text = st.text_area(
            "complaint",
            value=st.session_state["prefill_text"],
            height=160,
            label_visibility="collapsed",
            placeholder="Describe the issue...",
            key="text_input"
        )
        a, b = st.columns(2)
        a.metric("👥 Population", f"{population:,}")
        b.metric("📍 Location", location[:18] + "..." if len(location) > 18 else location)
        st.markdown("<br>", unsafe_allow_html=True)
        run_text = st.button("⚡ RUN AI PIPELINE", key="btn_text")

    with col_r:
        st.markdown("#### 📊 Results")
        if run_text:
            if complaint_text.strip():
                ph = st.empty()
                try:
                    nr, sr, pr, cid, orig, trans = run_pipeline(complaint_text, population, location, ph)
                    st.markdown("---")
                    show_results(nr, sr, pr, cid, orig, trans)
                except Exception as e:
                    st.error(f"❌ {e}")
            else:
                st.warning("⚠️ Please enter a complaint.")
        else:
            render_empty_result("📝", "Type a complaint and click<br><strong style='color:#00D4FF'>RUN AI PIPELINE</strong>")


# ────────────────────────────────────────────────────────────
#  TAB 2 — LIVE VOICE
# ────────────────────────────────────────────────────────────
with tab_voice:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("#### 🎙️ Record Your Complaint")
        st.markdown(f"""
        <div class="voice-card">
          <div class="voice-title">🎙️ Live Voice Recording</div>
          <div style="color:#B8D4F0;font-size:0.9rem;line-height:1.7;">
            Engine : <strong style="color:#7B2FBE">{stt_engine}</strong><br>
            Language : <strong style="color:#00D4FF">{voice_language}</strong><br>
            <span style="color:#6B8CAE;font-size:0.82rem;">
              Press the mic → speak → press stop → transcribe
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Try mic recorder ──────────────────────────────────
        try:
            from streamlit_mic_recorder import mic_recorder

            audio = mic_recorder(
                start_prompt        = "🎙️  Start Recording",
                stop_prompt         = "⏹️  Stop Recording",
                just_once           = False,
                use_container_width = True,
                key                 = "mic"
            )

            if audio and audio.get("bytes"):
                st.audio(audio["bytes"], format="audio/wav")
                st.success("✅ Audio captured! Click **Transcribe** below.")

                if st.button("🔤 TRANSCRIBE AUDIO", key="btn_transcribe_mic"):
                    with st.spinner(f"Transcribing with {stt_engine}..."):
                        try:
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                                f.write(audio["bytes"])
                                tmp = f.name

                            result = transcribe_audio_file(tmp, lang_code, stt_engine)
                            os.unlink(tmp)

                            # Store in session state
                            st.session_state["voice_transcript"] = result["transcript"]
                            st.session_state["voice_lang"]       = result.get("language", lang_code)
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Transcription failed: {e}")

        except ImportError:
            st.warning("""
**streamlit-mic-recorder not installed.**

Run in terminal:
```
pip install streamlit-mic-recorder
streamlit run app.py
```
            """)
            st.markdown("**👇 Or type manually to test the pipeline:**")

        # ── Show transcript & edit box ────────────────────────
        if st.session_state["voice_transcript"]:
            transcript = st.session_state["voice_transcript"]
            lang_d     = st.session_state["voice_lang"]

            # Show read-only transcript box
            st.markdown(f"""
            <div style="margin-top:14px;">
              <div class="transcript-label">
                Transcribed Text <span class="lang-badge">{lang_d.upper()}</span>
              </div>
              <div class="transcript-box">{transcript}</div>
            </div>
            """, unsafe_allow_html=True)

            # FIX: Use transcript hash as key so edit box ALWAYS resets
            # when a new transcript arrives — no stale value problem
            edit_key = f"voice_edit_{hash(transcript) % 99999}"
            edited = st.text_area(
                "✏️ Edit if Groq made mistakes:",
                value=transcript,
                height=100,
                key=edit_key,
                help="Groq transcript appears here. Fix any mistakes before analyzing."
            )

            col_analyze, col_clear = st.columns([3, 1])
            with col_analyze:
                if st.button("⚡ ANALYZE THIS COMPLAINT", key="btn_run_voice"):
                    st.session_state["voice_transcript"]   = edited
                    st.session_state["run_voice_pipeline"] = True
                    st.rerun()
            with col_clear:
                if st.button("🗑️ Clear", key="btn_clear_voice"):
                    st.session_state["voice_transcript"]   = ""
                    st.session_state["voice_lang"]         = lang_code
                    st.session_state["run_voice_pipeline"] = False
                    st.rerun()

        else:
            # Manual fallback input
            st.markdown("---")
            st.markdown("**Or type a complaint manually to test:**")
            manual = st.text_area(
                "Manual",
                placeholder="Type what you would say in Hindi or English...",
                height=100,
                label_visibility="collapsed",
                key="voice_manual"
            )
            if st.button("⚡ ANALYZE AS VOICE", key="btn_manual_voice"):
                if manual.strip():
                    st.session_state["voice_transcript"]   = manual
                    st.session_state["voice_lang"]         = lang_code
                    st.session_state["run_voice_pipeline"] = True
                    st.rerun()
                else:
                    st.warning("⚠️ Please enter something.")

    # ── RIGHT column — results ────────────────────────────────
    with col_r:
        st.markdown("#### 📊 Voice Pipeline Results")

        if st.session_state["run_voice_pipeline"] and st.session_state["voice_transcript"]:
            # Reset flag immediately so it doesn't loop
            st.session_state["run_voice_pipeline"] = False
            transcript = st.session_state["voice_transcript"]

            ph = st.empty()
            try:
                nr, sr, pr, cid, orig, trans = run_pipeline(transcript, population, location, ph)
                st.markdown("---")
                show_results(nr, sr, pr, cid, orig, trans)
            except Exception as e:
                st.error(f"❌ {e}")

        elif st.session_state["voice_transcript"]:
            # Transcript exists but pipeline not triggered — show waiting state
            render_empty_result("🎙️", "Transcript ready.<br>Click <strong style='color:#7B2FBE'>ANALYZE THIS COMPLAINT</strong>")
        else:
            render_empty_result("🎙️", "Record your complaint and click<br><strong style='color:#7B2FBE'>TRANSCRIBE AUDIO</strong>")


# ────────────────────────────────────────────────────────────
#  TAB 3 — UPLOAD AUDIO FILE
# ────────────────────────────────────────────────────────────
with tab_upload:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("#### 📁 Upload Audio File")
        st.markdown("""
        <div class="result-card" style="margin-bottom:14px;">
          <div class="result-card-title">Supported Formats</div>
          <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px;">
            <span style="background:#0A1628;border:1px solid #1A3A6B;padding:3px 10px;border-radius:8px;font-size:0.8rem;color:#00D4FF;">WAV</span>
            <span style="background:#0A1628;border:1px solid #1A3A6B;padding:3px 10px;border-radius:8px;font-size:0.8rem;color:#00D4FF;">MP3</span>
            <span style="background:#0A1628;border:1px solid #1A3A6B;padding:3px 10px;border-radius:8px;font-size:0.8rem;color:#00D4FF;">OGG</span>
            <span style="background:#0A1628;border:1px solid #1A3A6B;padding:3px 10px;border-radius:8px;font-size:0.8rem;color:#00D4FF;">M4A</span>
            <span style="background:#0A1628;border:1px solid #7B2FBE44;padding:3px 10px;border-radius:8px;font-size:0.8rem;color:#B47FE8;">WhatsApp OGG ✓</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop audio here",
            type=["wav","mp3","ogg","m4a","flac"],
            label_visibility="collapsed"
        )

        if uploaded:
            st.audio(uploaded)
            st.success(f"✅ **{uploaded.name}** ({uploaded.size // 1024} KB)")

            st.markdown(f"""
            <div class="result-card" style="margin-top:10px;">
              <div class="result-card-title">File Info</div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:6px;">
                <div><span style="color:#6B8CAE;font-size:0.8rem;">Engine</span><br>
                     <span style="color:#7B2FBE;font-size:0.85rem;">{stt_engine}</span></div>
                <div><span style="color:#6B8CAE;font-size:0.8rem;">Language</span><br>
                     <span style="color:#00D4FF;font-size:0.85rem;">{voice_language}</span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🔤 TRANSCRIBE FILE", key="btn_transcribe_upload"):
                with st.spinner(f"Transcribing with {stt_engine}..."):
                    try:
                        ext = "." + uploaded.name.split(".")[-1]
                        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                            f.write(uploaded.read())
                            tmp = f.name

                        result = transcribe_audio_file(tmp, lang_code, stt_engine)
                        os.unlink(tmp)

                        st.session_state["upload_transcript"]   = result["transcript"]
                        st.session_state["upload_lang"]         = result.get("language", lang_code)
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Transcription failed: {e}")

        # ── Show transcript ───────────────────────────────────
        if st.session_state["upload_transcript"]:
            transcript = st.session_state["upload_transcript"]
            lang_d     = st.session_state["upload_lang"]

            st.markdown(f"""
            <div style="margin-top:14px;">
              <div class="transcript-label">
                Transcript <span class="lang-badge">{lang_d.upper()}</span>
              </div>
              <div class="transcript-box">"{transcript}"</div>
            </div>
            """, unsafe_allow_html=True)

            # FIX: dynamic key so edit box always resets when new transcript arrives
            up_edit_key = f"upload_edit_{hash(transcript) % 99999}"
            edited_up = st.text_area(
                "✏️ Edit if Groq made mistakes:",
                value=transcript,
                height=100,
                key=up_edit_key,
                help="Fix any transcription errors before running the pipeline."
            )

            col_a, col_b = st.columns([3, 1])
            with col_a:
                if st.button("⚡ ANALYZE THIS COMPLAINT", key="btn_run_upload"):
                    st.session_state["upload_transcript"]   = edited_up
                    st.session_state["run_upload_pipeline"] = True
                    st.rerun()
            with col_b:
                if st.button("🗑️ Clear", key="btn_clear_upload"):
                    st.session_state["upload_transcript"]   = ""
                    st.session_state["run_upload_pipeline"] = False
                    st.rerun()

    # ── RIGHT column — results ────────────────────────────────
    with col_r:
        st.markdown("#### 📊 Upload Pipeline Results")

        if st.session_state["run_upload_pipeline"] and st.session_state["upload_transcript"]:
            st.session_state["run_upload_pipeline"] = False
            transcript = st.session_state["upload_transcript"]

            ph = st.empty()
            try:
                nr, sr, pr, cid, orig, trans = run_pipeline(transcript, population, location, ph)
                st.markdown("---")
                show_results(nr, sr, pr, cid, orig, trans)
            except Exception as e:
                st.error(f"❌ {e}")

        elif st.session_state["upload_transcript"]:
            render_empty_result("📁", "Transcript ready.<br>Click <strong style='color:#00D4FF'>ANALYZE THIS COMPLAINT</strong>")
        else:
            render_empty_result("📁", "Upload a file and click<br><strong style='color:#00D4FF'>TRANSCRIBE FILE</strong>")


# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#1A3A6B;font-size:0.8rem;padding:8px 0;font-family:'Space Mono',monospace;">
  JanAI Pipeline Tester &nbsp;·&nbsp; Text · Voice · Upload &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)