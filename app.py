"""
VoiceBot AI — Intelligent Conversational Agent
==============================================
Multimodal voice & text chatbot featuring:
  - Real-time Speech-to-Text with live on-screen interim subtitles
  - Automatic Text-to-Speech (TTS) via Web Speech Synthesis
  - Automatic API fallback & rollback: uses Cloud LLM if available,
    and automatically/silently shifts to the local BiLSTM deep learning
    model if the API key is invalid, missing, rate-limited, or fails.
  - Zero friction: never prompts the end-user for API keys.
"""

import os
import json
import pickle
import random
import requests
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load local .env if available
load_dotenv()

# ─────────────────────────────────────────────────────────
# PAGE CONFIGURATION & STYLING
# ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VoiceBot AI — Conversational Agent",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .badge-bilstm {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.4);
        padding: 3px 10px;
        border-radius: 16px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-llm {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 3px 10px;
        border-radius: 16px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-fallback {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 3px 10px;
        border-radius: 16px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# SECRET & ENVIRONMENT HELPER
# ─────────────────────────────────────────────────────────

def get_secret(key_name: str) -> str:
    """Retrieve secret from .env, environment variable, or Streamlit secrets."""
    val = os.getenv(key_name, "")
    if not val:
        try:
            if hasattr(st, "secrets") and key_name in st.secrets:
                val = st.secrets[key_name]
        except Exception:
            pass
    return str(val).strip() if val else ""


# ─────────────────────────────────────────────────────────
# LOAD LOCAL DEEP LEARNING MODEL & ARTIFACTS
# ─────────────────────────────────────────────────────────

@st.cache_resource
def load_bilstm_pipeline():
    """Load the trained BiLSTM model and tokenization objects."""
    model = load_model("chatbot_model.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    with open("intents.json", "r", encoding="utf-8") as f:
        intents_data = json.load(f)

    responses_lookup = {}
    for intent in intents_data["intents"]:
        responses_lookup[intent["tag"]] = intent["responses"]

    return model, tokenizer, label_encoder, metadata, responses_lookup


bilstm_model, tokenizer, label_encoder, metadata, intent_responses = load_bilstm_pipeline()
max_len = metadata["max_len"]


# ─────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "auto_tts" not in st.session_state:
    st.session_state.auto_tts = True

if "speech_to_speak" not in st.session_state:
    st.session_state.speech_to_speak = ""


# ─────────────────────────────────────────────────────────
# LOCAL DEEP LEARNING INFERENCE (BiLSTM)
# ─────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.45


def predict_bilstm(text: str, is_rollback: bool = False):
    """
    Classify user intent using the custom trained BiLSTM neural network.
    Acts as the primary offline model and the automated rollback engine.
    """
    seq = tokenizer.texts_to_sequences([text.lower().strip()])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    probabilities = bilstm_model.predict(padded, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    intent_tag = str(label_encoder.inverse_transform([predicted_index])[0])

    if confidence < CONFIDENCE_THRESHOLD:
        reply = (
            "I'm not completely sure what you mean. "
            "Could you please rephrase your question?"
        )
    else:
        candidates = intent_responses.get(intent_tag, ["I don't have a response for that."])
        reply = random.choice(candidates)

    engine_label = "BiLSTM (Auto Rollback)" if is_rollback else "BiLSTM Neural Network"

    return {
        "reply": reply,
        "engine": engine_label,
        "intent": intent_tag,
        "confidence": confidence,
    }


# ─────────────────────────────────────────────────────────
# CLOUD LLM APIS (OPTIONAL ENHANCEMENT)
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are VoiceBot AI, an intelligent, conversational AI assistant modeled after a friendly expert interviewer and tutor. "
    "Keep your spoken answers concise (2 to 4 sentences), accurate, natural, and conversational so they sound great when read aloud via text-to-speech. "
    "Avoid long markdown bullet lists, URLs, or complex ASCII formatting."
)


def try_groq(user_text: str, api_key: str):
    """Attempt generation via Groq API (Llama 3.3). Returns None on any failure."""
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in st.session_state.messages[-4:]:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": user_text})

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 250,
        }
        res = requests.post(url, json=payload, headers=headers, timeout=8)
        if res.status_code == 200:
            content = res.json()["choices"][0]["message"]["content"].strip()
            if content:
                return {
                    "reply": content,
                    "engine": "Cloud AI (Groq Llama 3.3)",
                    "intent": "Generative AI",
                    "confidence": 1.0
                }
    except Exception:
        pass
    return None


def try_gemini(user_text: str, api_key: str):
    """Attempt generation via Gemini API (targeting gemini-2.5-flash). Returns None on failure."""
    for model_name in ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"]:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key.strip()}"
            prompt = SYSTEM_PROMPT + f"\n\nUser: {user_text}\nAssistant: "
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 250, "temperature": 0.7}
            }
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=15)
            if res.status_code == 200:
                data = res.json()
                content = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                if content:
                    return {
                        "reply": content,
                        "engine": f"Cloud AI (Gemini 2.5 Flash)",
                        "intent": "Generative AI",
                        "confidence": 1.0
                    }
        except Exception:
            continue
    return None


def try_openai(user_text: str, api_key: str):
    """Attempt generation via OpenAI API. Returns None on any failure."""
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ]
        payload = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 250,
        }
        res = requests.post(url, json=payload, headers=headers, timeout=8)
        if res.status_code == 200:
            content = res.json()["choices"][0]["message"]["content"].strip()
            if content:
                return {
                    "reply": content,
                    "engine": "Cloud AI (OpenAI GPT-4o)",
                    "intent": "Generative AI",
                    "confidence": 1.0
                }
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────
# RESILIENT RESPONSE ROUTER (ZERO-FRICTION ROLLBACK)
# ─────────────────────────────────────────────────────────

def get_agent_response(user_text: str, force_local: bool = False):
    """
    Intelligent router with automatic rollback:
      1. If user forces local mode -> Use BiLSTM immediately.
      2. If cloud API key exists -> Try Cloud LLM (Gemini / Groq / OpenAI).
      3. If Cloud API fails or is unavailable -> Automatically and silently
         shift to local BiLSTM deep learning model as fallback.
    """
    if force_local:
        return predict_bilstm(user_text, is_rollback=False)

    gemini_key = get_secret("GEMINI_API_KEY")
    groq_key = get_secret("GROQ_API_KEY")
    openai_key = get_secret("OPENAI_API_KEY")

    has_cloud_key = bool(gemini_key or groq_key or openai_key)

    if has_cloud_key:
        if gemini_key:
            res = try_gemini(user_text, gemini_key)
            if res:
                return res

        if groq_key:
            res = try_groq(user_text, groq_key)
            if res:
                return res

        if openai_key:
            res = try_openai(user_text, openai_key)
            if res:
                return res

        # Automatic Rollback: Cloud API failed or rate-limited, shift to local BiLSTM
        return predict_bilstm(user_text, is_rollback=True)

    # Default: No cloud key present, use local BiLSTM directly
    return predict_bilstm(user_text, is_rollback=False)


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ System Status")

    # Detect background availability of keys
    groq_k = get_secret("GROQ_API_KEY")
    gemini_k = get_secret("GEMINI_API_KEY")
    openai_k = get_secret("OPENAI_API_KEY")
    has_api = bool(groq_k or gemini_k or openai_k)

    # Show active engine info without asking user for any input
    if has_api:
        api_name = "Groq (Llama 3.3)" if groq_k else ("Gemini" if gemini_k else "OpenAI")
        st.markdown(f"""
        <div class="status-card">
            <span style="color: #34d399; font-weight: 700;">🟢 Active Engine:</span><br>
            <span style="font-size: 0.9rem; color: #f1f5f9;">Cloud AI ({api_name})</span><br>
            <span style="color: #60a5fa; font-size: 0.8rem;">🛡️ Auto-Rollback: <b>BiLSTM Ready</b></span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card">
            <span style="color: #60a5fa; font-weight: 700;">🧠 Active Engine:</span><br>
            <span style="font-size: 0.9rem; color: #f1f5f9;">Local BiLSTM Neural Network</span><br>
            <span style="color: #94a3b8; font-size: 0.8rem;">Self-contained offline model</span>
        </div>
        """, unsafe_allow_html=True)

    force_bilstm = st.toggle(
        "🧠 Force Local BiLSTM (Lab Mode)",
        value=False,
        help="Enable to test exclusively with the custom trained 28-intent BiLSTM deep learning model."
    )

    st.session_state.auto_tts = st.toggle(
        "🔊 Auto Speak Responses (TTS)",
        value=st.session_state.auto_tts,
        help="When enabled, the browser will automatically speak the chatbot's answers aloud."
    )

    st.divider()

    st.subheader("📊 Model Specifications")
    st.markdown("""
    - **Architecture:** Bidirectional LSTM
    - **Intents:** 28 Categories
    - **Dataset V2:** 616 Utterances
    - **Held-out Test Acc:** **60.22%**
    - **Random Baseline:** 3.57% (1/28)
    - **Rollback System:** Built-in automatic failover
    """)

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.speech_to_speak = ""
        st.rerun()


# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────

st.title("🎙️ VoiceBot AI")
st.markdown(
    "**Conversational Voice Agent** powered by an **end-to-end Deep Learning BiLSTM model** "
    "with automatic cloud enhancement and live speech streaming. "
    "Click the microphone to speak, watch your speech appear live on screen, and listen to the response!"
)


# ─────────────────────────────────────────────────────────
# LIVE SPEECH RECOGNITION (WEB SPEECH API WITH INTERIM RESULTS)
# ─────────────────────────────────────────────────────────

# Process speech submitted from the live JavaScript component
incoming_speech = st.query_params.get("speech", "")
if incoming_speech:
    st.query_params.clear()
    user_query = incoming_speech.strip()
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.spinner("Processing speech & analyzing intent..."):
            agent_data = get_agent_response(user_query, force_local=force_bilstm)
        st.session_state.messages.append({
            "role": "assistant",
            "content": agent_data["reply"],
            "engine": agent_data.get("engine", "BiLSTM"),
            "intent": agent_data.get("intent", ""),
            "confidence": agent_data.get("confidence", 1.0),
        })
        if st.session_state.auto_tts:
            st.session_state.speech_to_speak = agent_data["reply"]
        st.rerun()


# Live Voice UI Component (SpeechRecognition with live subtitles)
live_voice_html = """
<div style="background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #3b82f6; border-radius: 14px; padding: 18px; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <button id="micBtn" onclick="toggleRecognition()" style="background: #2563eb; color: white; border: none; padding: 10px 18px; border-radius: 25px; font-size: 15px; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 8px; transition: all 0.2s ease;">
                <span id="micIcon">🎤</span> <span id="micLabel">Start Speaking</span>
            </button>
            <span id="statusIndicator" style="font-size: 13px; color: #94a3b8; font-weight: 500;">Ready — Click microphone to speak</span>
        </div>
        <button id="sendBtn" onclick="submitTranscript()" style="background: #10b981; color: white; border: none; padding: 8px 16px; border-radius: 20px; font-size: 13px; font-weight: 600; cursor: pointer; display: none;">
            🚀 Send Now
        </button>
    </div>

    <!-- Live Subtitle Display Box: Shows words in real-time as spoken -->
    <div style="background: rgba(15, 23, 42, 0.7); border: 1px dashed #64748b; border-radius: 10px; padding: 14px; min-height: 52px; display: flex; align-items: center;">
        <span style="color: #64748b; font-size: 12px; margin-right: 8px; font-weight: 700;">LIVE SPEECH:</span>
        <span id="liveTranscript" style="color: #38bdf8; font-size: 15px; font-weight: 500; font-style: italic;">
            (Click the microphone above and speak; your words will appear here in real-time)
        </span>
    </div>
</div>

<script>
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition = null;
    let isListening = false;
    let finalTranscript = '';
    let silenceTimer = null;

    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onstart = function() {
            isListening = true;
            document.getElementById('micBtn').style.background = '#ef4444';
            document.getElementById('micBtn').style.boxShadow = '0 0 15px rgba(239, 68, 68, 0.6)';
            document.getElementById('micIcon').innerText = '⏹️';
            document.getElementById('micLabel').innerText = 'Stop Recording';
            document.getElementById('statusIndicator').innerText = '🔴 Listening... Speak naturally';
            document.getElementById('statusIndicator').style.color = '#ef4444';
            document.getElementById('sendBtn').style.display = 'inline-block';
            document.getElementById('liveTranscript').innerText = '';
            finalTranscript = '';
        };

        recognition.onresult = function(event) {
            let interimTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript + ' ';
                } else {
                    interimTranscript += event.results[i][0].transcript;
                }
            }
            const fullDisplay = (finalTranscript + interimTranscript).trim();
            if (fullDisplay) {
                document.getElementById('liveTranscript').innerText = '"' + fullDisplay + '"';
                document.getElementById('liveTranscript').style.fontStyle = 'normal';
                document.getElementById('liveTranscript').style.color = '#38bdf8';
            }

            // Auto-submit after 2.5 seconds of silence
            clearTimeout(silenceTimer);
            silenceTimer = setTimeout(function() {
                if (isListening && fullDisplay.length > 2) {
                    submitTranscript();
                }
            }, 2500);
        };

        recognition.onerror = function(event) {
            console.warn('Speech recognition error:', event.error);
            document.getElementById('statusIndicator').innerText = 'Notice: ' + event.error;
            stopListening();
        };

        recognition.onend = function() {
            stopListening();
        };
    } else {
        document.getElementById('statusIndicator').innerText = 'Browser Web Speech API not supported. Please use Chrome/Edge or the text input below.';
    }

    function toggleRecognition() {
        if (!recognition) {
            alert('Your browser does not support the Web Speech API. Please use Google Chrome or Microsoft Edge.');
            return;
        }
        if (isListening) {
            submitTranscript();
        } else {
            try {
                recognition.start();
            } catch (e) {
                console.log(e);
            }
        }
    }

    function stopListening() {
        isListening = false;
        clearTimeout(silenceTimer);
        document.getElementById('micBtn').style.background = '#2563eb';
        document.getElementById('micBtn').style.boxShadow = 'none';
        document.getElementById('micIcon').innerText = '🎤';
        document.getElementById('micLabel').innerText = 'Start Speaking';
        document.getElementById('statusIndicator').innerText = 'Ready — Click microphone to speak';
        document.getElementById('statusIndicator').style.color = '#94a3b8';
    }

    function submitTranscript() {
        if (recognition) {
            try { recognition.stop(); } catch(e) {}
        }
        stopListening();
        const textToSend = finalTranscript.trim() || document.getElementById('liveTranscript').innerText.replace(/^"|"$/g, '').trim();
        if (textToSend && textToSend !== '(Click the microphone above and speak; your words will appear here in real-time)') {
            document.getElementById('statusIndicator').innerText = 'Submitting query...';
            window.parent.location.search = '?speech=' + encodeURIComponent(textToSend);
        }
    }
</script>
"""

st.components.v1.html(live_voice_html, height=140)


# ─────────────────────────────────────────────────────────
# CONVERSATION CHAT HISTORY
# ─────────────────────────────────────────────────────────

if len(st.session_state.messages) == 0:
    st.info("💡 **Welcome!** Speak into the microphone above or type a question below to start the conversation.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Display engine and intent badge
            if msg["role"] == "assistant":
                meta_cols = st.columns([1, 1, 3])
                engine_name = msg.get("engine", "BiLSTM")
                intent_name = msg.get("intent", "")
                conf_val = msg.get("confidence", 1.0)

                with meta_cols[0]:
                    if "Rollback" in engine_name:
                        st.markdown("<span class='badge-fallback'>🛡️ Local Rollback (BiLSTM)</span>", unsafe_allow_html=True)
                    elif "BiLSTM" in engine_name:
                        st.markdown("<span class='badge-bilstm'>🧠 BiLSTM</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span class='badge-llm'>⚡ {engine_name}</span>", unsafe_allow_html=True)

                with meta_cols[1]:
                    if intent_name and "BiLSTM" in engine_name:
                        st.caption(f"Intent: `{intent_name}` ({conf_val*100:.1f}%)")


# ─────────────────────────────────────────────────────────
# TEXT INPUT (STREAMLIT CHAT INPUT)
# ─────────────────────────────────────────────────────────

typed_input = st.chat_input("Type your question here (or speak using the microphone above)...")

if typed_input:
    user_query = typed_input.strip()
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.spinner("Analyzing question..."):
        agent_data = get_agent_response(user_query, force_local=force_bilstm)
    st.session_state.messages.append({
        "role": "assistant",
        "content": agent_data["reply"],
        "engine": agent_data.get("engine", "BiLSTM"),
        "intent": agent_data.get("intent", ""),
        "confidence": agent_data.get("confidence", 1.0),
    })
    if st.session_state.auto_tts:
        st.session_state.speech_to_speak = agent_data["reply"]
    st.rerun()


# ─────────────────────────────────────────────────────────
# AUTOMATIC TEXT-TO-SPEECH (TTS VIA WEB SPEECH SYNTHESIS)
# ─────────────────────────────────────────────────────────

if st.session_state.speech_to_speak and st.session_state.auto_tts:
    text_to_speak = (
        st.session_state.speech_to_speak
        .replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace('"', '\\"')
        .replace("\n", " ")
    )
    # Reset state so it speaks only once per response
    st.session_state.speech_to_speak = ""

    tts_js = f"""
    <script>
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            const text = "{text_to_speak}";
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;

            function setVoiceAndSpeak() {{
                const voices = window.speechSynthesis.getVoices();
                const preferred = voices.find(v => 
                    v.name.includes('Google UK English Female') || 
                    v.name.includes('Natural') || 
                    v.name.includes('Samantha') || 
                    v.name.includes('Zira') ||
                    (v.lang && v.lang.startsWith('en'))
                );
                if (preferred) utterance.voice = preferred;
                window.speechSynthesis.speak(utterance);
            }}

            if (window.speechSynthesis.getVoices().length > 0) {{
                setVoiceAndSpeak();
            }} else {{
                window.speechSynthesis.onvoiceschanged = setVoiceAndSpeak;
            }}
        }}
    </script>
    """
    st.components.v1.html(tts_js, height=0)


# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────

st.markdown("---")
st.caption("🎙️ VoiceBot AI • Powered by Bidirectional LSTM Neural Network with Automated Cloud Rollback • Web Speech Recognition & Synthesis")
