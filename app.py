"""
VoiceBot AI — Intelligent Conversational Agent
==============================================
YouTube-style voice conversational assistant:
  - Click to speak -> streams speech live on screen in real time.
  - Automatic silence detection (stops when you finish speaking, just like YouTube).
  - Directly delivers response without requiring a submit button.
  - Speaks answer aloud automatically via Web Speech Synthesis.
  - Seamless automatic rollback to trained BiLSTM model if cloud API fails.
"""

import os
import re
import time
import json
import pickle
import random
import requests
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
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

# Custom styling with vibrant animated background and glowing cyber orbs
st.markdown("""
<style>
    /* Full App Deep Canvas */
    .stApp {
        background-color: #030712 !important;
        overflow-x: hidden;
    }

    /* Ambient animated container */
    .animated-bg-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        overflow: hidden;
        z-index: 0;
        pointer-events: none;
    }

    /* Floating glowing neon orbs */
    .orb {
        position: absolute;
        border-radius: 50%;
        filter: blur(85px);
        opacity: 0.65;
        animation-timing-function: ease-in-out;
        animation-iteration-count: infinite;
        animation-direction: alternate;
    }
    .orb-1 {
        width: 520px;
        height: 520px;
        background: radial-gradient(circle, #2563eb 0%, #1e40af 60%, transparent 100%);
        top: -12%;
        left: -10%;
        animation: floatOrb1 14s infinite alternate;
    }
    .orb-2 {
        width: 560px;
        height: 560px;
        background: radial-gradient(circle, #8b5cf6 0%, #6d28d9 60%, transparent 100%);
        top: 20%;
        right: -15%;
        animation: floatOrb2 17s infinite alternate;
    }
    .orb-3 {
        width: 480px;
        height: 480px;
        background: radial-gradient(circle, #06b6d4 0%, #0e7490 60%, transparent 100%);
        bottom: -10%;
        left: 10%;
        animation: floatOrb3 15s infinite alternate;
    }
    .orb-4 {
        width: 420px;
        height: 420px;
        background: radial-gradient(circle, #ec4899 0%, #a21caf 60%, transparent 100%);
        top: 48%;
        left: 38%;
        opacity: 0.45;
        animation: floatOrb4 20s infinite alternate;
    }
    .orb-5 {
        width: 450px;
        height: 450px;
        background: radial-gradient(circle, #10b981 0%, #047857 60%, transparent 100%);
        bottom: 5%;
        right: 12%;
        opacity: 0.4;
        animation: floatOrb5 16s infinite alternate;
    }

    @keyframes floatOrb1 {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(160px, 90px) scale(1.18); }
        100% { transform: translate(80px, 170px) scale(0.92); }
    }
    @keyframes floatOrb2 {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(-140px, 90px) scale(1.22); }
        100% { transform: translate(-90px, -130px) scale(0.88); }
    }
    @keyframes floatOrb3 {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(120px, -100px) scale(1.15); }
        100% { transform: translate(-70px, -60px) scale(1.05); }
    }
    @keyframes floatOrb4 {
        0% { transform: translate(0, 0) scale(0.9); }
        50% { transform: translate(-100px, 120px) scale(1.2); }
        100% { transform: translate(80px, -90px) scale(0.95); }
    }
    @keyframes floatOrb5 {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(-120px, -110px) scale(1.18); }
        100% { transform: translate(60px, -70px) scale(0.88); }
    }

    /* Cyber grid overlay */
    .cyber-grid {
        position: absolute;
        inset: 0;
        background-image: 
            linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px);
        background-size: 55px 55px;
        mask-image: radial-gradient(circle at 50% 50%, black 45%, transparent 88%);
        pointer-events: none;
    }

    /* Floating glowing star particles */
    .particle {
        position: absolute;
        width: 4px;
        height: 4px;
        background: #38bdf8;
        border-radius: 50%;
        box-shadow: 0 0 14px 3px #38bdf8;
        opacity: 0.75;
        animation: floatParticle 8s infinite ease-in-out alternate;
    }
    .p1 { top: 18%; left: 14%; animation-duration: 9s; }
    .p2 { top: 32%; right: 18%; animation-duration: 11s; background: #c084fc; box-shadow: 0 0 14px 3px #c084fc; }
    .p3 { top: 62%; left: 22%; animation-duration: 13s; background: #34d399; box-shadow: 0 0 14px 3px #34d399; }
    .p4 { top: 78%; right: 28%; animation-duration: 10s; }
    .p5 { top: 12%; right: 32%; animation-duration: 14s; background: #f472b6; box-shadow: 0 0 14px 3px #f472b6; }
    .p6 { top: 48%; left: 8%; animation-duration: 12s; }

    @keyframes floatParticle {
        0% { transform: translateY(0px) translateX(0px); opacity: 0.35; }
        50% { transform: translateY(-45px) translateX(25px); opacity: 0.95; }
        100% { transform: translateY(15px) translateX(-20px); opacity: 0.45; }
    }

    /* Main container bounds */
    .main .block-container {
        max-width: 840px !important;
        padding-top: 1.4rem !important;
        padding-bottom: 2rem !important;
        position: relative;
        z-index: 1;
    }

    /* Shimmering Holographic Title */
    @keyframes titleShine {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .gradient-title {
        background: linear-gradient(90deg, #60a5fa, #c084fc, #34d399, #38bdf8, #60a5fa);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: titleShine 5s ease infinite;
    }

    /* Modern Glassmorphism Chat Bubbles */
    div[data-testid="stChatMessage"] {
        background: rgba(15, 23, 42, 0.72) !important;
        backdrop-filter: blur(18px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 18px !important;
        padding: 14px 20px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4) !important;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    div[data-testid="stChatMessage"]:hover {
        border-color: rgba(96, 165, 250, 0.45) !important;
        transform: translateY(-1px);
    }

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
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 12px;
    }

    /* Remove Streamlit default black bottom container & footer */
    [data-testid="stBottom"], [data-testid="stBottom"] > div, footer, header[data-testid="stHeader"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    footer {
        display: none !important;
        visibility: hidden !important;
    }
</style>

<!-- Animated Background Canvas Elements -->
<div class="animated-bg-container">
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
    <div class="orb orb-4"></div>
    <div class="orb orb-5"></div>
    <div class="cyber-grid"></div>
    <div class="particle p1"></div>
    <div class="particle p2"></div>
    <div class="particle p3"></div>
    <div class="particle p4"></div>
    <div class="particle p5"></div>
    <div class="particle p6"></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# SECRET & ENVIRONMENT HELPER
# ─────────────────────────────────────────────────────────

def get_secret(key_name: str) -> str:
    """Retrieve secret safely from .env, OS environment, or Streamlit secrets."""
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
# YOUTUBE-STYLE CUSTOM VOICE COMPONENT DECLARATION
# ─────────────────────────────────────────────────────────

voice_component_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_input")
voice_input_widget = components.declare_component("voice_input_widget", path=voice_component_dir)


# ─────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "auto_tts" not in st.session_state:
    st.session_state.auto_tts = True

if "speech_to_speak" not in st.session_state:
    st.session_state.speech_to_speak = ""

if "last_processed_speech" not in st.session_state:
    st.session_state.last_processed_speech = ""

if "last_processed_id" not in st.session_state:
    st.session_state.last_processed_id = ""

if "widget_counter" not in st.session_state:
    st.session_state.widget_counter = 0

if "cleared" not in st.session_state:
    st.session_state.cleared = False


# ─────────────────────────────────────────────────────────
# LOCAL DEEP LEARNING INFERENCE (BiLSTM)
# ─────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.45


def predict_bilstm(text: str, is_rollback: bool = False):
    """
    Classify user intent using the custom trained BiLSTM neural network.
    Acts as the primary offline model and the automated rollback engine.
    """
    cleaned = text.lower().strip()
    words = [w for w in cleaned.split() if len(w) > 1]
    stopwords = {
        "what", "is", "a", "an", "the", "tell", "me", "about", "how", "do",
        "does", "explain", "who", "which", "can", "you", "of", "in", "to", "for", "are"
    }
    content_words = [w for w in words if w not in stopwords]
    oov_content = [w for w in content_words if w not in tokenizer.word_index]

    # Out-of-Domain protection: if content words are mostly unlearned/unknown
    if content_words and (len(oov_content) / len(content_words) >= 0.6):
        if is_rollback:
            reply = (
                "I am currently operating in offline rollback mode (BiLSTM) because the cloud AI quota is temporarily refreshing. "
                "My local model is trained specifically on Computer Science and Artificial Intelligence questions. "
                "Please ask an AI/programming question, or try again in a few moments once the cloud service restores!"
            )
        else:
            reply = (
                "That topic appears to be outside my local training dataset. "
                "Please ask about Artificial Intelligence, Machine Learning, Deep Learning, Python, or related topics."
            )
        return {
            "reply": reply,
            "engine": "BiLSTM (Auto Rollback)" if is_rollback else "BiLSTM Neural Network",
            "intent": "out_of_domain",
            "confidence": 0.0,
        }

    seq = tokenizer.texts_to_sequences([cleaned])
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
# CLOUD LLM APIS (GEMINI / GROQ / OPENAI)
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are VoiceBot AI, an intelligent, highly accurate voice assistant. "
    "Keep your spoken answers concise (2 to 4 sentences max), strictly accurate and factual, and natural so they sound great when read aloud via text-to-speech. "
    "Never invent or hallucinate movie titles, names, dates, or false facts. "
    "Avoid markdown tables, asterisks, bullet lists, URLs, or complex ASCII formatting."
)


def try_gemini(user_text: str, api_key: str):
    """
    Attempt generation via Gemini API (cycling through supported flash/preview models with multi-turn memory).
    Returns None on failure so automatic rollback to BiLSTM seamlessly kicks in.
    """
    if not api_key:
        return None

    # Build multi-turn context
    contents = []
    if "messages" in st.session_state:
        for msg in st.session_state.messages[-6:]:
            role = "user" if msg.get("role") == "user" else "model"
            content = msg.get("content", "").strip()
            if content:
                contents.append({"role": role, "parts": [{"text": content}]})
    contents.append({"role": "user", "parts": [{"text": user_text}]})

    candidate_models = [
        "gemini-2.5-flash",
        "gemini-flash-latest",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
    ]

    for model_name in candidate_models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key.strip()}"
            payload = {
                "systemInstruction": {
                    "parts": [{"text": SYSTEM_PROMPT}]
                },
                "contents": contents,
                "generationConfig": {
                    "maxOutputTokens": 350,
                    "temperature": 0.3,
                    "thinkingConfig": {"thinkingBudget": 0}
                }
            }
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=8)
            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        content = parts[0]["text"].strip()
                        if content:
                            return {
                                "reply": content,
                                "engine": f"Cloud AI ({model_name.replace('-', ' ').title()})",
                                "intent": "Generative AI",
                                "confidence": 1.0
                            }
        except Exception:
            continue
    return None


def try_groq(user_text: str, api_key: str):
    """Attempt generation via Groq API (prioritizing GPT-OSS-120B / GPT-OSS-20B for maximum factual accuracy)."""
    if not api_key:
        return None

    groq_models = [
        ("openai/gpt-oss-120b", 550, 0.2),
        ("openai/gpt-oss-20b", 500, 0.2),
        ("qwen/qwen3.8-27b", 350, 0.2),
        ("allam-2-7b", 350, 0.3),
    ]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if "messages" in st.session_state:
        for m in st.session_state.messages[-4:]:
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_text})

    for model, max_tok, temp in groq_models:
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": max_tok,
            }
            res = requests.post(url, json=payload, headers=headers, timeout=8)
            if res.status_code == 200:
                raw = res.json()["choices"][0]["message"]["content"]
                if raw:
                    content = (
                        raw.replace("\u202f", " ")
                        .replace("**", "")
                        .replace("###", "")
                        .replace("##", "")
                        .strip()
                    )
                    if content:
                        model_display = model.split("/")[-1].replace("-", " ").title()
                        return {
                            "reply": content,
                            "engine": f"Cloud AI (Groq {model_display})",
                            "intent": "Generative AI",
                            "confidence": 1.0
                        }
        except Exception:
            continue
    return None


def try_openai(user_text: str, api_key: str):
    """Attempt generation via OpenAI API. Returns None on failure."""
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
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
      2. If cloud API key exists -> Try Groq (blazing fast) then Gemini or OpenAI.
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
        # 1. Prioritize Groq: ultra-fast (~0.8s) and high rate limits
        if groq_key:
            res = try_groq(user_text, groq_key)
            if res:
                return res

        # 2. Try Gemini
        if gemini_key:
            res = try_gemini(user_text, gemini_key)
            if res:
                return res

        # 3. Try OpenAI
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

    gemini_k = get_secret("GEMINI_API_KEY")
    groq_k = get_secret("GROQ_API_KEY")
    openai_k = get_secret("OPENAI_API_KEY")
    has_api = bool(gemini_k or groq_k or openai_k)

    if has_api:
        api_name = "Gemini 2.5 Flash" if gemini_k else ("Groq Llama 3.3" if groq_k else "OpenAI")
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

    with st.expander("📊 Lab Model Specifications"):
        st.markdown("""
        - **Architecture:** Bidirectional LSTM
        - **Intents:** 28 Categories
        - **Dataset V2:** 616 Utterances
        - **Held-out Test Acc:** **60.22%**
        - **Random Baseline:** 3.57% (1/28)
        - **Voice Engine:** Web Speech API (Auto-Silence)
        """)

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.speech_to_speak = ""
        st.session_state.widget_counter += 1
        st.session_state.cleared = True
        st.components.v1.html("<script>try { window.speechSynthesis.cancel(); if(window.parent && window.parent.speechSynthesis) window.parent.speechSynthesis.cancel(); } catch(e){}</script>", height=0)
        st.rerun()


# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────

st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 1px solid rgba(148, 163, 184, 0.15);">
    <div style="display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 2.2rem;">🎙️</span>
        <div>
            <h2 class="gradient-title" style="margin: 0; font-size: 1.75rem; font-weight: 800;">VoiceBot AI</h2>
            <p style="margin: 2px 0 0 0; color: #94a3b8; font-size: 0.85rem;">Speech Recognition & Deep Learning Conversational Agent</p>
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 6px; background: rgba(34, 197, 94, 0.12); border: 1px solid rgba(34, 197, 94, 0.3); padding: 4px 12px; border-radius: 20px;">
        <span style="width: 8px; height: 8px; border-radius: 50%; background: #22c55e; display: inline-block; box-shadow: 0 0 8px #22c55e;"></span>
        <span style="font-size: 0.76rem; color: #86efac; font-weight: 600;">Hands-Free Voice</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# BROWSER TEXT-TO-SPEECH HELPER
# ─────────────────────────────────────────────────────────

def trigger_browser_tts(text_to_speak: str):
    """Speaks the response aloud via Web Speech API in parallel with live caption streaming."""
    clean_text = (
        text_to_speak.replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace('"', '\\"')
        .replace("\n", " ")
    )
    tts_js = f"""
    <script>
        (function() {{
            var synth = window.speechSynthesis;
            try {{
                if (window.parent && window.parent.speechSynthesis) {{
                    synth = window.parent.speechSynthesis;
                }}
            }} catch(e) {{}}

            if (synth) {{
                synth.cancel();
                var utterance = new SpeechSynthesisUtterance("{clean_text}");
                utterance.rate = 1.0;
                utterance.pitch = 1.0;

                function setVoiceAndSpeak() {{
                    var voices = synth.getVoices();
                    var preferred = voices.find(function(v) {{
                        return v.name.includes('Google UK English Female') || 
                               v.name.includes('Natural') || 
                               v.name.includes('Samantha') || 
                               v.name.includes('Zira') ||
                               (v.lang && v.lang.startsWith('en'));
                    }});
                    if (preferred) utterance.voice = preferred;
                    synth.speak(utterance);
                }}

                if (synth.getVoices().length > 0) {{
                    setVoiceAndSpeak();
                }} else {{
                    synth.onvoiceschanged = setVoiceAndSpeak;
                }}
            }}
        }})();
    </script>
    """
    st.components.v1.html(tts_js, height=0)


# ─────────────────────────────────────────────────────────
# CONVERSATION CHAT CONTAINER & HISTORY
# ─────────────────────────────────────────────────────────

chat_container = st.container()

# Render unified bottom dock (Mic + Text Input + Stop button)
dock_container = st.container()
with dock_container:
    spoken_data = voice_input_widget(key=f"unified_input_bar_{st.session_state.widget_counter}")

# Guard against clearing
if st.session_state.get("cleared", False):
    st.session_state.cleared = False
    spoken_data = None

# Process submitted query (from either voice recognition or typed text)
if spoken_data:
    if isinstance(spoken_data, dict):
        user_query = str(spoken_data.get("text", "")).strip()
        query_id = str(spoken_data.get("ts", user_query))
    else:
        user_query = str(spoken_data).strip()
        query_id = user_query

    is_duplicate = (query_id == st.session_state.get("last_processed_id"))

    if user_query and not is_duplicate:
        st.session_state.last_processed_id = query_id

        with chat_container:
            # First render prior history
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant":
                        engine_name = msg.get("engine", "")
                        if "Rollback" in engine_name:
                            st.markdown("<span class='badge-fallback'>🛡️ Local Rollback (BiLSTM)</span>", unsafe_allow_html=True)
                        elif "BiLSTM" in engine_name and force_bilstm:
                            st.markdown("<span class='badge-bilstm'>🧠 BiLSTM</span>", unsafe_allow_html=True)

            # Render current user question
            with st.chat_message("user"):
                st.markdown(user_query)

            # Render assistant message with dynamic live caption streaming!
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    agent_data = get_agent_response(user_query, force_local=force_bilstm)

                reply_text = agent_data["reply"]

                # Trigger browser TTS immediately so words stream in sync with spoken voice
                if st.session_state.auto_tts:
                    trigger_browser_tts(reply_text)

                # Live caption streaming generator: parallelized to voice speech timing
                def stream_live_captions():
                    if st.session_state.auto_tts:
                        # Synchronize with Web Speech API audio initialization
                        time.sleep(0.38)

                    tokens = re.split(r'(\s+)', reply_text)
                    for token in tokens:
                        if token:
                            yield token
                            if not token.isspace():
                                if st.session_state.auto_tts:
                                    # Paced in direct parallel to vocal delivery (~170 WPM)
                                    word_clean = token.strip()
                                    delay = 0.15 + (len(word_clean) * 0.016)
                                    if word_clean.endswith((',', ';', ':')):
                                        delay += 0.16
                                    elif word_clean.endswith(('.', '!', '?')):
                                        delay += 0.28
                                    time.sleep(delay)
                                else:
                                    time.sleep(0.02)

                st.write_stream(stream_live_captions)

                engine_name = agent_data.get("engine", "")
                if "Rollback" in engine_name:
                    st.markdown("<span class='badge-fallback'>🛡️ Local Rollback (BiLSTM)</span>", unsafe_allow_html=True)
                elif "BiLSTM" in engine_name and force_bilstm:
                    st.markdown("<span class='badge-bilstm'>🧠 BiLSTM</span>", unsafe_allow_html=True)

        # Save to session state
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({
            "role": "assistant",
            "content": reply_text,
            "engine": agent_data.get("engine", "BiLSTM"),
            "intent": agent_data.get("intent", ""),
            "confidence": agent_data.get("confidence", 1.0),
        })

else:
    with chat_container:
        if len(st.session_state.messages) == 0:
            st.markdown("""
            <div style="text-align: center; padding: 45px 20px; background: rgba(15, 23, 42, 0.45); border-radius: 16px; border: 1px dashed rgba(148, 163, 184, 0.2); margin: 25px 0;">
                <div style="font-size: 2.8rem; margin-bottom: 10px;">🎙️</div>
                <h3 style="margin: 0 0 6px 0; color: #f1f5f9; font-weight: 700;">Ready to Chat</h3>
                <p style="margin: 0; color: #94a3b8; font-size: 0.9rem;">Tap the microphone below to speak naturally, or type your question in the text box.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant":
                        engine_name = msg.get("engine", "")
                        if "Rollback" in engine_name:
                            st.markdown("<span class='badge-fallback'>🛡️ Local Rollback (BiLSTM)</span>", unsafe_allow_html=True)
                        elif "BiLSTM" in engine_name and force_bilstm:
                            st.markdown("<span class='badge-bilstm'>🧠 BiLSTM</span>", unsafe_allow_html=True)
