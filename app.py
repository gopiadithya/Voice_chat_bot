"""
VoiceBot AI — Intelligent Conversational Agent
==============================================
Multimodal voice & text chatbot supporting:
  - Real-time Speech-to-Text with live on-screen interim transcription
  - Automatic Text-to-Speech (TTS) via Web Speech Synthesis
  - Generative AI responses via LLM APIs (Groq / Gemini / OpenAI) loaded from .env
  - Offline Deep Learning intent classification via trained BiLSTM model (28 intents)
  - Interactive chat history with confidence scores & threshold fallbacks
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
from streamlit_mic_recorder import speech_to_text

# Load environment variables from .env
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

# Custom CSS for glowing voice waves and modern chat styling
st.markdown("""
<style>
    .live-voice-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    .badge-bilstm {
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid #3b82f6;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-llm {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #10b981;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


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
# PREDICTION & GENERATION FUNCTIONS
# ─────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.45


def predict_bilstm(text: str):
    """Classify user intent using the custom trained BiLSTM neural network."""
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

    return {
        "reply": reply,
        "engine": "BiLSTM",
        "intent": intent_tag,
        "confidence": confidence,
    }


def generate_llm(messages: list, provider: str, api_key: str):
    """Generate dynamic conversational reply using a cloud LLM."""
    system_prompt = (
        "You are VoiceBot AI, an intelligent, conversational AI assistant modeled after a friendly expert interviewer and tutor. "
        "Keep your spoken answers concise (2 to 4 sentences), accurate, natural, and conversational so they sound great when read aloud via text-to-speech. "
        "Avoid long markdown bullet lists, URLs, or complex ASCII formatting."
    )

    formatted_messages = [{"role": "system", "content": system_prompt}]
    # Add last 6 turns of conversation for context
    for msg in messages[-6:]:
        formatted_messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        if provider == "Groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": formatted_messages,
                "temperature": 0.7,
                "max_tokens": 250,
            }
            res = requests.post(url, json=payload, headers=headers, timeout=12)
            if res.status_code == 200:
                content = res.json()["choices"][0]["message"]["content"].strip()
                return {"reply": content, "engine": "Groq (Llama 3.3 70B)", "intent": "Generative AI", "confidence": 1.0}
            else:
                st.warning(f"Groq API error ({res.status_code}): {res.text}")

        elif provider == "Google Gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key.strip()}"
            prompt_text = system_prompt + "\n\n"
            for m in messages[-4:]:
                prompt_text += f"{m['role'].capitalize()}: {m['content']}\n"
            prompt_text += "Assistant: "

            payload = {
                "contents": [{"parts": [{"text": prompt_text}]}],
                "generationConfig": {"maxOutputTokens": 250, "temperature": 0.7}
            }
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=12)
            if res.status_code == 200:
                data = res.json()
                content = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                return {"reply": content, "engine": "Google Gemini 1.5 Flash", "intent": "Generative AI", "confidence": 1.0}
            else:
                st.warning(f"Gemini API error ({res.status_code}): {res.text}")

        elif provider == "OpenAI":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": formatted_messages,
                "temperature": 0.7,
                "max_tokens": 250,
            }
            res = requests.post(url, json=payload, headers=headers, timeout=12)
            if res.status_code == 200:
                content = res.json()["choices"][0]["message"]["content"].strip()
                return {"reply": content, "engine": "OpenAI (GPT-4o mini)", "intent": "Generative AI", "confidence": 1.0}
            else:
                st.warning(f"OpenAI API error ({res.status_code}): {res.text}")

    except Exception as e:
        st.warning(f"LLM API request failed: {e}. Falling back to BiLSTM.")

    # Fallback to local BiLSTM if API call fails
    return predict_bilstm(messages[-1]["content"])


def get_agent_response(user_text: str, engine_choice: str, api_key: str):
    """Route input to the appropriate intelligence engine."""
    # Temporarily append user message to build context for LLM
    temp_messages = list(st.session_state.messages) + [{"role": "user", "content": user_text}]

    if engine_choice == "BiLSTM (Offline Deep Learning)" or not api_key:
        return predict_bilstm(user_text)
    else:
        return generate_llm(temp_messages, engine_choice, api_key)


# ─────────────────────────────────────────────────────────
# SIDEBAR CONFIGURATION
# ─────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    # API Keys from environment
    env_groq = os.getenv("GROQ_API_KEY", "")
    env_gemini = os.getenv("GEMINI_API_KEY", "")
    env_openai = os.getenv("OPENAI_API_KEY", "")

    # Engine selection
    engine_options = [
        "BiLSTM (Offline Deep Learning)",
        "Groq",
        "Google Gemini",
        "OpenAI",
    ]

    default_index = 0
    if env_groq:
        default_index = 1
    elif env_gemini:
        default_index = 2
    elif env_openai:
        default_index = 3

    selected_engine = st.selectbox(
        "🧠 Response Engine",
        options=engine_options,
        index=default_index,
        help="Select whether to use the custom BiLSTM deep learning model or a generative LLM API."
    )

    active_api_key = ""
    if selected_engine == "Groq":
        active_api_key = st.text_input(
            "Groq API Key",
            value=env_groq,
            type="password",
            help="Free key from console.groq.com. Powers ultra-fast Llama 3.3."
        )
    elif selected_engine == "Google Gemini":
        active_api_key = st.text_input(
            "Gemini API Key",
            value=env_gemini,
            type="password",
            help="From aistudio.google.com"
        )
    elif selected_engine == "OpenAI":
        active_api_key = st.text_input(
            "OpenAI API Key",
            value=env_openai,
            type="password",
            help="From platform.openai.com"
        )

    st.session_state.auto_tts = st.toggle(
        "🔊 Auto Speak Responses (TTS)",
        value=st.session_state.auto_tts,
        help="When enabled, the browser will automatically speak the chatbot's answers aloud."
    )

    st.divider()

    st.subheader("📊 Model Architecture")
    st.markdown("""
    - **Model:** Bidirectional LSTM
    - **Intents:** 28 Categories
    - **Dataset V2:** 616 Utterances
    - **Held-out Test Acc:** **60.22%**
    - **Random Baseline:** 3.57% (1/28)
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
    "**Conversational Voice Agent** powered by a **Bidirectional LSTM** and real-time **Speech-to-Text & Voice Synthesis**. "
    "Click the microphone to speak, watch your speech appear live on screen, and listen to the agent respond!"
)


# ─────────────────────────────────────────────────────────
# LIVE SPEECH RECOGNITION (WEB SPEECH API WITH INTERIM RESULTS)
# ─────────────────────────────────────────────────────────

# Check if query params received speech from the live component
incoming_speech = st.query_params.get("speech", "")
if incoming_speech:
    st.query_params.clear()
    user_query = incoming_speech.strip()
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.spinner("Analyzing intent & preparing response..."):
            agent_data = get_agent_response(user_query, selected_engine, active_api_key)
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


# Render the interactive Web Speech API Console with LIVE on-screen subtitles
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
            (Click the microphone above and start speaking; your speech will appear here in real-time)
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
        if (textToSend && textToSend !== '(Click the microphone above and start speaking; your speech will appear here in real-time)') {
            document.getElementById('statusIndicator').innerText = 'Submitting query...';
            // Send to parent Streamlit via query param
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
            # Display model metadata badge if assistant
            if msg["role"] == "assistant":
                meta_cols = st.columns([1, 1, 3])
                engine_name = msg.get("engine", "BiLSTM")
                intent_name = msg.get("intent", "")
                conf_val = msg.get("confidence", 1.0)

                with meta_cols[0]:
                    if "BiLSTM" in engine_name:
                        st.markdown(f"<span class='badge-bilstm'>🧠 BiLSTM</span>", unsafe_allow_html=True)
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
        agent_data = get_agent_response(user_query, selected_engine, active_api_key)
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

            // Pick natural English voice if available
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
st.caption("🎙️ VoiceBot AI • Powered by Bidirectional LSTM Neural Network & LLM Generative AI • Web Speech Recognition & Synthesis")
