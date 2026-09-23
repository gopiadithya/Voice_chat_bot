"""
✨ Nova AI — Conversational Voice Assistant
============================================
An intelligent, multimodal voice chatbot powered by a Bidirectional LSTM
neural network for intent classification across 28 technical domains.
"""

import json
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from streamlit_mic_recorder import speech_to_text

# ─────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Nova AI — Voice Assistant",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CUSTOM STYLING (Nova AI Conversational Theme)
# ─────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Main header styling */
    .nova-header {
        text-align: center;
        padding: 1.2rem 0 0.5rem 0;
    }
    .nova-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .nova-subtitle {
        font-size: 0.95rem;
        color: #888888;
        margin-bottom: 1.2rem;
    }
    .nova-status {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        background-color: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    /* Quick chip buttons */
    .chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1.2rem;
    }
    /* Intent badge */
    .intent-badge {
        display: inline-block;
        font-size: 0.75rem;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        background-color: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        font-weight: 600;
        margin-top: 0.4rem;
        margin-right: 0.5rem;
    }
    .confidence-badge {
        display: inline-block;
        font-size: 0.75rem;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        background-color: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        font-weight: 600;
        margin-top: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────
# LOAD MODEL & ARTIFACTS (Cached)
# ─────────────────────────────────────────────────────────

@st.cache_resource
def load_nova_brain():
    model = load_model("chatbot_model.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    with open("intents.json", "r", encoding="utf-8") as f:
        intents_data = json.load(f)

    return model, tokenizer, label_encoder, metadata, intents_data


model, tokenizer, label_encoder, metadata, intents_data = load_nova_brain()
max_len = metadata["max_len"]

responses = {}
for intent in intents_data["intents"]:
    responses[intent["tag"]] = intent["responses"]

# ─────────────────────────────────────────────────────────
# SESSION STATE MANAGEMENT
# ─────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **Hello! I'm Nova AI**, your deep-learning conversational voice assistant. You can speak into your microphone or message me about Machine Learning, Neural Networks, Transformers, Python, DSA, or study advice!",
            "intent": "greeting",
            "confidence": 1.0,
        }
    ]

if "last_spoken_id" not in st.session_state:
    st.session_state.last_spoken_id = None

# ─────────────────────────────────────────────────────────
# PREDICTION & RESPONSE ENGINE
# ─────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.45


def predict_intent(text: str):
    seq = tokenizer.texts_to_sequences([text.lower().strip()])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

    probabilities = model.predict(padded, verbose=0)[0]
    predicted_index = np.argmax(probabilities)
    confidence = float(probabilities[predicted_index])
    intent = label_encoder.inverse_transform([predicted_index])[0]

    return intent, confidence


def generate_response(text: str):
    intent, confidence = predict_intent(text)

    if confidence < CONFIDENCE_THRESHOLD:
        response = (
            "I'm not completely sure about that. "
            "Could you rephrase your question or ask about an AI/CS topic?"
        )
    else:
        candidates = responses.get(
            intent,
            ["I understand what you mean, but I don't have a direct answer prepared yet."]
        )
        response = str(np.random.choice(candidates))

    return intent, confidence, response


# ─────────────────────────────────────────────────────────
# JAVASCRIPT TEXT-TO-SPEECH (Browser Native SpeechSynthesis)
# ─────────────────────────────────────────────────────────

def trigger_speech(text: str):
    safe_text = (
        text.replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("\n", " ")
        .replace('"', '\\"')
        .replace("*", "")
    )
    st.components.v1.html(
        f"""
        <script>
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                const speech = new SpeechSynthesisUtterance("{safe_text}");
                speech.rate = 1.02;
                speech.pitch = 1.0;
                speech.volume = 1.0;
                window.speechSynthesis.speak(speech);
            }}
        </script>
        """,
        height=0,
    )


# ─────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/artificial-intelligence.png", width=90)
    st.title("✨ Nova AI")
    st.caption("Deep Learning Conversational Assistant")

    st.markdown("---")
    st.subheader("🎙️ Voice & Audio Settings")
    auto_speak = st.toggle("🔊 Auto-speak responses", value=True, help="Nova will speak responses aloud using browser speech synthesis.")

    st.markdown("---")
    st.subheader("🤖 Neural Architecture")
    st.markdown(
        """
        - **Model:** Bidirectional LSTM
        - **Embedding:** 128-dim + SpatialDropout
        - **Hidden Layers:** 64 units forward + 64 backward
        - **Intents:** 28 Categories
        - **Dataset:** 616 Utterances
        - **Held-out Test Acc:** **60.22%**
        """
    )

    st.markdown("---")
    if st.button("🗑️ Reset Conversation", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 Conversation reset! I'm **Nova AI**. What would you like to explore next?",
                "intent": "greeting",
                "confidence": 1.0,
            }
        ]
        st.session_state.last_spoken_id = None
        st.rerun()

# ─────────────────────────────────────────────────────────
# MAIN HERO HEADER
# ─────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="nova-header">
        <div class="nova-title">✨ Nova AI Assistant</div>
        <div class="nova-subtitle">Voice-Enabled Deep Learning Conversational Bot</div>
        <div class="nova-status">🟢 BiLSTM Model Active & Listening</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────
# VOICE INPUT BAR
# ─────────────────────────────────────────────────────────

voice_col1, voice_col2 = st.columns([3, 1])

with voice_col1:
    voice_prompt = speech_to_text(
        language="en",
        start_prompt="🎤 Speak to Nova",
        stop_prompt="⏹️ Stop Recording",
        just_once=True,
        use_container_width=True,
        key="nova_voice_recorder",
    )

with voice_col2:
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        if st.button("🔊 Replay", use_container_width=True, help="Replay Nova's last response"):
            trigger_speech(st.session_state.messages[-1]["content"])

# ─────────────────────────────────────────────────────────
# SUGGESTED PROMPT PILLS (Empty state or quick questions)
# ─────────────────────────────────────────────────────────

st.markdown("**💡 Quick Suggestions:**")
chip_cols = st.columns(4)

suggested_query = None
if chip_cols[0].button("🧠 Deep Learning", use_container_width=True):
    suggested_query = "What is deep learning?"
if chip_cols[1].button("⚡ Transformers", use_container_width=True):
    suggested_query = "Explain transformer models"
if chip_cols[2].button("👁️ Vision & CNN", use_container_width=True):
    suggested_query = "What is a convolutional neural network?"
if chip_cols[3].button("🚀 Project Ideas", use_container_width=True):
    suggested_query = "Suggest some good AI project ideas"

# ─────────────────────────────────────────────────────────
# DISPLAY CONVERSATION STREAM (Chat Bubbles)
# ─────────────────────────────────────────────────────────

for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant", avatar="✨"):
            st.markdown(msg["content"])
            if "intent" in msg and msg["intent"]:
                st.markdown(
                    f'<span class="intent-badge">🏷️ {msg["intent"]}</span>'
                    f'<span class="confidence-badge">🎯 {msg["confidence"]*100:.1f}% confidence</span>',
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────────────────
# HANDLE INCOMING INPUT (Voice, Suggested Chip, or Text Input)
# ─────────────────────────────────────────────────────────

chat_text = st.chat_input("Message Nova AI or use the microphone above...")

incoming_text = None
if voice_prompt:
    incoming_text = voice_prompt
elif suggested_query:
    incoming_text = suggested_query
elif chat_text:
    incoming_text = chat_text

if incoming_text:
    user_query = incoming_text.strip()

    # 1. Append user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 2. Predict intent & generate response using BiLSTM
    intent, confidence, response = generate_response(user_query)

    # 3. Append assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "intent": intent,
            "confidence": confidence,
        }
    )

    # 4. Auto-speak response if enabled
    if auto_speak:
        trigger_speech(response)

    st.rerun()
