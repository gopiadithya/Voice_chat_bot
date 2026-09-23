import json
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from streamlit_mic_recorder import speech_to_text


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="VoiceBot AI",
    page_icon="🎙️",
    layout="centered"
)


# ---------------------------------------------------------
# LOAD FILES
# ---------------------------------------------------------

@st.cache_resource
def load_chatbot():

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


model, tokenizer, label_encoder, metadata, intents_data = load_chatbot()


# ---------------------------------------------------------
# RESPONSE DATABASE
# ---------------------------------------------------------

responses = {}

for intent in intents_data["intents"]:
    responses[intent["tag"]] = intent["responses"]


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


if "recognized_text" not in st.session_state:
    st.session_state.recognized_text = ""


# ---------------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------------

def predict_intent(text):

    sequence = tokenizer.texts_to_sequences([text.lower()])

    padded = pad_sequences(
        sequence,
        maxlen=metadata["max_len"],
        padding="post",
        truncating="post"
    )

    probabilities = model.predict(
        padded,
        verbose=0
    )[0]

    predicted_index = np.argmax(probabilities)

    confidence = float(probabilities[predicted_index])

    intent = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    return intent, confidence


# ---------------------------------------------------------
# RESPONSE FUNCTION
# ---------------------------------------------------------

def generate_response(text):

    intent, confidence = predict_intent(text)

    # Low confidence handling
    if confidence < 0.45:

        response = (
            "I'm not completely sure what you mean. "
            "Could you please rephrase your question?"
        )

        return intent, confidence, response

    response = np.random.choice(
        responses.get(
            intent,
            ["Sorry, I don't know how to answer that."]
        )
    )

    return intent, confidence, response


# ---------------------------------------------------------
# TEXT TO SPEECH
# ---------------------------------------------------------

def speak_text(text):

    safe_text = (
        text.replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("\n", " ")
    )

    st.components.v1.html(
        f"""
        <script>
            const text = `{safe_text}`;

            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();

                const speech =
                    new SpeechSynthesisUtterance(text);

                speech.rate = 1.0;
                speech.pitch = 1.0;
                speech.volume = 1.0;

                window.speechSynthesis.speak(speech);
            }}
        </script>
        """,
        height=0
    )


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("🎙️ VoiceBot AI")

st.markdown(
    """
    ### Deep Learning Based Voice-Enabled Chatbot

    Speak naturally or type your question.  
    The system converts your speech into text, predicts the
    user's intent using a **Bidirectional LSTM**, and generates
    an appropriate response.
    """
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("🤖 Model Information")

    st.write("**Architecture:** BiLSTM")
    st.write("**Intents:** 28")
    st.write("**Dataset:** 616 utterances")
    st.write("**Test Accuracy:** 60.22%")
    st.write("**Vocabulary:** 712 words")

    st.divider()

    st.write(
        "Speech recognition is handled through the browser "
        "microphone interface."
    )

    st.write(
        "The chatbot uses a trained deep learning model "
        "for intent classification."
    )


# ---------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------

st.subheader("💬 Conversation")

if len(st.session_state.messages) == 0:

    st.info(
        "Start by speaking into the microphone or typing "
        "a question below."
    )

else:

    for message in st.session_state.messages:

        if message["role"] == "user":

            with st.chat_message("user"):
                st.write(message["content"])

        else:

            with st.chat_message("assistant"):
                st.write(message["content"])


# ---------------------------------------------------------
# MICROPHONE INPUT
# ---------------------------------------------------------

st.subheader("🎤 Voice Input")

voice_text = speech_to_text(
    language="en",
    start_prompt="🎤 Start Speaking",
    stop_prompt="⏹️ Stop Recording",
    just_once=True,
    use_container_width=True,
    key="voice_input"
)


# ---------------------------------------------------------
# PROCESS VOICE INPUT
# ---------------------------------------------------------

if voice_text:

    st.session_state.recognized_text = voice_text


# ---------------------------------------------------------
# DISPLAY RECOGNIZED SPEECH
# ---------------------------------------------------------

if st.session_state.recognized_text:

    st.success(
        f"🗣️ Recognized Speech: "
        f"**{st.session_state.recognized_text}**"
    )


# ---------------------------------------------------------
# TEXT INPUT
# ---------------------------------------------------------

st.subheader("⌨️ Text Input")

typed_text = st.text_input(
    "Type your question",
    placeholder="Example: What is deep learning?"
)


# ---------------------------------------------------------
# SEND BUTTON
# ---------------------------------------------------------

if st.button(
    "🚀 Ask Chatbot",
    use_container_width=True
):

    user_text = typed_text.strip()

    if not user_text:

        user_text = st.session_state.recognized_text.strip()

    if not user_text:

        st.warning(
            "Please speak into the microphone or type "
            "a question."
        )

    else:

        # Add user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_text
            }
        )

        # Generate response
        intent, confidence, response = generate_response(
            user_text
        )

        # Add bot response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        # Store prediction information
        st.session_state.last_intent = intent
        st.session_state.last_confidence = confidence
        st.session_state.last_response = response

        # Clear recognized text
        st.session_state.recognized_text = ""

        # Refresh UI
        st.rerun()


# ---------------------------------------------------------
# MODEL PREDICTION DETAILS
# ---------------------------------------------------------

if "last_intent" in st.session_state:

    st.divider()

    st.subheader("🧠 Model Prediction")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Intent",
            st.session_state.last_intent
        )

    with col2:

        st.metric(
            "Confidence",
            f"{st.session_state.last_confidence * 100:.2f}%"
        )

    # Confidence progress bar
    st.progress(
        min(
            st.session_state.last_confidence,
            1.0
        )
    )


# ---------------------------------------------------------
# TEXT TO SPEECH
# ---------------------------------------------------------

if "last_response" in st.session_state:

    if st.button(
        "🔊 Speak Response",
        use_container_width=True
    ):

        speak_text(
            st.session_state.last_response
        )


# ---------------------------------------------------------
# CLEAR CHAT
# ---------------------------------------------------------

st.divider()

if st.button(
    "🗑️ Clear Conversation",
    use_container_width=True
):

    st.session_state.messages = []
    st.session_state.recognized_text = ""

    if "last_intent" in st.session_state:
        del st.session_state.last_intent

    if "last_confidence" in st.session_state:
        del st.session_state.last_confidence

    if "last_response" in st.session_state:
        del st.session_state.last_response

    st.rerun()
