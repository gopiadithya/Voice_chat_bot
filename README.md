# 🎙️ VoiceBot AI

An intelligent voice-enabled chatbot powered by a **Bidirectional LSTM (BiLSTM)** deep learning model for intent classification across 28 technical and conversational domains.

## Architecture

```text
User Speech (Mic) ─► Speech Recognition (Web Speech API)
                              │
                              ▼
                       Recognized Text
                              │
                              ▼
            Text Preprocessing (Lower, Tokenize, Pad)
                              │
                              ▼
                   Embedding Layer (128-dim)
                              │
                              ▼
                      SpatialDropout1D (0.2)
                              │
                              ▼
                 Bidirectional LSTM (64 units)
                              │
                              ▼
                    Dense Layer (64, ReLU)
                              │
                              ▼
                        Dropout (0.3)
                              │
                              ▼
                     Softmax Classifier (28)
                              │
                              ▼
                  Predicted Intent & Confidence
                              │
                              ▼
                     Educational Response
```

## Key Highlights

- 🎤 **Multimodal Interaction:** Voice recording via browser microphone with seamless typed-text fallback.
- 🧠 **Genuine Deep Learning:** End-to-end BiLSTM neural network trained in TensorFlow/Keras (no external commercial API wrapper).
- 📚 **Dataset V2:** 28 distinct intent classes, 616 curated training utterances covering AI, ML, Deep Learning, CNN, RNN, Transformers, LLMs, Python, PyTorch, SQL, DSA, and college study advice.
- 📊 **Academic Evaluation Suite:** Automatic generation of precision/recall/F1 classification report, confusion matrix heatmap, and training/validation loss-accuracy curves.
- 🛡️ **Confidence Fallback:** Configurable rejection threshold ($< 0.45$) to safely reject out-of-domain queries.

---

## Project Structure

```text
voice_chatbot/
│
├── app.py                     # Streamlit multimodal web application
├── train.py                   # BiLSTM training & evaluation pipeline
├── intents.json               # Dataset V2 (28 intents, 616 patterns)
│
├── chatbot_model.keras        # Trained Keras neural network model
├── tokenizer.pkl              # Serialized Keras text tokenizer
├── label_encoder.pkl          # Serialized scikit-learn label encoder
├── metadata.pkl               # Model metadata (max_len = 9)
│
├── requirements.txt           # Environment dependencies
├── README.md                  # Project overview & instructions
├── REPORT.md                  # Comprehensive academic lab report
│
└── evaluation/
    ├── classification_report.txt  # Held-out test set metrics
    ├── confusion_matrix.png       # 28x28 confusion matrix heatmap
    └── training_history.png       # Loss & accuracy epoch curves
```

---

## Quickstart Guide

### 1. Set Up Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train & Evaluate the Model

```bash
python train.py
```

This trains the BiLSTM on Dataset V2, runs stratified testing on a held-out test split, and generates all artifact and evaluation files inside `evaluation/`.

### 4. Launch the Web Interface

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Click **🎤 Start speaking** to ask questions with your microphone or type your question in the text box.

---

## Evaluation Summary (Dataset V2)

- **Total Classes:** 28 intents
- **Held-out Test Samples:** 93 samples (strictly unseen during training)
- **Random Chance Baseline:** 3.57% ($1/28$)
- **Held-out Test Accuracy:** **60.22%** (~17× higher than random baseline)
- **Macro Precision:** **0.66**
- **Weighted F1-Score:** **0.60**
- **Top Performing Intents (F1 = 1.00):** `python`, `thanks`, `transformer`
