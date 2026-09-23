"""
VoiceBot AI — Training & Evaluation Script (Stage 2)
===================================================
Trains a Bidirectional LSTM model for intent classification on Dataset V2 (28 intents).
Generates detailed evaluation metrics:
  - evaluation/classification_report.txt
  - evaluation/confusion_matrix.png
  - evaluation/training_history.png
Saves production artifacts:
  - chatbot_model.keras
  - tokenizer.pkl
  - label_encoder.pkl
  - metadata.pkl
"""

import os
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Ensure output directory exists
os.makedirs("evaluation", exist_ok=True)

# ─────────────────────────────────────────────
# 1. Load Dataset V2
# ─────────────────────────────────────────────

with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern.lower().strip())
        labels.append(intent["tag"])

print("=" * 60)
print(" VoiceBot AI — Dataset V2 & BiLSTM Training")
print("=" * 60)
print(f"Total training samples : {len(texts)}")
print(f"Number of intents      : {len(set(labels))}")
print(f"All intent tags        : {sorted(set(labels))}")
print()

# ─────────────────────────────────────────────
# 2. Encode Labels
# ─────────────────────────────────────────────

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# ─────────────────────────────────────────────
# 3. Tokenize Text
# ─────────────────────────────────────────────

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
vocab_size = len(tokenizer.word_index) + 1
max_len = max(len(seq) for seq in sequences)

padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

print(f"Vocabulary size        : {vocab_size}")
print(f"Max sequence length    : {max_len}")
print()

# ─────────────────────────────────────────────
# 4. Stratified Split (70% Train, 15% Val, 15% Test)
# ─────────────────────────────────────────────

X_train, X_temp, y_train, y_temp = train_test_split(
    padded, encoded_labels, test_size=0.30, random_state=42, stratify=encoded_labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Train split size       : {len(X_train)} samples")
print(f"Validation split size  : {len(X_val)} samples")
print(f"Test split size        : {len(X_test)} samples")
print()

# ─────────────────────────────────────────────
# 5. Build BiLSTM Model Architecture
# ─────────────────────────────────────────────

EMBEDDING_DIM = 128
LSTM_UNITS = 64

from tensorflow.keras.layers import SpatialDropout1D

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(LSTM_UNITS)),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()
print()

# ─────────────────────────────────────────────
# 6. Train with Early Stopping
# ─────────────────────────────────────────────

early_stop = EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=20,
    restore_best_weights=True,
    verbose=1,
)

print("Starting training...")
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1,
)

# ─────────────────────────────────────────────
# 7. Evaluate on Held-out Test Set
# ─────────────────────────────────────────────

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print()
print("=" * 60)
print(f" Test Accuracy : {test_acc * 100:.2f}%")
print(f" Test Loss     : {test_loss:.4f}")
print("=" * 60)

# Predictions for classification metrics
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Classification Report
report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    zero_division=0,
)

print("\nClassification Report:\n")
print(report)

with open("evaluation/classification_report.txt", "w", encoding="utf-8") as f:
    f.write(f"VoiceBot AI — Test Set Evaluation Report\n")
    f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n")
    f.write(f"Test Loss: {test_loss:.4f}\n\n")
    f.write(report)

print("Saved: evaluation/classification_report.txt")

# ─────────────────────────────────────────────
# 8. Confusion Matrix Heatmap
# ─────────────────────────────────────────────

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    cbar=True,
)
plt.title(f"Confusion Matrix on Held-out Test Set (Accuracy: {test_acc*100:.1f}%)", fontsize=14, pad=15)
plt.xlabel("Predicted Intent", fontsize=12)
plt.ylabel("Ground Truth Intent", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig("evaluation/confusion_matrix.png", dpi=300)
plt.close()
print("Saved: evaluation/confusion_matrix.png")

# ─────────────────────────────────────────────
# 9. Training History Plot (Accuracy & Loss Curves)
# ─────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy Curve
ax1.plot(history.history["accuracy"], label="Training Accuracy", color="#1f77b4", linewidth=2)
ax1.plot(history.history["val_accuracy"], label="Validation Accuracy", color="#ff7f0e", linewidth=2, linestyle="--")
ax1.set_title("Model Accuracy Across Epochs", fontsize=12, fontweight="bold")
ax1.set_xlabel("Epoch", fontsize=10)
ax1.set_ylabel("Accuracy", fontsize=10)
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

# Loss Curve
ax2.plot(history.history["loss"], label="Training Loss", color="#1f77b4", linewidth=2)
ax2.plot(history.history["val_loss"], label="Validation Loss", color="#d62728", linewidth=2, linestyle="--")
ax2.set_title("Model Loss Across Epochs", fontsize=12, fontweight="bold")
ax2.set_xlabel("Epoch", fontsize=10)
ax2.set_ylabel("Sparse Categorical Crossentropy", fontsize=10)
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("evaluation/training_history.png", dpi=300)
plt.close()
print("Saved: evaluation/training_history.png")

# ─────────────────────────────────────────────
# 10. Save Production Artifacts
# ─────────────────────────────────────────────

model.save("chatbot_model.keras")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

with open("metadata.pkl", "wb") as f:
    pickle.dump({"max_len": max_len}, f)

print()
print("Saved production artifacts:")
print("  - chatbot_model.keras")
print("  - tokenizer.pkl")
print("  - label_encoder.pkl")
print("  - metadata.pkl")
print()
print("Stage 2 Training & Evaluation Complete!")
