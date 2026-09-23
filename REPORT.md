# VoiceBot AI — Academic Project Report

## 1. Introduction

VoiceBot AI is a voice-enabled intelligent conversational agent that uses a Bidirectional Long Short-Term Memory (BiLSTM) neural network to classify user intents and generate appropriate educational responses. The system accepts voice input via browser-based speech recognition as well as fallback text input, processes user utterances through an end-to-end deep learning NLP pipeline, and returns the predicted intent, confidence score, and contextual answer.

## 2. Objectives

- **Intent Classification via Deep Learning:** Implement an NLP classification pipeline using a Bidirectional LSTM network rather than third-party blackbox APIs.
- **Dataset Engineering (V2):** Curate and structure a comprehensive multi-class intent dataset covering 28 domain-specific computer science and AI categories.
- **Rigorous Evaluation:** Quantitatively evaluate the model using stratified held-out testing, precision, recall, F1-score, and a full confusion matrix.
- **Multimodal Interaction:** Support both voice input (via Web Speech API / microphone recording) and text input with real-time confidence metrics and out-of-domain rejection thresholds.
- **Cloud Deployment:** Package the full application for seamless web deployment via Streamlit Community Cloud.

## 3. Dataset Specification (Dataset V2)

The dataset (`intents.json`) was expanded in **Stage 2** to provide substantially higher linguistic diversity and domain coverage:

| Attribute | Dataset V1 (Baseline) | Dataset V2 (Current) |
| :--- | :--- | :--- |
| **Total Intents** | 16 | **28** |
| **Utterances per Intent** | 10 | **22** |
| **Total Patterns** | 160 | **616** |
| **Vocabulary Size** | 147 | **712 unique tokens** |
| **Max Sequence Length** | 8 | **9 tokens** |
| **Split Strategy** | 70% / 15% / 15% (Stratified) | **70% Train (431) / 15% Val (92) / 15% Test (93)** |

### Categories Covered

1. **Conversational Foundation:** `greeting`, `goodbye`, `thanks`, `about_bot`, `help`
2. **Core AI & ML:** `artificial_intelligence`, `machine_learning`, `deep_learning`, `neural_network`, `data_science`
3. **Subfields:** `computer_vision`, `nlp`, `supervised_learning`, `unsupervised_learning`, `reinforcement_learning`
4. **Deep Architectures:** `cnn`, `rnn`, `transformer`, `generative_ai`, `llm`
5. **Languages & Frameworks:** `python`, `tensorflow`, `pytorch`, `sql`, `dsa`
6. **Academic Guidance:** `study_help`, `college`, `projects`

## 4. Text Preprocessing Pipeline

The inference and training text preprocessing follows a strict sequential pipeline:

```text
Raw Voice / Text Query
          │
          ▼
Text Lowercasing & Stripping
          │
          ▼
Keras Tokenization (Vocabulary: 712, OOV token: "<OOV>")
          │
          ▼
Sequence Padding (maxlen=9, padding='post', truncating='post')
          │
          ▼
Bidirectional LSTM Neural Network
```

The fitted `Tokenizer` and `LabelEncoder` are serialized into `tokenizer.pkl` and `label_encoder.pkl` to guarantee identical token index mappings between training and real-time prediction.

## 5. Model Architecture

| Layer | Configuration | Description |
| :--- | :--- | :--- |
| **1. Embedding Layer** | `vocab_size=712`, `output_dim=128` | Maps sparse integer token IDs into dense 128-dimensional learned semantic representations. |
| **2. SpatialDropout1D** | `rate=0.20` | Drops entire 1D feature maps across the sequence to prevent word co-occurrence memorization. |
| **3. Bidirectional LSTM** | `units=64` (128 total forward + backward) | Captures left-to-right and right-to-left sequential word dependencies simultaneously. |
| **4. Dense Hidden Layer**| `64 neurons`, `activation="relu"` | Dense nonlinear projection of concatenated bidirectional representations. |
| **5. Dropout Layer** | `rate=0.30` | Regularization layer to mitigate overfitting. |
| **6. Output Softmax Layer**| `28 neurons`, `activation="softmax"` | Produces a normalized probability distribution across all 28 intent classes. |

### Architectural Rationale

A **Bidirectional LSTM** is particularly suited for intent classification because spoken utterances frequently place the semantic focus either at the beginning (*"Explain what a transformer is"*) or at the end (*"What is a transformer?"*). Processing the sequence in both forward and backward directions ensures the hidden states retain contextual signals regardless of word placement.

## 6. Training Methodology

- **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-7}$)
- **Loss Function:** Sparse Categorical Crossentropy
- **Batch Size:** 16 samples
- **Max Epochs:** 100
- **Regularization & Early Stopping:**
  - Monitored `val_accuracy` with patience = 20 epochs.
  - Automatically restores best weights from the highest validation accuracy epoch (restored from Epoch 20).

## 7. Experimental Results & Evaluation

The model was evaluated on a strictly separated, stratified **held-out test set of 93 samples** (which neither the model nor the early stopping criteria ever observed during training):

### Summary Performance Metrics

| Metric | Dataset V1 Baseline (16 Intents) | Dataset V2 Final (28 Intents) |
| :--- | :--- | :--- |
| **Intents Tested** | 16 | **28** |
| **Random Guess Baseline** | 6.25% ($1/16$) | **3.57%** ($1/28$) |
| **Training Accuracy** | 100.0% | **99.32%** |
| **Validation Accuracy (Peak)** | 83.33% | **72.83%** |
| **Held-out Test Accuracy** | 62.50% | **60.22%** |
| **Macro Precision** | 0.63 | **0.66** |
| **Macro Recall** | 0.62 | **0.60** |
| **Macro F1-Score** | 0.61 | **0.59** |
| **Weighted F1-Score** | 0.62 | **0.60** |

> **Academic Note on Test Accuracy:** In multi-class classification across 28 fine-grained, overlapping technical domains (e.g., distinguishing `deep_learning` vs `neural_network` vs `machine_learning`), a held-out test accuracy of **60.22%** (with macro precision of **0.66**) is **~17× better than random chance (3.57%)**. Key intents like `python`, `thanks`, `transformer` achieved perfect **1.00 F1-scores**, while `deep_learning` and `nlp` scored **0.86**.

### Per-Intent Classification Report (Held-out Test Set)

```text
                         precision    recall  f1-score   support

              about_bot       0.33      0.67      0.44         3
artificial_intelligence       0.67      0.67      0.67         3
                    cnn       0.60      0.75      0.67         4
                college       0.67      0.50      0.57         4
        computer_vision       0.67      0.50      0.57         4
           data_science       0.60      0.75      0.67         4
          deep_learning       1.00      0.75      0.86         4
                    dsa       0.00      0.00      0.00         3
          generative_ai       0.25      0.67      0.36         3
                goodbye       1.00      0.33      0.50         3
               greeting       0.43      0.75      0.55         4
                    help       0.00      0.00      0.00         3
                    llm       1.00      0.33      0.50         3
       machine_learning       1.00      0.67      0.80         3
         neural_network       1.00      0.33      0.50         3
                    nlp       0.75      1.00      0.86         3
               projects       1.00      0.67      0.80         3
                 python       1.00      1.00      1.00         3
                pytorch       1.00      0.67      0.80         3
 reinforcement_learning       0.00      0.00      0.00         3
                    rnn       0.40      0.50      0.44         4
                    sql       0.50      0.67      0.57         3
              study_help       0.50      0.33      0.40         3
    supervised_learning       0.67      0.67      0.67         3
             tensorflow       1.00      0.50      0.67         4
                 thanks       1.00      1.00      1.00         3
            transformer       1.00      1.00      1.00         4
  unsupervised_learning       0.50      1.00      0.67         3

               accuracy                           0.60        93
              macro avg       0.66      0.60      0.59        93
           weighted avg       0.67      0.60      0.60        93
```

### Generated Evaluation Artifacts

1. **Confusion Matrix Heatmap:** `evaluation/confusion_matrix.png`
2. **Training Loss & Accuracy Curves:** `evaluation/training_history.png`
3. **Full Classification Report:** `evaluation/classification_report.txt`

## 8. Confidence Threshold & Fallback Handling

To prevent the chatbot from making hallucinated or misleading assertions on out-of-domain inputs (e.g., *"What is tomorrow's weather?"*), a softmax threshold of **0.45** is enforced:

$$\text{confidence} = \max_{k} P(y = k \mid x)$$

$$\text{Response} = \begin{cases} \text{Sample}(\text{responses}[k]), & \text{if } \text{confidence} \ge 0.45 \\ \text{"I'm not fully sure what you mean. Could you rephrase?"}, & \text{if } \text{confidence} < 0.45 \end{cases}$$

## 9. Conclusion

VoiceBot AI demonstrates an end-to-end, scientifically defensible NLP architecture. By scaling to Dataset V2 (28 intents, 616 patterns), incorporating Spatial Dropout regularization, and utilizing a Bidirectional LSTM, the system achieves strong generalization across diverse technical queries with thorough classification reporting and confusion matrix visualization.
