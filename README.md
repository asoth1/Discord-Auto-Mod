# Discord AutoMod NLP System

---

# Project Overview

This project is a Discord AutoModeration system powered by NLP techniques.  
It analyzes user messages in real time and determines whether they should be:

- 🟢 Allowed
- 🟡 Warned
- 🔴 Deleted
- 🟣 Mute
- ⚫ Ban

The moderation system combines multiple NLP signals including toxicity detection, sentiment analysis, sarcasm detection, and spam behavior analysis into a unified feature fusion model. A decision system then calculates a risk score and selects the final moderation action.

The project also includes an interactive Streamlit dashboard for visualizing moderation decisions, feature scores, explanations, and moderation history.

---

# Key Features

## 1) Multi-model NLP pipeline

The system integrates several NLP components:

- Toxicity detection
- Sentiment analysis
- Emotion detection
- Sarcasm detection
- Spam classification
- Behavioral spam analysis
- Feature fusion
- Rule + score-based moderation decision system
- Interactive moderation dashboard
- Context-aware message analysis

---

## 2) Interactive Dashboard

The Streamlit dashboard provides:

- Real-time message moderation
- Risk score visualization
- Feature importance explanations
- Moderation history tracking
- Raw model output inspection

---

## 3) Context-Aware Analysis

The system supports contextual moderation by analyzing previous chat messages alongside the current message.

Example:

```text
Context:
    "we are joking lol"

Message:
    "shut up idiot"
```

This helps the system distinguish between harmful toxicity and casual banter.

---

# System Architecture

```text
Input Message
    ↓
Toxicity Detection
Sentiment + Emotion + Sarcasm
Spam + Behavioral Analysis
    ↓
Feature Fusion
    ↓
Risk Score Calculation
    ↓
Decision System
    ↓
Dashboard Output
```

---

# Project Structure

```text
discord-automod/
│
├── dashboard.py              # Streamlit moderation dashboard
├── main.py                   # Main NLP pipeline
├── fusion.py                 # Feature fusion module
├── decision_system.py        # Moderation decision logic
├── toxicity.py               # Toxicity detection
├── sentiment.py              # Sentiment/emotion/sarcasm analysis
├── spam_behavioral.py        # Spam + behavioral analysis
├── evaluation.py             # Evaluation script
├── evaluation_data.json      # Evaluation dataset
├── requirements.txt
└── README.md
```

---

# Requirements

## Software

- Python 3.9+
- pip

## Main Libraries

- transformers
- torch
- streamlit
- pandas
- numpy
- scikit-learn

---

# Installation

## Step 1: Clone the Repository

```bash
git clone https://github.com/spotlur2/Discord-Auto-Mod.git
cd Discord-Auto-Mod
```

---

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Run the Interactive Dashboard

```bash
streamlit run dashboard.py
```

or

```bash
python -m streamlit run dashboard.py
```

Then open the local Streamlit address shown in the terminal (usually http://localhost:8501).

The dashboard allows users to:

- Enter Discord messages
- Add context messages
- View moderation decisions
- Analyze feature scores
- Inspect risk explanations

---

## Optional: Run the Backend Pipeline Only

```bash
python main.py
```

This runs console-based examples of the moderation pipeline without the dashboard UI.

---

# Example Test Cases

## 1) Normal Message

```text
"Hey everyone, how are you doing today?"
```

→ 🟢 Allow

---

## 2) Toxic Message

```text
"You are so stupid and useless"
```

→ 🔴 Delete

---

## 3) Severe Threat

```text
"I will hurt you"
```

→ ⚫ Ban

---

## 4) Spam Message

```text
"FREE MONEY CLICK HERE http://spam.com NOW!!!"
```

→ 🔴 Delete

---

# Decision Logic

The system computes a weighted risk score using fused NLP features.

Simplified example:

```text
risk =
    toxicity +
    threat +
    spam +
    anger +
    behavioral signals
```

The system also includes hard moderation rules for severe threats, hate speech, and aggressive spam behavior.

Example moderation thresholds:

```text
0.00 – 0.27         → Allow
0.28 – 0.49         → Warn
0.50 – 0.74         → Delete
0.75+               → Mute
Extreme threat/hate → Ban
```

Based on the implementation in `decision_system.py`

---

# Evaluation

The project includes an evaluation pipeline using labeled moderation examples stored in `evaluation_data.json`.

## To run evaluation

```bash
python evaluation.py
```

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score

---

# NLP Models Used

## Toxicity Detection

`unitary/toxic-bert`

## Sentiment Analysis

`cardiffnlp/twitter-roberta-base-sentiment-latest`

## Emotion Detection

`j-hartmann/emotion-english-distilroberta-base`

## Sarcasm Detection

`cardiffnlp/twitter-roberta-base-irony`

## Spam Classification

`mshenoda/roberta-spam`