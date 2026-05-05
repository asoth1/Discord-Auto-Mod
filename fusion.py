"""
Feature Fusion
1) Combines outputs from multiple models into one unified feature representation.
2) Prepares features for downstream scoring and moderation decisions.
"""


# Helper functions

def clamp_to_01(value):
    """Force value into [0, 1]."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(value, 1.0))


# Adapter functions

def adapt_toxicity_output(toxicity_output):
    """
    Accepts either:
      - {"scores": {...}, "combined_text": "..."}
      - or just a raw scores dict

    Expected source keys from toxicity module include:
      toxic, insult, threat, obscene, identity_hate, etc.
    """
    if not isinstance(toxicity_output, dict):
        toxicity_output = {}

    scores = toxicity_output.get("scores", toxicity_output)

    return {
        "toxicity": clamp_to_01(scores.get("toxic", 0.0)),
        "insult": clamp_to_01(scores.get("insult", 0.0)),
        "threat": clamp_to_01(scores.get("threat", 0.0)),
        "obscene": clamp_to_01(scores.get("obscene", 0.0)),
        "identity_hate": clamp_to_01(scores.get("identity_hate", 0.0)),
    }


def adapt_sentiment_output(sentiment_output):
    """
    Accepts output from sentiment.py analyze_message(...), which returns:
      {
        "message": ...,
        "sentiment": {...},
        "emotion": {...},
        "sarcasm": {...}
      }

    We use:
      - negative sentiment as risk-oriented sentiment signal
      - anger as emotion signal
      - irony as sarcasm score
    """
    if not isinstance(sentiment_output, dict):
        sentiment_output = {}

    sentiment_scores = sentiment_output.get("sentiment", {})
    emotion_scores = sentiment_output.get("emotion", {})
    sarcasm_scores = sentiment_output.get("sarcasm", {})

    return {
        "sentiment": clamp_to_01(sentiment_scores.get("negative", 0.0)),
        "anger": clamp_to_01(emotion_scores.get("anger", 0.0)),
        "sarcasm": clamp_to_01(
            sarcasm_scores.get("irony",
            sarcasm_scores.get("sarcastic", 0.0))
        ),
        "disgust": clamp_to_01(emotion_scores.get("disgust", 0.0)),
    }


def adapt_spam_output(spam_output):
    """
    Accepts output from SpamBehavioralAnalyzer.analyze(...).
    That object may be an AnalysisResult or a dict.
    """
    if hasattr(spam_output, "to_dict"):
        spam_output = spam_output.to_dict()

    if not isinstance(spam_output, dict):
        spam_output = {}

    return {
        "spam": clamp_to_01(spam_output.get("spam_score", 0.0)),
        "repetition": clamp_to_01(spam_output.get("repetition_score", 0.0)),
        "url_risk": clamp_to_01(spam_output.get("link_score", 0.0)),
        "behavioral": clamp_to_01(spam_output.get("behavioral_score", 0.0)),
        "flood": clamp_to_01(spam_output.get("flood_score", 0.0)),
        "mention": clamp_to_01(spam_output.get("mention_score", 0.0)),
        "char_spam": clamp_to_01(spam_output.get("char_spam_score", 0.0)),
    }


# Main fusion function

def fuse_features(message, model_outputs):
    """
    Combines all model outputs into one fused feature dictionary.

    Expected model_outputs keys:
      - toxicity_model
      - sentiment_model
      - spam_model

    sarcasm is extracted from sentiment_model because that teammate's
    module already includes sarcasm inside the same output.
    """
    if not isinstance(model_outputs, dict):
        model_outputs = {}

    toxicity_data = adapt_toxicity_output(model_outputs.get("toxicity_model", {}))
    sentiment_data = adapt_sentiment_output(model_outputs.get("sentiment_model", {}))
    spam_data = adapt_spam_output(model_outputs.get("spam_model", {}))

    fused = {
        "message": message,

        # Toxicity
        "toxicity": toxicity_data["toxicity"],
        "insult": toxicity_data["insult"],
        "threat": toxicity_data["threat"],
        "obscene": toxicity_data["obscene"],
        "identity_hate": toxicity_data["identity_hate"],

        # Sentiment / emotion / sarcasm
        "sentiment": sentiment_data["sentiment"],   # negative sentiment probability
        "anger": sentiment_data["anger"],
        "disgust": sentiment_data["disgust"],
        "sarcasm": sentiment_data["sarcasm"],

        # Spam / behavioral
        "spam": spam_data["spam"],
        "repetition": spam_data["repetition"],
        "url_risk": spam_data["url_risk"],
        "behavioral": spam_data["behavioral"],
        "flood": spam_data["flood"],
        "mention": spam_data["mention"],
        "char_spam": spam_data["char_spam"],
    }

    return fused


# Batch fusion

def fuse_multiple(messages_with_outputs):
    """
    messages_with_outputs = [
        {
            "message": "...",
            "outputs": {
                "toxicity_model": ...,
                "sentiment_model": ...,
                "spam_model": ...
            }
        }
    ]
    """
    fused_outputs = []

    for item in messages_with_outputs:
        message = item.get("message", "")
        outputs = item.get("outputs", {})
        fused_outputs.append(fuse_features(message, outputs))

    return fused_outputs


# ===========================
# Example test
# ===========================

if __name__ == "__main__":
    print("----- Feature Fusion Test -----\n")

    test_data = [
        {
            "message": "you are so bad at this lol",
            "outputs": {
                "toxicity_model": {
                    "scores": {
                        "toxic": 0.83,
                        "insult": 0.72,
                        "threat": 0.10,
                        "obscene": 0.21
                    }
                },
                "sentiment_model": {
                    "sentiment": {
                        "negative": 0.81,
                        "neutral": 0.15,
                        "positive": 0.04
                    },
                    "emotion": {
                        "anger": 0.70,
                        "joy": 0.05,
                        "disgust": 0.33
                    },
                    "sarcasm": {
                        "irony": 0.25,
                        "non_irony": 0.75
                    }
                },
                "spam_model": {
                    "spam_score": 0.12,
                    "repetition_score": 0.20,
                    "link_score": 0.05,
                    "behavioral_score": 0.20,
                    "flood_score": 0.10,
                    "mention_score": 0.00,
                    "char_spam_score": 0.00
                }
            }
        },
        {
            "message": "Join my server now! discord.gg/test_data",
            "outputs": {
                "toxicity_model": {
                    "scores": {
                        "toxic": 0.03,
                        "insult": 0.01,
                        "threat": 0.00
                    }
                },
                "sentiment_model": {
                    "sentiment": {
                        "negative": 0.10,
                        "neutral": 0.20,
                        "positive": 0.70
                    },
                    "emotion": {
                        "anger": 0.03,
                        "disgust": 0.02
                    },
                    "sarcasm": {
                        "irony": 0.01,
                        "non_irony": 0.99
                    }
                },
                "spam_model": {
                    "spam_score": 0.90,
                    "repetition_score": 0.60,
                    "link_score": 0.95,
                    "behavioral_score": 0.95,
                    "flood_score": 0.50,
                    "mention_score": 0.00,
                    "char_spam_score": 0.10
                }
            }
        }
    ]

    fused_outputs = fuse_multiple(test_data)

    print("Fused result:\n")
    for i, output in enumerate(fused_outputs):
        print(f"--- Message {i+1} ---")
        for key, value in output.items():
            print(f"{key}: {value}")
        print()
