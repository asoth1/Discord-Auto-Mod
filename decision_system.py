"""
Decision System

This takes the fused features from the feature fusion
and decides what moderation action to apply.

Possible actions:
    - allow   : message is safe
    - warn    : mild issue, notify user
    - delete  : remove the message
    - mute    : temporarily restrict user
    - ban     : severe violation, remove user
"""


def decide_action(fused):
    """
    Decide moderation action based on fused features.

    Parameters:
        fused (dict): output from feature fusion

    Returns:
        dict:
            {
                "risk_score": float,
                "action": str,
                "reason": str
            }
    """

    # Step 1: Compute risk score
    # Weighted combination of all features
    # Higher weight = more important signal

    risk_score = (
        0.25 * fused.get("toxicity", 0.0) +
        0.08 * fused.get("insult", 0.0) +
        0.15 * fused.get("threat", 0.0) +
        0.08 * fused.get("obscene", 0.0) +
        0.12 * fused.get("identity_hate", 0.0) +
        0.08 * fused.get("sentiment", 0.0) +   # negative sentiment
        0.05 * fused.get("anger", 0.0) +
        0.05 * fused.get("disgust", 0.0) +
        0.02 * fused.get("sarcasm", 0.0) +
        0.02 * fused.get("spam", 0.0) +
        0.05 * fused.get("repetition", 0.0) +
        0.10 * fused.get("url_risk", 0.0) +
        0.10 * fused.get("behavioral", 0.0) +
        0.08 * fused.get("mention", 0.0) +
        0.02 * fused.get("char_spam", 0.0)
    )

    # Step 2: Hard rules (priority)
    # These override the score when content is obviously harmful

    # Very high threat -> immediate ban
    if fused.get("threat", 0.0) >= 0.70:
        action = "ban"
        reason = "high threat content"

    # Strong hate + toxicity -> ban
    elif fused.get("identity_hate", 0.0) >= 0.70 and fused.get("toxicity", 0.0) >= 0.75:
        action = "ban"
        reason = "severe hateful toxic content"

    # Medium threat or hate -> mute
    elif fused.get("threat", 0.0) >= 0.60 or fused.get("identity_hate", 0.0) >= 0.60:
        action = "mute"
        reason = "serious harmful content"

    # Aggressive spam behavior -> mute
    elif fused.get("url_risk", 0.0) >= 0.85 and fused.get("behavioral", 0.0) >= 0.75:
        action = "mute"
        reason = "aggressive spam behavior"

    # Clear spam -> delete message
    elif fused.get("url_risk", 0.0) >= 0.65 or fused.get("repetition", 0.0) >= 0.85:
        action = "delete"
        reason = "spam or repeated message"

    # Step 3: Score-based decisions

    # Very high overall risk
    elif risk_score >= 0.75:
        action = "mute"
        reason = "very high overall risk"

    # High risk
    elif risk_score >= 0.50:
        action = "delete"
        reason = "high overall risk"

    # Medium risk
    elif risk_score >= 0.28:
        action = "warn"
        reason = "moderate risk"

    # Low risk
    else:
        action = "allow"
        reason = "low risk"

    # Step 4: Return result

    return {
        "risk_score": round(risk_score, 4),
        "action": action,
        "reason": reason
    }
