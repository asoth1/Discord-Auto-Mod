"""
main_pipeline.py

Runs the full Discord AutoMod:
1. Toxicity with context
2. Sentiment + emotion + sarcasm
3. Spam + behavioral analysis
4. Feature fusion

"""

from toxicity import predict_with_context
from sentiment import analyze_message
from spam_behavioral import SpamBehavioralAnalyzer
from fusion import fuse_features
from decision_system import decide_action


# keep one analyzer instance alive so user history is preserved across messages
spam_analyzer = SpamBehavioralAnalyzer()


def run_pipeline(user_id: str, message: str, context_messages: list[str] | None = None) -> dict:
    """
    Run all modules and return one fused feature dictionary.

    Args:
        user_id: Discord user id or username
        message: current message to analyze
        context_messages: list of previous messages for context

    Returns:
        dict: fused feature representation
    """
    if context_messages is None:
        context_messages = []

    # 1) Toxicity module
    toxic_output = predict_with_context(context_messages, message)

    # 2) Sentiment + emotion + sarcasm module
    sentiment_output = analyze_message(message, context_messages)

    # 3) Spam + behavioral module
    spam_result = spam_analyzer.analyze(user_id, message)
    spam_output = spam_result.to_dict()

    # 4) Feature fusion
    fused_output = fuse_features(
        message,
        {
            "toxicity_model": toxic_output,
            "sentiment_model": sentiment_output,
            "spam_model": spam_output,
        },
    )
    
     # 5) Decision system
    decision = decide_action(fused_output)

    return {
        "message": message,
        "user_id": user_id,
        "context": context_messages,
        "raw_outputs": {
            "toxicity_model": toxic_output,
            "sentiment_model": sentiment_output,
            "spam_model": spam_output,
        },
        "fused_output": fused_output,
        "decision": decision
    }


def print_pipeline_result(result: dict) -> None:
    """
    Pretty-print the pipeline result.
    """
    print("\n" + "=" * 50)
    print("PIPELINE RESULT".center(50))
    print("=" * 50)

    print(f"User ID: {result['user_id']}")
    print(f"Message: {result['message']}")
    print(f"Context: {result['context']}")

    print("\n--- FUSED OUTPUT ---")
    for key, value in result["fused_output"].items():
        print(f"{key}: {value}")

    print("\n--- RAW SPAM VERDICT ---")
    spam_raw = result["raw_outputs"]["spam_model"]
    print("verdict:", spam_raw.get("verdict"))
    print("overall_score:", spam_raw.get("overall_score"))
    print("reasons:", spam_raw.get("reasons"))

    print("\n--- RAW TOXICITY SCORES ---")
    tox_raw = result["raw_outputs"]["toxicity_model"]
    print(tox_raw.get("scores", {}))

    print("\n--- RAW SENTIMENT / EMOTION / SARCASM ---")
    sent_raw = result["raw_outputs"]["sentiment_model"]
    print("sentiment:", sent_raw.get("sentiment", {}))
    print("emotion:", sent_raw.get("emotion", {}))
    print("sarcasm:", sent_raw.get("sarcasm", {}))
    
    print("\n--- DECISION SYSTEM ---")
    decision = result["decision"]
    print(f"Risk Score: {decision['risk_score']}")
    print(f"Action: {decision['action']}")
    print(f"Reason: {decision['reason']}")


if __name__ == "__main__":
    # Example 1
    result1 = run_pipeline(
        user_id="user123",
        message="you are so bad at this game",
        context_messages=["bro you sold that round", "nah i was lagging"],
    )
    print_pipeline_result(result1)

    # Example 2
    result2 = run_pipeline(
        user_id="spammer01",
        message="FREE NITRO CLICK HERE discord.gg/scamlink",
        context_messages=["join now", "best server ever"],
    )
    print_pipeline_result(result2)
