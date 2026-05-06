import json
from main import run_pipeline
from sklearn.metrics import classification_report, accuracy_score

def evaluate():
    """
    Evaluate the moderation system using a labeled dataset.

    1. Loads evaluation data with gold labels
    2. Runs the full NLP pipeline on each message
    3. Compares predicted actions with gold labels
    4. Computes evaluation metrics (accuracy, precision, recall, F1) """

    # load evaluation dataset (contains message, context, and gold label)
    with open("evaluation_data.json") as f:
        data = json.load(f)

    # lists to store ground truth labels and model predictions
    y_true = []   # actual labels (gold)
    y_pred = []   # predicted labels from system

    print("\n" + "-" * 50)
    print("EVALUATION RESULTS")
    print("-" * 50)

    # loop through each example in the dataset
    for item in data:

        # run full moderation pipeline
        # include: toxicity -> sentiment -> spam -> fusion -> decision
        result = run_pipeline(
            user_id = "test_user",
            message = item["message"],
            context_messages = item["context"]
        )

        # extract predicted action from decision system
        predicted = result["decision"]["action"]

        # extract gold (true) label from dataset
        gold = item["label"]

        # store results for metric calculation
        y_true.append(gold)
        y_pred.append(predicted)

        # print detailed comparison for each example
        print("\nMessage:", item["message"])
        print("Context:", item["context"])
        print("Gold:", gold)
        print("Predicted:", predicted)
        print("Risk Score:", result["decision"]["risk_score"])

    print("\n" + "-" * 50)
    print("FINAL METRICS")
    print("-" * 50)

    # overall accuracy 
    print("\nAccuracy:", round(accuracy_score(y_true, y_pred), 4))

    # detailed classification report:
    # Precision, Recall, F1-score
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits = 4))


if __name__ == "__main__":
    evaluate()