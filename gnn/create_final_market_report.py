import torch
import pandas as pd


# ============================================================
# PATHS
# ============================================================

prediction_path = "results/enhanced_gnn_predictions.pt"

evaluation_path = (
    "results/enhanced_training_only_gnn_evaluation.pt"
)

output_csv = "results/final_market_intelligence_report.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("===== LOADING PREDICTIONS =====")

prediction_data = torch.load(
    prediction_path,
    weights_only=False
)

predictions = prediction_data["predictions"]

print("Predictions:", len(predictions))


print()
print("===== LOADING EVALUATION =====")

evaluation = torch.load(
    evaluation_path,
    weights_only=False
)


# ============================================================
# CREATE DATAFRAME
# ============================================================

print()
print("===== CREATING REPORT =====")

df = pd.DataFrame(predictions)


# ============================================================
# PREDICTION LABEL
# ============================================================

df["prediction_label"] = df["prediction"].map(
    {
        0: "DOWN",
        1: "UP"
    }
)


# ============================================================
# CONFIDENCE
# ============================================================

df["confidence"] = df[
    [
        "probability_down",
        "probability_up"
    ]
].max(axis=1)


df["confidence_percent"] = (
    df["confidence"] * 100
).round(2)


# ============================================================
# SIGNAL STRENGTH
# ============================================================

def get_signal_strength(confidence):

    if confidence >= 0.60:
        return "Strong"

    elif confidence >= 0.55:
        return "Moderate"

    else:
        return "Weak"


df["signal_strength"] = df[
    "confidence"
].apply(get_signal_strength)


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

df = df[
    [
        "date",
        "ticker",
        "prediction",
        "prediction_label",
        "probability_down",
        "probability_up",
        "confidence_percent",
        "signal_strength"
    ]
]


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [
        "date",
        "confidence_percent"
    ],
    ascending=[
        True,
        False
    ]
)


# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    output_csv,
    index=False
)


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

accuracy = evaluation["accuracy"]
precision = evaluation["precision"]
recall = evaluation["recall"]
f1 = evaluation["f1"]


print()
print("========================================")
print("     FINAL MARKET INTELLIGENCE REPORT")
print("========================================")

print()
print("Total predictions:", len(df))

print(
    "UP predictions:",
    int((df["prediction"] == 1).sum())
)

print(
    "DOWN predictions:",
    int((df["prediction"] == 0).sum())
)

print()
print("===== MODEL PERFORMANCE =====")

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


# ============================================================
# SIGNAL DISTRIBUTION
# ============================================================

print()
print("===== SIGNAL STRENGTH =====")

print(
    df["signal_strength"].value_counts()
)


# ============================================================
# TOP CONFIDENCE PREDICTIONS
# ============================================================

print()
print("===== TOP 10 CONFIDENCE PREDICTIONS =====")

print(
    df[
        [
            "date",
            "ticker",
            "prediction_label",
            "probability_down",
            "probability_up",
            "confidence_percent",
            "signal_strength"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# SUCCESS
# ============================================================

print()
print("===== SUCCESS =====")

print(
    "Final report saved:"
)

print(output_csv)