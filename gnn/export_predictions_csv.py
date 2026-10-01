import torch
import pandas as pd


input_path = "results/enhanced_gnn_predictions.pt"
output_path = "results/enhanced_gnn_predictions.csv"


print("===== LOADING PREDICTIONS =====")

data = torch.load(
    input_path,
    weights_only=False
)

predictions = data["predictions"]

print("Total predictions:", len(predictions))


print()
print("===== CREATING DATAFRAME =====")

df = pd.DataFrame(predictions)

df["prediction_label"] = df["prediction"].map({
    0: "DOWN",
    1: "UP"
})

df["confidence"] = df[
    ["probability_down", "probability_up"]
].max(axis=1)

df["confidence_percent"] = (
    df["confidence"] * 100
).round(2)


df = df[
    [
        "date",
        "ticker",
        "prediction",
        "prediction_label",
        "probability_down",
        "probability_up",
        "confidence_percent"
    ]
]


print()
print("===== SAMPLE =====")

print(
    df.head(10).to_string(
        index=False
    )
)


print()
print("===== SAVING CSV =====")

df.to_csv(
    output_path,
    index=False
)

print()
print("===== SUCCESS =====")

print(
    "CSV saved:"
)

print(output_path)