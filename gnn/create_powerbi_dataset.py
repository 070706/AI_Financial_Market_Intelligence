import pandas as pd


input_path = "results/final_market_intelligence_report.csv"
output_path = "results/powerbi_market_intelligence.csv"


print("===== LOADING FINAL REPORT =====")

df = pd.read_csv(input_path)

print("Rows:", len(df))


print()
print("===== PREPARING POWER BI DATA =====")

df["date"] = pd.to_datetime(
    df["date"]
)

df["prediction_label"] = df[
    "prediction_label"
].astype(str)

df["signal_strength"] = df[
    "signal_strength"
].astype(str)


# Convert probabilities to percentages
df["down_probability_percent"] = (
    df["probability_down"] * 100
).round(2)

df["up_probability_percent"] = (
    df["probability_up"] * 100
).round(2)


# Keep dashboard-friendly columns
df = df[
    [
        "date",
        "ticker",
        "prediction_label",
        "down_probability_percent",
        "up_probability_percent",
        "confidence_percent",
        "signal_strength"
    ]
]


# Sort by date and ticker
df = df.sort_values(
    [
        "date",
        "ticker"
    ]
)


print()
print("===== SAMPLE =====")

print(
    df.head(10).to_string(
        index=False
    )
)


print()
print("===== SAVING =====")

df.to_csv(
    output_path,
    index=False
)


print()
print("===== SUCCESS =====")

print(
    "Power BI dataset saved:"
)

print(output_path)