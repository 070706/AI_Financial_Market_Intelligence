import torch


# ============================================================
# ORIGINAL MODEL RESULTS
# ============================================================

original_accuracy = 0.5152


# ============================================================
# NEW MODEL RESULTS
# ============================================================

new_results = torch.load(
    "results/training_only_gnn_evaluation.pt",
    weights_only=False
)

new_accuracy = new_results["accuracy"]
new_precision = new_results["precision"]
new_recall = new_results["recall"]
new_f1 = new_results["f1"]


# ============================================================
# COMPARISON
# ============================================================

print()
print("========================================")
print("       GNN MODEL COMPARISON")
print("========================================")

print()

print(
    "Original GNN accuracy :",
    f"{original_accuracy * 100:.2f}%"
)

print(
    "New GNN accuracy      :",
    f"{new_accuracy * 100:.2f}%"
)

print()

difference = (
    new_accuracy - original_accuracy
)

print(
    "Accuracy difference   :",
    f"{difference * 100:.2f} percentage points"
)

print()

print("===== NEW MODEL METRICS =====")

print(
    "Precision:",
    f"{new_precision:.4f}"
)

print(
    "Recall   :",
    f"{new_recall:.4f}"
)

print(
    "F1 Score :",
    f"{new_f1:.4f}"
)


# ============================================================
# SUMMARY
# ============================================================

print()

print("===== SUMMARY =====")

if difference > 0:

    print(
        "New model has higher test accuracy "
        "than the original model."
    )

elif difference < 0:

    print(
        "New model has lower test accuracy "
        "than the original model."
    )

else:

    print(
        "Both models have the same test accuracy."
    )