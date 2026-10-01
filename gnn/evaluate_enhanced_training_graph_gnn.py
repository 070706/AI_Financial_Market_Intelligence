import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("===== DEVICE =====")
print(device)


# ============================================================
# PATHS
# ============================================================

dataset_path = "gnn/normalized_enhanced_training_daily_graph_dataset.pt"

model_path = "models/enhanced_training_only_financial_gnn.pt"

results_path = "results/enhanced_training_only_gnn_evaluation.pt"


# ============================================================
# LOAD DATASET
# ============================================================

dataset = torch.load(
    dataset_path,
    weights_only=False
)

test_data = dataset["test"]

companies = dataset["companies"]

print()
print("===== TEST DATA =====")
print("Test snapshots:", len(test_data))
print("Companies:", len(companies))


# ============================================================
# MODEL
# ============================================================

class FinancialGNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = SAGEConv(
            6,
            32
        )

        self.conv2 = SAGEConv(
            32,
            32
        )

        self.dropout = nn.Dropout(
            0.2
        )

        self.output = nn.Linear(
            32,
            2
        )

    def forward(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.output(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    model_path,
    weights_only=False
)

model = FinancialGNN().to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


print()
print("===== MODEL =====")
print(
    "Best validation accuracy:",
    checkpoint["best_validation_accuracy"]
)


# ============================================================
# PREDICTIONS
# ============================================================

all_predictions = []

all_targets = []

company_predictions = {
    ticker: []
    for ticker in companies
}

company_targets = {
    ticker: []
    for ticker in companies
}


with torch.no_grad():

    for snapshot in test_data:

        x = snapshot["x"].to(device)

        edge_index = snapshot[
            "edge_index"
        ].to(device)

        y = snapshot["y"].to(device)

        mask = snapshot["mask"].to(device)

        output = model(
            x,
            edge_index
        )

        predictions = output.argmax(
            dim=1
        )

        valid_predictions = predictions[
            mask
        ]

        valid_targets = y[
            mask
        ]

        all_predictions.extend(
            valid_predictions.cpu().tolist()
        )

        all_targets.extend(
            valid_targets.cpu().tolist()
        )

        for ticker, node_id in dataset[
            "company_to_id"
        ].items():

            if mask[node_id]:

                company_predictions[
                    ticker
                ].append(
                    predictions[node_id].item()
                )

                company_targets[
                    ticker
                ].append(
                    y[node_id].item()
                )


# ============================================================
# METRICS
# ============================================================

tp = 0
tn = 0
fp = 0
fn = 0

for prediction, target in zip(
    all_predictions,
    all_targets
):

    if prediction == 1 and target == 1:
        tp += 1

    elif prediction == 0 and target == 0:
        tn += 1

    elif prediction == 1 and target == 0:
        fp += 1

    elif prediction == 0 and target == 1:
        fn += 1


total = len(all_targets)

accuracy = (
    (tp + tn) / total
    if total > 0
    else 0
)

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0
)

recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0
)

f1 = (
    2 * precision * recall
    / (precision + recall)
    if (precision + recall) > 0
    else 0
)


# ============================================================
# OUTPUT
# ============================================================

print()
print("========================================")
print("       TEST SET EVALUATION")
print("========================================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Accuracy %: {accuracy * 100:.2f}"
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
# CONFUSION MATRIX
# ============================================================

print()
print("===== CONFUSION MATRIX =====")

print(
    "[[{} {}]".format(
        tn,
        fp
    )
)

print(
    " [{} {}]]".format(
        fn,
        tp
    )
)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

predicted_0 = all_predictions.count(0)

predicted_1 = all_predictions.count(1)

actual_0 = all_targets.count(0)

actual_1 = all_targets.count(1)

print()
print("===== PREDICTION DISTRIBUTION =====")

print(
    "Predicted 0:",
    predicted_0
)

print(
    "Predicted 1:",
    predicted_1
)

print(
    "Actual 0:",
    actual_0
)

print(
    "Actual 1:",
    actual_1
)


# ============================================================
# COMPANY-WISE ACCURACY
# ============================================================

print()
print("===== COMPANY-WISE ACCURACY =====")

company_accuracy = {}

for ticker in companies:

    predictions = company_predictions[
        ticker
    ]

    targets = company_targets[
        ticker
    ]

    if len(targets) > 0:

        correct = sum(
            p == t
            for p, t in zip(
                predictions,
                targets
            )
        )

        acc = correct / len(targets)

        company_accuracy[ticker] = acc

        print(
            f"{ticker:6s}: "
            f"{acc * 100:.2f}%"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

torch.save(
    {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": [
            [tn, fp],
            [fn, tp]
        ],
        "predictions": all_predictions,
        "targets": all_targets,
        "company_accuracy": company_accuracy
    },
    results_path
)


print()
print("===== EVALUATION COMPLETE =====")

print(
    "Results saved:"
)

print(results_path)