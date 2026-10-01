import torch
import torch.nn as nn

from torch_geometric.nn import SAGEConv
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# PATHS
# ============================================================

DATA_PATH = "gnn/normalized_daily_graph_dataset.pt"
MODEL_PATH = "models/financial_gnn.pt"

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n===== DEVICE =====")
print(device)

# ============================================================
# LOAD DATA
# ============================================================

dataset = torch.load(
    DATA_PATH,
    weights_only=False
)

test_data = dataset["test"]
companies = dataset["companies"]

# ============================================================
# MODEL
# ============================================================

class FinancialGraphSAGE(nn.Module):

    def __init__(
        self,
        input_channels,
        hidden_channels
    ):

        super().__init__()

        self.conv1 = SAGEConv(
            input_channels,
            hidden_channels
        )

        self.conv2 = SAGEConv(
            hidden_channels,
            hidden_channels
        )

        self.output = nn.Linear(
            hidden_channels,
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

        x = torch.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.output(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    weights_only=False
)

model = FinancialGraphSAGE(
    checkpoint["input_channels"],
    checkpoint["hidden_channels"]
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print("\n===== MODEL LOADED =====")

print(
    "Best validation accuracy:",
    round(
        checkpoint["validation_accuracy"],
        4
    )
)

# ============================================================
# PREDICTIONS
# ============================================================

all_targets = []
all_predictions = []

company_targets = {
    ticker: []
    for ticker in companies
}

company_predictions = {
    ticker: []
    for ticker in companies
}

print("\n===== RUNNING TEST SET =====")

with torch.no_grad():

    for snapshot in test_data:

        x = snapshot["x"].to(device)

        y = snapshot["y"].to(device)

        mask = snapshot["mask"].to(device)

        edge_index = snapshot[
            "edge_index"
        ].to(device)

        output = model(
            x,
            edge_index
        )

        predictions = output.argmax(
            dim=1
        )

        valid_nodes = torch.where(
            mask
        )[0]

        for node_id in valid_nodes:

            node_id = int(
                node_id.item()
            )

            target = int(
                y[node_id].item()
            )

            prediction = int(
                predictions[node_id].item()
            )

            ticker = companies[
                node_id
            ]

            all_targets.append(
                target
            )

            all_predictions.append(
                prediction
            )

            company_targets[
                ticker
            ].append(target)

            company_predictions[
                ticker
            ].append(prediction)

# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    all_targets,
    all_predictions
)

precision = precision_score(
    all_targets,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_targets,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_targets,
    all_predictions,
    zero_division=0
)

cm = confusion_matrix(
    all_targets,
    all_predictions
)

# ============================================================
# RESULTS
# ============================================================

print("\n===== FINAL TEST METRICS =====")

print(
    "Accuracy :",
    round(accuracy, 4)
)

print(
    "Accuracy %:",
    round(
        accuracy * 100,
        2
    )
)

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall   :",
    round(recall, 4)
)

print(
    "F1 Score :",
    round(f1, 4)
)

print("\n===== CONFUSION MATRIX =====")

print(cm)

# ============================================================
# COMPANY-WISE PERFORMANCE
# ============================================================

print("\n===== COMPANY-WISE ACCURACY =====")

company_results = []

for ticker in companies:

    targets = company_targets[ticker]

    predictions = company_predictions[ticker]

    if len(targets) == 0:

        continue

    company_accuracy = accuracy_score(
        targets,
        predictions
    )

    company_results.append(
        (
            ticker,
            len(targets),
            company_accuracy
        )
    )

company_results.sort(
    key=lambda x: x[2],
    reverse=True
)

for ticker, count, acc in company_results:

    print(
        f"{ticker:6s} "
        f"Samples: {count:5d} "
        f"Accuracy: {acc * 100:.2f}%"
    )

# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1": f1,

    "confusion_matrix": cm,

    "company_results": company_results
}

torch.save(
    results,
    "results/gnn_evaluation.pt"
)

print("\n===== SUCCESS =====")

print(
    "Evaluation saved to:"
)

print(
    "results/gnn_evaluation.pt"
)