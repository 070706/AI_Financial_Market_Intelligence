import torch
import numpy as np

from torch_geometric.nn import SAGEConv
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("===== DEVICE =====")
print(device)


# ============================================================
# LOAD DATASET
# ============================================================

dataset = torch.load(
    "gnn/normalized_training_daily_graph_dataset.pt",
    weights_only=False
)

test = dataset["test"]

companies = dataset["companies"]


print()
print("===== TEST DATA =====")
print("Test snapshots:", len(test))
print("Companies:", len(companies))


# ============================================================
# MODEL
# ============================================================

class FinancialGraphSAGE(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = SAGEConv(
            2,
            32
        )

        self.conv2 = SAGEConv(
            32,
            32
        )

        self.linear = torch.nn.Linear(
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

        x = torch.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.linear(x)

        return x


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = FinancialGraphSAGE().to(device)

checkpoint = torch.load(
    "models/training_only_financial_gnn.pt",
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


print()
print("===== MODEL =====")

print(
    "Best validation accuracy:",
    checkpoint["validation_accuracy"]
)


# ============================================================
# COLLECT PREDICTIONS
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

    for snapshot in test:

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

            node_id = node_id.item()

            prediction = predictions[
                node_id
            ].item()

            target = y[
                node_id
            ].item()

            ticker = companies[
                node_id
            ]


            all_predictions.append(
                prediction
            )

            all_targets.append(
                target
            )


            company_predictions[
                ticker
            ].append(
                prediction
            )

            company_targets[
                ticker
            ].append(
                target
            )


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

confusion = confusion_matrix(
    all_targets,
    all_predictions
)


print()
print("========================================")
print("       TEST SET EVALUATION")
print("========================================")

print(
    "Accuracy :",
    f"{accuracy:.4f}"
)

print(
    "Accuracy %:",
    f"{accuracy * 100:.2f}"
)

print(
    "Precision:",
    f"{precision:.4f}"
)

print(
    "Recall   :",
    f"{recall:.4f}"
)

print(
    "F1 Score :",
    f"{f1:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("===== CONFUSION MATRIX =====")

print(confusion)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("===== PREDICTION DISTRIBUTION =====")

print(
    "Predicted 0:",
    all_predictions.count(0)
)

print(
    "Predicted 1:",
    all_predictions.count(1)
)

print(
    "Actual 0:",
    all_targets.count(0)
)

print(
    "Actual 1:",
    all_targets.count(1)
)


# ============================================================
# COMPANY-WISE ACCURACY
# ============================================================

print()
print("===== COMPANY-WISE ACCURACY =====")

company_results = {}

for ticker in companies:

    targets = company_targets[ticker]

    predictions = company_predictions[ticker]

    if len(targets) == 0:

        continue

    company_accuracy = accuracy_score(
        targets,
        predictions
    )

    company_results[ticker] = (
        company_accuracy
    )

    print(
        f"{ticker:6s} : "
        f"{company_accuracy * 100:.2f}% "
        f"({len(targets)} samples)"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1": f1,

    "confusion_matrix": confusion,

    "predictions": all_predictions,

    "targets": all_targets,

    "company_accuracy": company_results
}


torch.save(
    results,
    "results/training_only_gnn_evaluation.pt"
)


print()
print("===== EVALUATION COMPLETE =====")

print(
    "Results saved:"
)

print(
    "results/training_only_gnn_evaluation.pt"
)