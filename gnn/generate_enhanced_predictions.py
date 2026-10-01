import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


# ============================================================
# PATHS
# ============================================================

dataset_path = "gnn/normalized_enhanced_training_daily_graph_dataset.pt"

model_path = "models/enhanced_training_only_financial_gnn.pt"

output_path = "results/enhanced_gnn_predictions.pt"


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

print()
print("===== LOADING DATASET =====")

dataset = torch.load(
    dataset_path,
    weights_only=False
)

test_data = dataset["test"]

companies = dataset["companies"]

company_to_id = dataset["company_to_id"]

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

print()
print("===== LOADING MODEL =====")

checkpoint = torch.load(
    model_path,
    weights_only=False
)

model = FinancialGNN().to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Best validation accuracy:",
    checkpoint["best_validation_accuracy"]
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print()
print("===== GENERATING PREDICTIONS =====")

predictions = []

with torch.no_grad():

    for snapshot in test_data:

        x = snapshot["x"].to(device)

        edge_index = snapshot[
            "edge_index"
        ].to(device)

        mask = snapshot["mask"].to(device)

        output = model(
            x,
            edge_index
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = output.argmax(
            dim=1
        )

        date = snapshot["date"]

        for ticker, node_id in company_to_id.items():

            if mask[node_id]:

                predictions.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "prediction": int(
                            predicted_class[node_id].item()
                        ),
                        "probability_down": float(
                            probabilities[node_id, 0].item()
                        ),
                        "probability_up": float(
                            probabilities[node_id, 1].item()
                        )
                    }
                )


# ============================================================
# DISPLAY SAMPLE
# ============================================================

print()
print("===== SAMPLE PREDICTIONS =====")

for row in predictions[:10]:

    print(
        row["date"],
        "|",
        row["ticker"],
        "| Prediction:",
        row["prediction"],
        "| Down:",
        round(row["probability_down"], 4),
        "| Up:",
        round(row["probability_up"], 4)
    )


# ============================================================
# DISTRIBUTION
# ============================================================

prediction_0 = sum(
    row["prediction"] == 0
    for row in predictions
)

prediction_1 = sum(
    row["prediction"] == 1
    for row in predictions
)

print()
print("===== PREDICTION DISTRIBUTION =====")

print(
    "Prediction 0:",
    prediction_0
)

print(
    "Prediction 1:",
    prediction_1
)

print(
    "Total predictions:",
    len(predictions)
)


# ============================================================
# SAVE
# ============================================================

torch.save(
    {
        "predictions": predictions,
        "companies": companies,
        "best_validation_accuracy":
            checkpoint["best_validation_accuracy"]
    },
    output_path
)


# ============================================================
# SUCCESS
# ============================================================

print()
print("===== SUCCESS =====")

print(
    "Predictions saved:"
)

print(output_path)