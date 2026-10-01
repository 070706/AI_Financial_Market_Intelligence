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


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("===== LOADING DATASET =====")

dataset = torch.load(
    dataset_path,
    weights_only=False
)

train_data = dataset["train"]
validation_data = dataset["validation"]

print("Training snapshots:", len(train_data))
print("Validation snapshots:", len(validation_data))


# ============================================================
# MODEL
# ============================================================

class FinancialGNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = SAGEConv(
            in_channels=6,
            out_channels=32
        )

        self.conv2 = SAGEConv(
            in_channels=32,
            out_channels=32
        )

        self.dropout = nn.Dropout(
            p=0.2
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
# CREATE MODEL
# ============================================================

model = FinancialGNN().to(device)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print()
print("===== CALCULATING CLASS WEIGHTS =====")

class_counts = torch.zeros(
    2,
    dtype=torch.float,
    device=device
)

for snapshot in train_data:

    mask = snapshot["mask"]

    y = snapshot["y"]

    valid_y = y[mask]

    class_counts[0] += (
        valid_y == 0
    ).sum()

    class_counts[1] += (
        valid_y == 1
    ).sum()


class_weights = class_counts.sum() / (
    2 * class_counts
)

print(
    "Class 0:",
    int(class_counts[0].item())
)

print(
    "Class 1:",
    int(class_counts[1].item())
)

print(
    "Class weights:",
    class_weights.detach().cpu().tolist()
)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate():

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for snapshot in validation_data:

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

            correct += (
                predictions[mask] == y[mask]
            ).sum().item()

            total += mask.sum().item()

    return correct / total


# ============================================================
# TRAINING
# ============================================================

print()
print("===== TRAINING =====")

epochs = 30

best_validation_accuracy = 0.0

for epoch in range(1, epochs + 1):

    model.train()

    total_loss = 0.0

    for snapshot in train_data:

        x = snapshot["x"].to(device)

        edge_index = snapshot[
            "edge_index"
        ].to(device)

        y = snapshot["y"].to(device)

        mask = snapshot["mask"].to(device)

        optimizer.zero_grad()

        output = model(
            x,
            edge_index
        )

        loss = criterion(
            output[mask],
            y[mask]
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    validation_accuracy = evaluate()

    average_loss = (
        total_loss / len(train_data)
    )

    print(
        f"Epoch {epoch:02d} | "
        f"Loss: {average_loss:.4f} | "
        f"Validation Accuracy: "
        f"{validation_accuracy:.4f}"
    )

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = (
            validation_accuracy
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "best_validation_accuracy":
                    best_validation_accuracy,

                "feature_names":
                    dataset["feature_names"],

                "companies":
                    dataset["companies"]
            },
            model_path
        )


# ============================================================
# FINAL
# ============================================================

print()
print("===== TRAINING COMPLETE =====")

print(
    "Best validation accuracy:",
    best_validation_accuracy
)

print()
print("Model saved:")

print(model_path)