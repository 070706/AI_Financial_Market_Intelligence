import torch
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
# LOAD DATASET
# ============================================================

dataset = torch.load(
    "gnn/normalized_training_daily_graph_dataset.pt",
    weights_only=False
)

train = dataset["train"]
validation = dataset["validation"]

print()
print("===== DATASET =====")
print("Training snapshots:", len(train))
print("Validation snapshots:", len(validation))


# ============================================================
# MODEL
# ============================================================

class FinancialGraphSAGE(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = SAGEConv(2, 32)
        self.conv2 = SAGEConv(32, 32)
        self.linear = torch.nn.Linear(32, 2)

    def forward(self, x, edge_index):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=0.2,
            training=self.training
        )

        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.linear(x)

        return x


# ============================================================
# CREATE MODEL
# ============================================================

model = FinancialGraphSAGE().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# CLASS-WEIGHTED LOSS
# ============================================================

# Training target counts
class_counts = torch.tensor(
    [
        514048,
        534161
    ],
    dtype=torch.float,
    device=device
)

class_weights = class_counts.sum() / (
    2 * class_counts
)

print()
print("===== CLASS WEIGHTS =====")
print("Class 0 weight:", class_weights[0].item())
print("Class 1 weight:", class_weights[1].item())


criterion = torch.nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

epochs = 30

best_validation_accuracy = 0.0

best_model_state = None


# ============================================================
# TRAINING
# ============================================================

print()
print("===== TRAINING =====")

for epoch in range(1, epochs + 1):

    model.train()

    total_loss = 0.0

    for snapshot in train:

        x = snapshot["x"].to(device)

        y = snapshot["y"].to(device)

        mask = snapshot["mask"].to(device)

        edge_index = snapshot["edge_index"].to(device)

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


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    correct = 0

    total = 0

    with torch.no_grad():

        for snapshot in validation:

            x = snapshot["x"].to(device)

            y = snapshot["y"].to(device)

            mask = snapshot["mask"].to(device)

            edge_index = snapshot["edge_index"].to(device)

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


    validation_accuracy = correct / total

    average_loss = total_loss / len(train)


    print(
        f"Epoch {epoch:02d} | "
        f"Loss: {average_loss:.4f} | "
        f"Validation Accuracy: {validation_accuracy:.4f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = validation_accuracy

        best_model_state = {
            "model_state_dict": model.state_dict(),
            "validation_accuracy": validation_accuracy
        }


# ============================================================
# SAVE MODEL
# ============================================================

output_model = (
    "models/training_only_financial_gnn.pt"
)

torch.save(
    best_model_state,
    output_model
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("===== TRAINING COMPLETE =====")

print(
    "Best validation accuracy:",
    best_validation_accuracy
)

print()
print("Model saved:")
print(output_model)