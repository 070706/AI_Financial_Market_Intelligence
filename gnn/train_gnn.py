import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv

# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "gnn/normalized_daily_graph_dataset.pt"
MODEL_PATH = "models/financial_gnn.pt"

EPOCHS = 30
LEARNING_RATE = 0.001
HIDDEN_CHANNELS = 32

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

print("\n===== LOADING DATASET =====")

dataset = torch.load(
    DATA_PATH,
    weights_only=False
)

train_data = dataset["train"]
validation_data = dataset["validation"]
test_data = dataset["test"]

companies = dataset["companies"]

print(
    "Companies:",
    len(companies)
)

print(
    "Training snapshots:",
    len(train_data)
)

print(
    "Validation snapshots:",
    len(validation_data)
)

print(
    "Test snapshots:",
    len(test_data)
)

# ============================================================
# GNN MODEL
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

        x = self.output(x)

        return x


# ============================================================
# CREATE MODEL
# ============================================================

model = FinancialGraphSAGE(
    input_channels=2,
    hidden_channels=HIDDEN_CHANNELS
)

model = model.to(device)

print("\n===== MODEL =====")
print(model)

# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    model,
    snapshots
):

    model.eval()

    correct = 0
    total = 0

    total_loss = 0.0

    with torch.no_grad():

        for snapshot in snapshots:

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

            valid_output = output[mask]

            valid_target = y[mask]

            loss = criterion(
                valid_output,
                valid_target
            )

            total_loss += loss.item()

            prediction = valid_output.argmax(
                dim=1
            )

            correct += (
                prediction == valid_target
            ).sum().item()

            total += valid_target.numel()

    accuracy = correct / total

    average_loss = (
        total_loss / len(snapshots)
    )

    return average_loss, accuracy


# ============================================================
# TRAINING
# ============================================================

print("\n===== STARTING TRAINING =====")

best_validation_accuracy = 0.0

for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    total_loss = 0.0

    correct = 0
    total = 0

    for snapshot in train_data:

        x = snapshot["x"].to(device)

        y = snapshot["y"].to(device)

        mask = snapshot["mask"].to(device)

        edge_index = snapshot[
            "edge_index"
        ].to(device)

        optimizer.zero_grad()

        output = model(
            x,
            edge_index
        )

        valid_output = output[mask]

        valid_target = y[mask]

        loss = criterion(
            valid_output,
            valid_target
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        prediction = valid_output.argmax(
            dim=1
        )

        correct += (
            prediction == valid_target
        ).sum().item()

        total += valid_target.numel()

    train_accuracy = (
        correct / total
    )

    train_loss = (
        total_loss / len(train_data)
    )

    validation_loss, validation_accuracy = evaluate(
        model,
        validation_data
    )

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f} | "
        f"Validation Loss: {validation_loss:.4f} | "
        f"Validation Accuracy: {validation_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = validation_accuracy

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "input_channels":
                    2,

                "hidden_channels":
                    HIDDEN_CHANNELS,

                "validation_accuracy":
                    validation_accuracy,

                "companies":
                    companies
            },
            MODEL_PATH
        )

        print(
            "  Best model saved."
        )

# ============================================================
# FINAL TEST
# ============================================================

print("\n===== LOADING BEST MODEL =====")

checkpoint = torch.load(
    MODEL_PATH,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

test_loss, test_accuracy = evaluate(
    model,
    test_data
)

print("\n===== FINAL TEST RESULT =====")

print(
    "Test Loss:",
    round(test_loss, 4)
)

print(
    "Test Accuracy:",
    round(test_accuracy, 4)
)

print(
    "Test Accuracy (%):",
    round(
        test_accuracy * 100,
        2
    )
)

print(
    "\nBest Validation Accuracy:",
    round(
        checkpoint["validation_accuracy"],
        4
    )
)

print(
    "\n===== SUCCESS ====="
)

print(
    "GNN training completed!"
)

print(
    "Model:",
    MODEL_PATH
)