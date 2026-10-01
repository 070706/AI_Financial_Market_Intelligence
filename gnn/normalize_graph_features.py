import torch

# ============================================================
# PATHS
# ============================================================

input_path = "gnn/daily_graph_dataset.pt"
output_path = "gnn/normalized_daily_graph_dataset.pt"

# ============================================================
# LOAD DATASET
# ============================================================

print("\n===== LOADING DAILY GRAPH DATASET =====")

dataset = torch.load(
    input_path,
    weights_only=False
)

train = dataset["train"]
validation = dataset["validation"]
test = dataset["test"]

print("Training snapshots:", len(train))
print("Validation snapshots:", len(validation))
print("Test snapshots:", len(test))

# ============================================================
# CALCULATE TRAINING STATISTICS
# ============================================================

print("\n===== CALCULATING TRAINING STATISTICS =====")

volume_values = []
return_values = []

for snapshot in train:

    x = snapshot["x"]
    mask = snapshot["mask"]

    # Only use available nodes
    valid_x = x[mask]

    if valid_x.numel() == 0:
        continue

    volume_values.append(
        valid_x[:, 0]
    )

    return_values.append(
        valid_x[:, 1]
    )

volume_values = torch.cat(
    volume_values
)

return_values = torch.cat(
    return_values
)

volume_mean = volume_values.mean()
volume_std = volume_values.std()

return_mean = return_values.mean()
return_std = return_values.std()

# Safety against division by zero
if volume_std == 0:
    volume_std = torch.tensor(1.0)

if return_std == 0:
    return_std = torch.tensor(1.0)

print("\n===== TRAINING STATISTICS =====")

print(
    "Volume mean:",
    volume_mean.item()
)

print(
    "Volume std:",
    volume_std.item()
)

print(
    "Current return mean:",
    return_mean.item()
)

print(
    "Current return std:",
    return_std.item()
)

# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def normalize_snapshots(
    snapshots,
    split_name
):

    normalized = []

    for snapshot in snapshots:

        x = snapshot["x"].clone()

        mask = snapshot["mask"]

        # Normalize only available nodes
        x[mask, 0] = (
            x[mask, 0] - volume_mean
        ) / volume_std

        x[mask, 1] = (
            x[mask, 1] - return_mean
        ) / return_std

        normalized_snapshot = {
            "date": snapshot["date"],
            "x": x,
            "y": snapshot["y"],
            "mask": snapshot["mask"],
            "edge_index": snapshot["edge_index"],
            "edge_attr": snapshot["edge_attr"]
        }

        normalized.append(
            normalized_snapshot
        )

    print(
        split_name,
        "snapshots normalized:",
        len(normalized)
    )

    return normalized

# ============================================================
# NORMALIZE TRAINING DATA
# ============================================================

print("\n===== NORMALIZING TRAINING DATA =====")

normalized_train = normalize_snapshots(
    train,
    "Training"
)

# ============================================================
# NORMALIZE VALIDATION DATA
# ============================================================

print("\n===== NORMALIZING VALIDATION DATA =====")

normalized_validation = normalize_snapshots(
    validation,
    "Validation"
)

# ============================================================
# NORMALIZE TEST DATA
# ============================================================

print("\n===== NORMALIZING TEST DATA =====")

normalized_test = normalize_snapshots(
    test,
    "Test"
)

# ============================================================
# CHECK FIRST SNAPSHOT
# ============================================================

print("\n===== NORMALIZED SAMPLE =====")

sample = normalized_train[0]

print(
    "Date:",
    sample["date"]
)

print(
    "X shape:",
    sample["x"].shape
)

print(
    "Y shape:",
    sample["y"].shape
)

print(
    "Mask shape:",
    sample["mask"].shape
)

print(
    "Edge index:",
    sample["edge_index"].shape
)

print(
    "Edge attributes:",
    sample["edge_attr"].shape
)

# ============================================================
# FINAL DATASET
# ============================================================

normalized_dataset = {

    "graph": dataset["graph"],

    "companies": dataset["companies"],

    "company_to_id": dataset["company_to_id"],

    "train": normalized_train,

    "validation": normalized_validation,

    "test": normalized_test,

    "normalization": {
        "volume_mean": volume_mean,
        "volume_std": volume_std,
        "return_mean": return_mean,
        "return_std": return_std
    }
}

# ============================================================
# SAVE
# ============================================================

torch.save(
    normalized_dataset,
    output_path
)

print("\n===== SUCCESS =====")

print(
    "Normalized dataset saved successfully!"
)

print(
    "Location:",
    output_path
)