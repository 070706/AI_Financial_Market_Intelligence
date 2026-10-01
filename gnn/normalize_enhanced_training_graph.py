import torch


# ============================================================
# PATHS
# ============================================================

input_path = "gnn/enhanced_training_daily_graph_dataset.pt"

output_path = "gnn/normalized_enhanced_training_daily_graph_dataset.pt"


# ============================================================
# LOAD DATA
# ============================================================

print("===== LOADING ENHANCED GRAPH DATASET =====")

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

print()
print("===== CALCULATING TRAINING STATISTICS =====")

all_train_features = []

for snapshot in train:

    x = snapshot["x"]
    mask = snapshot["mask"]

    valid_x = x[mask]

    all_train_features.append(valid_x)


all_train_features = torch.cat(
    all_train_features,
    dim=0
)


mean = all_train_features.mean(
    dim=0
)

std = all_train_features.std(
    dim=0
)


# Prevent division by zero

std = torch.where(
    std == 0,
    torch.ones_like(std),
    std
)


feature_names = [
    "Volume",
    "Current_Return",
    "Volatility",
    "Momentum",
    "Moving_Average",
    "Volume_Change"
]


print()
print("===== TRAINING FEATURE STATISTICS =====")

for i, name in enumerate(feature_names):

    print(
        f"{name:18s} "
        f"Mean: {mean[i].item():.6f} "
        f"Std: {std[i].item():.6f}"
    )


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def normalize_snapshots(snapshots):

    normalized = []

    for snapshot in snapshots:

        new_snapshot = snapshot.copy()

        x = snapshot["x"].clone()

        mask = snapshot["mask"]

        x[mask] = (
            x[mask] - mean
        ) / std

        new_snapshot["x"] = x

        normalized.append(
            new_snapshot
        )

    return normalized


# ============================================================
# NORMALIZE ALL SPLITS
# ============================================================

print()
print("===== NORMALIZING DATA =====")

normalized_train = normalize_snapshots(
    train
)

normalized_validation = normalize_snapshots(
    validation
)

normalized_test = normalize_snapshots(
    test
)


# ============================================================
# CHECK
# ============================================================

print()
print("===== NORMALIZATION CHECK =====")

first = normalized_train[0]

print(
    "Date:",
    first["date"]
)

print(
    "X shape:",
    first["x"].shape
)

print(
    "Y shape:",
    first["y"].shape
)

print(
    "Mask shape:",
    first["mask"].shape
)

print(
    "Edge index shape:",
    first["edge_index"].shape
)

print(
    "Edge attributes shape:",
    first["edge_attr"].shape
)


# ============================================================
# SAVE
# ============================================================

torch.save(
    {
        "graph": dataset["graph"],
        "companies": dataset["companies"],
        "company_to_id": dataset["company_to_id"],
        "train": normalized_train,
        "validation": normalized_validation,
        "test": normalized_test,
        "mean": mean,
        "std": std,
        "feature_names": feature_names
    },
    output_path
)


print()
print("===== SUCCESS =====")

print(
    "Normalized enhanced dataset saved:"
)

print(output_path)