import torch


# ============================================================
# PATHS
# ============================================================

input_path = "gnn/training_daily_graph_dataset.pt"

output_path = "gnn/normalized_training_daily_graph_dataset.pt"


# ============================================================
# LOAD DATASET
# ============================================================

print("===== LOADING DATASET =====")

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
# CALCULATE TRAINING-ONLY STATISTICS
# ============================================================

print()
print("===== CALCULATING TRAINING STATISTICS =====")

volume_values = []
return_values = []

for snapshot in train:

    x = snapshot["x"]
    mask = snapshot["mask"]

    volume_values.append(
        x[mask, 0]
    )

    return_values.append(
        x[mask, 1]
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


print("Volume mean:", volume_mean.item())
print("Volume std:", volume_std.item())

print("Return mean:", return_mean.item())
print("Return std:", return_std.item())


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def normalize_snapshots(snapshots):

    normalized = []

    for snapshot in snapshots:

        new_snapshot = snapshot.copy()

        x = snapshot["x"].clone()

        mask = snapshot["mask"]

        # Volume
        x[mask, 0] = (
            x[mask, 0] - volume_mean
        ) / (volume_std + 1e-8)

        # Current return
        x[mask, 1] = (
            x[mask, 1] - return_mean
        ) / (return_std + 1e-8)

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
# SAVE
# ============================================================

torch.save(
    {
        "graph": dataset["graph"],

        "companies": dataset["companies"],

        "company_to_id":
            dataset["company_to_id"],

        "train": normalized_train,

        "validation": normalized_validation,

        "test": normalized_test,

        "normalization": {
            "volume_mean": volume_mean,

            "volume_std": volume_std,

            "return_mean": return_mean,

            "return_std": return_std
        }
    },
    output_path
)


print()
print("===== SUCCESS =====")

print(
    "Normalized dataset saved:"
)

print(output_path)