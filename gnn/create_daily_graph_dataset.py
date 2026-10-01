import torch
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================

input_path = "gnn/pyg_training_dataset.pt"
output_path = "gnn/daily_graph_dataset.pt"

# ============================================================
# LOAD DATASET
# ============================================================

print("\n===== LOADING PY G DATASET =====")

dataset = torch.load(
    input_path,
    weights_only=False
)

graph = dataset["graph"]
companies = dataset["companies"]
company_to_id = dataset["company_to_id"]

train_records = dataset["train"]
validation_records = dataset["validation"]
test_records = dataset["test"]

print("Companies:", len(companies))
print("Graph nodes:", graph.num_nodes)
print("Graph edges:", graph.num_edges)

print("Train records:", len(train_records))
print("Validation records:", len(validation_records))
print("Test records:", len(test_records))

# ============================================================
# GROUP RECORDS BY DATE
# ============================================================

def group_by_date(records):

    grouped = defaultdict(list)

    for record in records:

        grouped[record["date"]].append(
            record
        )

    return grouped


print("\n===== GROUPING DATA BY DATE =====")

train_by_date = group_by_date(
    train_records
)

validation_by_date = group_by_date(
    validation_records
)

test_by_date = group_by_date(
    test_records
)

print(
    "Training dates:",
    len(train_by_date)
)

print(
    "Validation dates:",
    len(validation_by_date)
)

print(
    "Test dates:",
    len(test_by_date)
)

# ============================================================
# CREATE DAILY SNAPSHOT
# ============================================================

def create_snapshots(
    grouped_records,
    split_name
):

    snapshots = []

    for date in sorted(grouped_records.keys()):

        records = grouped_records[date]

        # ----------------------------------------------------
        # Create empty feature and target tensors
        # ----------------------------------------------------

        x = torch.zeros(
            (len(companies), 2),
            dtype=torch.float
        )

        y = torch.full(
            (len(companies),),
            -1,
            dtype=torch.long
        )

        mask = torch.zeros(
            (len(companies),),
            dtype=torch.bool
        )

        # ----------------------------------------------------
        # Fill node information
        #
        # Feature 0 = Volume
        # Feature 1 = Current Return
        # ----------------------------------------------------

        for record in records:

            node_id = record["node_id"]

            x[node_id, 0] = record[
                "volume"
            ]

            x[node_id, 1] = record[
                "current_return"
            ]

            y[node_id] = record[
                "target"
            ]

            mask[node_id] = True

        # ----------------------------------------------------
        # Save daily graph snapshot
        # ----------------------------------------------------

        snapshot = {

            "date": date,

            "x": x,

            "y": y,

            "mask": mask,

            "edge_index": graph.edge_index,

            "edge_attr": graph.edge_attr
        }

        snapshots.append(
            snapshot
        )

    print(
        split_name,
        "snapshots:",
        len(snapshots)
    )

    return snapshots


# ============================================================
# CREATE TRAIN SNAPSHOTS
# ============================================================

print("\n===== CREATING TRAINING SNAPSHOTS =====")

train_snapshots = create_snapshots(
    train_by_date,
    "Training"
)

# ============================================================
# CREATE VALIDATION SNAPSHOTS
# ============================================================

print("\n===== CREATING VALIDATION SNAPSHOTS =====")

validation_snapshots = create_snapshots(
    validation_by_date,
    "Validation"
)

# ============================================================
# CREATE TEST SNAPSHOTS
# ============================================================

print("\n===== CREATING TEST SNAPSHOTS =====")

test_snapshots = create_snapshots(
    test_by_date,
    "Test"
)

# ============================================================
# CHECK FIRST SNAPSHOT
# ============================================================

print("\n===== FIRST TRAINING SNAPSHOT =====")

first_snapshot = train_snapshots[0]

print(
    "Date:",
    first_snapshot["date"]
)

print(
    "Node features:",
    first_snapshot["x"].shape
)

print(
    "Targets:",
    first_snapshot["y"].shape
)

print(
    "Mask:",
    first_snapshot["mask"].shape
)

print(
    "Edges:",
    first_snapshot["edge_index"].shape
)

# ============================================================
# COUNT AVAILABLE NODES
# ============================================================

print("\n===== NODE AVAILABILITY =====")

print(
    "First day available nodes:",
    int(
        first_snapshot["mask"].sum()
    )
)

# ============================================================
# SAVE DATASET
# ============================================================

final_dataset = {

    "graph": graph,

    "companies": companies,

    "company_to_id": company_to_id,

    "train": train_snapshots,

    "validation": validation_snapshots,

    "test": test_snapshots
}

torch.save(
    final_dataset,
    output_path
)

# ============================================================
# SUCCESS
# ============================================================

print("\n===== SUCCESS =====")

print(
    "Daily graph dataset saved successfully!"
)

print(
    "Location:",
    output_path
)

print(
    "Training snapshots:",
    len(train_snapshots)
)

print(
    "Validation snapshots:",
    len(validation_snapshots)
)

print(
    "Test snapshots:",
    len(test_snapshots)
)