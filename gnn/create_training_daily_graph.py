import torch
import csv
import glob
from collections import defaultdict
from datetime import datetime


# ============================================================
# PATHS
# ============================================================

graph_path = "gnn/final_training_graph.pt"

# IMPORTANT:
# Use gnn_features, NOT gnn_training_final.
# gnn_features contains all companies, including CINF and DLR.
feature_path = "data/processed/gnn_features"

output_path = "gnn/training_daily_graph_dataset.pt"


# ============================================================
# LOAD FINAL TRAINING GRAPH
# ============================================================

print("===== LOADING TRAINING GRAPH =====")

checkpoint = torch.load(
    graph_path,
    weights_only=False
)

graph = checkpoint["data"]
companies = checkpoint["companies"]
company_to_id = checkpoint["company_to_id"]

print("Companies:", len(companies))
print("Nodes:", graph.num_nodes)
print("Edges:", graph.num_edges)


# ============================================================
# READ GNN FEATURES
# ============================================================

print()
print("===== READING GNN FEATURES =====")

files = glob.glob(
    feature_path + "/part-*.csv"
)

print("Feature files:", len(files))

records = []

for file in files:

    with open(
        file,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            ticker = row["Ticker"]

            # Keep only companies present
            # in the new training graph.
            if ticker not in company_to_id:
                continue

            date = row["Date"]

            volume = float(row["Volume"])
            current_return = float(row["Current_Return"])
            target = int(row["Target"])

            records.append(
                {
                    "Ticker": ticker,
                    "Date": date,
                    "Volume": volume,
                    "Current_Return": current_return,
                    "Target": target
                }
            )


print("Records for training graph:", len(records))


# ============================================================
# CHECK COMPANIES
# ============================================================

companies_found = sorted(
    set(
        record["Ticker"]
        for record in records
    )
)

print(
    "Companies found:",
    len(companies_found)
)

missing = [
    ticker
    for ticker in companies
    if ticker not in companies_found
]

if missing:

    print()
    print("WARNING - Missing companies:")

    for ticker in missing:
        print(ticker)

else:

    print("All 26 graph companies have data.")


# ============================================================
# GROUP RECORDS BY DATE
# ============================================================

print()
print("===== GROUPING BY DATE =====")

daily_records = defaultdict(list)

for record in records:

    daily_records[
        record["Date"]
    ].append(record)


dates = sorted(
    daily_records.keys(),
    key=lambda x: datetime.strptime(
        x,
        "%Y-%m-%d"
    )
)


print("Total dates:", len(dates))

if len(dates) > 0:

    print("First date:", dates[0])
    print("Last date:", dates[-1])


# ============================================================
# DATE SPLITS
# ============================================================

TRAIN_END = "2018-04-17"
VALIDATION_END = "2022-03-16"


train_dates = [
    date
    for date in dates
    if date < TRAIN_END
]

validation_dates = [
    date
    for date in dates
    if TRAIN_END <= date < VALIDATION_END
]

test_dates = [
    date
    for date in dates
    if date >= VALIDATION_END
]


print()
print("===== DATE SPLIT =====")

print(
    "Training dates:",
    len(train_dates)
)

print(
    "Validation dates:",
    len(validation_dates)
)

print(
    "Test dates:",
    len(test_dates)
)


# ============================================================
# CREATE SNAPSHOT FUNCTION
# ============================================================

def create_snapshot(date):

    x = torch.zeros(
        (len(companies), 2),
        dtype=torch.float
    )

    y = torch.zeros(
        len(companies),
        dtype=torch.long
    )

    mask = torch.zeros(
        len(companies),
        dtype=torch.bool
    )

    for record in daily_records[date]:

        ticker = record["Ticker"]

        node_id = company_to_id[ticker]

        x[node_id, 0] = record["Volume"]

        x[node_id, 1] = record["Current_Return"]

        y[node_id] = record["Target"]

        mask[node_id] = True

    snapshot = {
        "date": date,

        "x": x,

        "y": y,

        "mask": mask,

        "edge_index": graph.edge_index.clone(),

        "edge_attr": graph.edge_attr.clone()
    }

    return snapshot


# ============================================================
# CREATE DATASET SPLITS
# ============================================================

print()
print("===== CREATING SNAPSHOTS =====")


train_snapshots = []

for date in train_dates:

    train_snapshots.append(
        create_snapshot(date)
    )


validation_snapshots = []

for date in validation_dates:

    validation_snapshots.append(
        create_snapshot(date)
    )


test_snapshots = []

for date in test_dates:

    test_snapshots.append(
        create_snapshot(date)
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


# ============================================================
# FIRST SNAPSHOT CHECK
# ============================================================

if len(train_snapshots) == 0:

    raise RuntimeError(
        "No training snapshots were created."
    )


first = train_snapshots[0]

print()
print("===== FIRST TRAINING SNAPSHOT =====")

print(
    "Date:",
    first["date"]
)

print(
    "Available nodes:",
    first["mask"].sum().item()
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
        "graph": graph,

        "companies": companies,

        "company_to_id": company_to_id,

        "train": train_snapshots,

        "validation": validation_snapshots,

        "test": test_snapshots
    },
    output_path
)


print()
print("===== SUCCESS =====")

print(
    "Training daily graph dataset saved:"
)

print(output_path)