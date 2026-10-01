import torch
import csv
import glob
from collections import defaultdict
from datetime import datetime


# ============================================================
# PATHS
# ============================================================

graph_path = "gnn/final_training_graph.pt"

feature_path = "data/processed/enhanced_gnn_features"

output_path = "gnn/enhanced_training_daily_graph_dataset.pt"


# ============================================================
# LOAD TRAINING GRAPH
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
# READ ENHANCED FEATURES
# ============================================================

print()
print("===== READING ENHANCED FEATURES =====")

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

            if ticker not in company_to_id:
                continue

            records.append(
                {
                    "Ticker": ticker,
                    "Date": row["Date"],
                    "Volume": float(row["Volume"]),
                    "Current_Return": float(
                        row["Current_Return"]
                    ),
                    "Volatility": float(
                        row["Volatility"]
                    ),
                    "Momentum": float(
                        row["Momentum"]
                    ),
                    "Moving_Average": float(
                        row["Moving_Average"]
                    ),
                    "Volume_Change": float(
                        row["Volume_Change"]
                    ),
                    "Target": int(row["Target"])
                }
            )


print(
    "Records for training graph:",
    len(records)
)

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

    print(
        "All 26 graph companies have data."
    )


# ============================================================
# GROUP BY DATE
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
    key=lambda x:
        datetime.strptime(
            x,
            "%Y-%m-%d"
        )
)

print(
    "Total dates:",
    len(dates)
)

if len(dates) > 0:

    print(
        "First date:",
        dates[0]
    )

    print(
        "Last date:",
        dates[-1]
    )


# ============================================================
# DATE SPLIT
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
# CREATE SNAPSHOT
# ============================================================

def create_snapshot(date):

    # 6 node features
    x = torch.zeros(
        (len(companies), 6),
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


        # ----------------------------------------------------
        # NODE FEATURES
        # ----------------------------------------------------

        x[node_id, 0] = record["Volume"]

        x[node_id, 1] = record[
            "Current_Return"
        ]

        x[node_id, 2] = record[
            "Volatility"
        ]

        x[node_id, 3] = record[
            "Momentum"
        ]

        x[node_id, 4] = record[
            "Moving_Average"
        ]

        x[node_id, 5] = record[
            "Volume_Change"
        ]


        # ----------------------------------------------------
        # TARGET
        # ----------------------------------------------------

        y[node_id] = record["Target"]

        mask[node_id] = True


    return {
        "date": date,
        "x": x,
        "y": y,
        "mask": mask,
        "edge_index": graph.edge_index.clone(),
        "edge_attr": graph.edge_attr.clone()
    }


# ============================================================
# CREATE SNAPSHOTS
# ============================================================

print()
print("===== CREATING ENHANCED SNAPSHOTS =====")


train_snapshots = [
    create_snapshot(date)
    for date in train_dates
]


validation_snapshots = [
    create_snapshot(date)
    for date in validation_dates
]


test_snapshots = [
    create_snapshot(date)
    for date in test_dates
]


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
# CHECK FIRST SNAPSHOT
# ============================================================

if len(train_snapshots) == 0:

    raise RuntimeError(
        "No training snapshots were created."
    )


first = train_snapshots[0]


print()
print(
    "===== FIRST ENHANCED TRAINING SNAPSHOT ====="
)

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
    "Enhanced training daily graph dataset saved:"
)

print(output_path)