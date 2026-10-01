import torch
import csv
import glob

# ============================================================
# PATHS
# ============================================================

graph_path = "gnn/training_financial_graph.pt"

feature_path = "data/graph/training_node_features"

output_path = "gnn/final_training_graph.pt"

# ============================================================
# LOAD GRAPH
# ============================================================

checkpoint = torch.load(
    graph_path,
    weights_only=False
)

data = checkpoint["data"]
companies = checkpoint["companies"]
company_to_id = checkpoint["company_to_id"]

print("\n===== GRAPH =====")
print("Companies:", len(companies))
print("Edges:", data.num_edges)

# ============================================================
# LOAD FEATURES
# ============================================================

features = {}

files = glob.glob(
    feature_path + "/*.csv"
)

for file in files:

    with open(
        file,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            ticker = row["Ticker"]

            features[ticker] = [
                float(row["Average_Price"]),
                float(row["Price_Volatility"]),
                float(row["Average_Volume"])
            ]

print(
    "Companies with features:",
    len(features)
)

# ============================================================
# ALIGN FEATURES WITH GRAPH
# ============================================================

node_features = []

missing = []

for ticker in companies:

    if ticker in features:

        node_features.append(
            features[ticker]
        )

    else:

        missing.append(ticker)

        node_features.append(
            [0.0, 0.0, 0.0]
        )

# ============================================================
# CREATE FEATURE MATRIX
# ============================================================

data.x = torch.tensor(
    node_features,
    dtype=torch.float
)

# ============================================================
# DISPLAY
# ============================================================

print("\n===== FINAL TRAINING GRAPH =====")

print(
    "Nodes:",
    data.num_nodes
)

print(
    "Edges:",
    data.num_edges
)

print(
    "Features:",
    data.x.shape
)

if missing:

    print("\nMissing companies:")

    for ticker in missing:
        print(ticker)

else:

    print(
        "\nAll companies have features."
    )

# ============================================================
# SAVE
# ============================================================

torch.save(
    {
        "data": data,
        "companies": companies,
        "company_to_id": company_to_id
    },
    output_path
)

print("\n===== SUCCESS =====")
print(
    "Saved:",
    output_path
)