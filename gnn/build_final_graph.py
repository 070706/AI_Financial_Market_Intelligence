import torch
import csv
import glob

# Load existing PyG graph
checkpoint = torch.load(
    "gnn/financial_graph.pt",
    weights_only=False
)

data = checkpoint["data"]
companies = checkpoint["companies"]
company_to_id = checkpoint["company_to_id"]

# Find node feature CSV files
feature_files = glob.glob(
    "data/graph/node_features/*.csv"
)

features = {}

# Read all feature files
for file in feature_files:

    with open(file, "r", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        for row in reader:

            ticker = row["Ticker"]

            features[ticker] = [
                float(row["Average_Price"]),
                float(row["Price_Volatility"]),
                float(row["Average_Volume"]),
                float(row["Trading_Days"])
            ]

print("Companies in PyG graph:", len(companies))
print("Companies with features:", len(features))

# Match features with PyG graph nodes
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
            [0.0, 0.0, 0.0, 0.0]
        )

# Convert features to PyTorch tensor
x = torch.tensor(
    node_features,
    dtype=torch.float
)

# Add features to graph
data.x = x

print()
print("===== FINAL GRAPH =====")
print("Nodes:", data.num_nodes)
print("Edges:", data.num_edges)
print("Node features:", data.x.shape)
print("Edge index:", data.edge_index.shape)
print("Edge attributes:", data.edge_attr.shape)

# Check missing companies
if missing:

    print()
    print("Companies without features:")

    for ticker in missing:
        print(ticker)

else:

    print()
    print("All companies have financial features.")

# Save final graph
torch.save(
    {
        "data": data,
        "companies": companies,
        "company_to_id": company_to_id
    },
    "gnn/final_financial_graph.pt"
)

print()
print("FINAL PyG GRAPH SAVED SUCCESSFULLY!")
print("File: gnn/final_financial_graph.pt")