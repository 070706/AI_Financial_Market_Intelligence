import os
import sys
import torch
import csv
import glob

# ============================================================
# PATHS
# ============================================================

company_path = "data/graph/training_companies"
correlation_path = "data/graph/training_correlations"

output_path = "gnn/training_financial_graph.pt"

# ============================================================
# READ COMPANIES
# ============================================================

companies = []

company_files = glob.glob(
    company_path + "/*.csv"
)

for file in company_files:
    with open(file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            ticker = row["Ticker"]

            if ticker not in companies:
                companies.append(ticker)

companies = sorted(companies)

company_to_id = {
    ticker: i
    for i, ticker in enumerate(companies)
}

print("\n===== COMPANIES =====")
print("Number of companies:", len(companies))

print(companies)

# ============================================================
# READ CORRELATIONS
# ============================================================

edges = []
edge_values = []

correlation_files = glob.glob(
    correlation_path + "/*.csv"
)

for file in correlation_files:
    with open(file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:

            ticker_a = row["Ticker_A"]
            ticker_b = row["Ticker_B"]
            correlation = float(row["Correlation"])

            if (
                ticker_a in company_to_id
                and ticker_b in company_to_id
            ):

                a = company_to_id[ticker_a]
                b = company_to_id[ticker_b]

                # Add both directions
                edges.append([a, b])
                edges.append([b, a])

                edge_values.append(correlation)
                edge_values.append(correlation)

# ============================================================
# CREATE PYTORCH TENSORS
# ============================================================

edge_index = torch.tensor(
    edges,
    dtype=torch.long
).t().contiguous()

edge_attr = torch.tensor(
    edge_values,
    dtype=torch.float
)

# ============================================================
# NODE FEATURES
# ============================================================

# Temporary initial feature.
# Real financial features will be added later.

x = torch.ones(
    (len(companies), 1),
    dtype=torch.float
)

# ============================================================
# CREATE GRAPH DATA
# ============================================================

from torch_geometric.data import Data

data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr
)

# ============================================================
# DISPLAY
# ============================================================

print("\n===== TRAINING GRAPH =====")

print(
    "Nodes:",
    data.num_nodes
)

print(
    "Edges:",
    data.num_edges
)

print(
    "Node features:",
    data.x.shape
)

print(
    "Edge index:",
    data.edge_index.shape
)

print(
    "Edge attributes:",
    data.edge_attr.shape
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
    "Training PyG graph saved successfully!"
)

print(
    "Location:",
    output_path
)