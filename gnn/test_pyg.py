import torch

graph_file = "gnn/financial_graph.pt"

checkpoint = torch.load(
    graph_file,
    weights_only=False
)

data = checkpoint["data"]
companies = checkpoint["companies"]
company_to_id = checkpoint["company_to_id"]

print("===== PYTORCH GEOMETRIC GRAPH =====")

print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)

print("\nNode feature shape:")
print(data.x.shape)

print("\nEdge index shape:")
print(data.edge_index.shape)

print("\nEdge attribute shape:")
print(data.edge_attr.shape)

print("\nFirst 10 companies:")

for i, ticker in enumerate(companies[:10]):
    print(i, ticker)

print("\nFirst 10 edges:")

for i in range(min(10, data.num_edges)):
    source = data.edge_index[0, i].item()
    target = data.edge_index[1, i].item()

    print(
        companies[source],
        "->",
        companies[target],
        "Correlation:",
        data.edge_attr[i].item()
    )