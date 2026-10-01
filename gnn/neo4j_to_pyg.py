from neo4j import GraphDatabase
import torch
from torch_geometric.data import Data


URI = "neo4j://127.0.0.1:7687"
USERNAME = "neo4j"
PASSWORD = "muvva@77"
DATABASE = "financialmarketgraph"


driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)


def load_graph():

    with driver.session(database=DATABASE) as session:

        company_result = session.run(
            """
            MATCH (c:Company)
            RETURN c.ticker AS ticker
            ORDER BY c.ticker
            """
        )

        companies = [
            record["ticker"]
            for record in company_result
        ]

        edge_result = session.run(
            """
            MATCH (a:Company)-[r:CORRELATED]->(b:Company)
            RETURN
                a.ticker AS ticker1,
                b.ticker AS ticker2,
                r.value AS correlation
            """
        )

        edges = [
            (
                record["ticker1"],
                record["ticker2"],
                record["correlation"]
            )
            for record in edge_result
        ]

    return companies, edges


print("Connecting to Neo4j...")

driver.verify_connectivity()

print("Neo4j connection successful!")

companies, edges = load_graph()

print()
print("===== NEO4J GRAPH =====")
print("Companies:", len(companies))
print("Edges:", len(edges))

print()
print("Companies:")
print(companies[:10])

print()
print("First 5 edges:")

for edge in edges[:5]:
    print(edge)


# Create mapping:
# company ticker -> integer node ID

company_to_id = {
    ticker: index
    for index, ticker in enumerate(companies)
}


# Create PyG edge list

edge_list = []

edge_weights = []

for ticker1, ticker2, correlation in edges:

    source = company_to_id[ticker1]
    target = company_to_id[ticker2]

    # Add both directions because
    # PyG message passing normally uses
    # directed edges.

    edge_list.append([source, target])
    edge_list.append([target, source])

    edge_weights.append(float(correlation))
    edge_weights.append(float(correlation))


# Convert edges to PyTorch tensor

edge_index = torch.tensor(
    edge_list,
    dtype=torch.long
).t().contiguous()


# Correlation values

edge_attr = torch.tensor(
    edge_weights,
    dtype=torch.float
)


# Initial node features
#
# For now each company gets one simple feature.
# We will replace this later with financial features.

x = torch.ones(
    (len(companies), 1),
    dtype=torch.float
)


# Create PyTorch Geometric graph

data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr
)


print()
print("===== PYTORCH GEOMETRIC GRAPH =====")

print(data)

print()
print("Node feature shape:")
print(data.x.shape)

print()
print("Edge index shape:")
print(data.edge_index.shape)

print()
print("Edge attribute shape:")
print(data.edge_attr.shape)

print()
print("Number of nodes:")
print(data.num_nodes)

print()
print("Number of edges:")
print(data.num_edges)


# Save graph

torch.save(
    {
        "data": data,
        "companies": companies,
        "company_to_id": company_to_id
    },
    "gnn/financial_graph.pt"
)


print()
print("PyG graph saved successfully!")

driver.close()