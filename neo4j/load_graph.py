from neo4j import GraphDatabase

URI = "neo4j://127.0.0.1:7687"
USERNAME = "neo4j"
PASSWORD = "muvva@77"
DATABASE = "financialmarketgraph"

driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)


def create_graph(tx, companies, correlations):

    # Create company nodes
    for ticker in companies:
        tx.run(
            """
            MERGE (c:Company {ticker: $ticker})
            """,
            ticker=ticker
        )

    # Create correlation relationships
    for ticker1, ticker2, correlation in correlations:
        tx.run(
            """
            MATCH (a:Company {ticker: $ticker1})
            MATCH (b:Company {ticker: $ticker2})
            MERGE (a)-[r:CORRELATED]->(b)
            SET r.value = $correlation
            """,
            ticker1=ticker1,
            ticker2=ticker2,
            correlation=float(correlation)
        )


try:
    driver.verify_connectivity()

    print("Neo4j connection successful!")

    # Read companies CSV files
    import glob
    import csv

    company_files = glob.glob(
        "data/graph/companies/*.csv"
    )

    correlation_files = glob.glob(
        "data/graph/correlations/*.csv"
    )

    companies = []

    for file in company_files:
        with open(file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                companies.append(row["Ticker"])

    correlations = []

    for file in correlation_files:
        with open(file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                correlations.append(
                    (
                        row["Ticker1"],
                        row["Ticker2"],
                        row["Correlation"]
                    )
                )

    # Remove duplicate company tickers
    companies = list(set(companies))

    print("Companies:", len(companies))
    print("Correlations:", len(correlations))

    with driver.session(database=DATABASE) as session:

        # Clear existing project graph
        session.run(
            """
            MATCH (n)
            DETACH DELETE n
            """
        )

        print("Old graph cleared.")

        session.execute_write(
            create_graph,
            companies,
            correlations
        )

    print("\n===== GRAPH LOADED SUCCESSFULLY =====")

finally:
    driver.close()