from neo4j import GraphDatabase

URI = "neo4j://127.0.0.1:7687"
USERNAME = "neo4j"
PASSWORD = "muvva@77"
DATABASE = "financialmarketgraph"

driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)

try:
    driver.verify_connectivity()
    print("Neo4j connection successful!")

    with driver.session(database=DATABASE) as session:
        result = session.run(
            "MATCH (n) RETURN count(n) AS count"
        )

        record = result.single()

        print("Number of nodes:", record["count"])

finally:
    driver.close()