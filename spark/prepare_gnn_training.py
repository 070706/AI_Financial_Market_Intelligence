import os
import sys
import torch

# ============================================================
# WINDOWS HADOOP CONFIGURATION
# ============================================================

os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["hadoop_home_dir"] = "C:\\hadoop"

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ============================================================
# LOAD SPARK
# ============================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("PrepareFinalGNNTrainingData") \
    .master("local[*]") \
    .getOrCreate()

# ============================================================
# PATHS
# ============================================================

input_path = "data/processed/gnn_features"
output_path = "data/processed/gnn_training_final"

graph_path = "gnn/final_financial_graph.pt"

# ============================================================
# LOAD ACTUAL GRAPH COMPANIES
# ============================================================

print("\n===== LOADING GRAPH =====")

graph = torch.load(
    graph_path,
    weights_only=False
)

graph_companies = graph["companies"]

print("Graph companies:", len(graph_companies))

for ticker in graph_companies:
    print(ticker)

# ============================================================
# LOAD GNN FEATURES
# ============================================================

df = spark.read.csv(
    input_path,
    header=True,
    inferSchema=True
)

print("\n===== ORIGINAL GNN FEATURES =====")
print("Rows:", df.count())

print(
    "Companies in feature dataset:",
    df.select("Ticker").distinct().count()
)

# ============================================================
# FILTER TO ACTUAL GRAPH COMPANIES
# ============================================================

df = df.filter(
    col("Ticker").isin(graph_companies)
)

print("\n===== FILTERED TO GRAPH COMPANIES =====")

print("Rows:", df.count())

print(
    "Companies:",
    df.select("Ticker").distinct().count()
)

# ============================================================
# CHECK WHICH GRAPH COMPANIES ARE PRESENT
# ============================================================

feature_companies = set(
    row["Ticker"]
    for row in df.select("Ticker").distinct().collect()
)

missing_companies = [
    ticker
    for ticker in graph_companies
    if ticker not in feature_companies
]

print("\n===== COMPANY ALIGNMENT =====")

if missing_companies:
    print("Missing graph companies:")
    for ticker in missing_companies:
        print(ticker)
else:
    print("All graph companies have training data.")

# ============================================================
# DATE RANGE
# ============================================================

print("\n===== DATE RANGE =====")

df.selectExpr(
    "min(Date) AS Minimum_Date",
    "max(Date) AS Maximum_Date"
).show()

# ============================================================
# COMPANY COUNTS
# ============================================================

print("\n===== COMPANY COUNTS =====")

df.groupBy("Ticker") \
    .count() \
    .orderBy("Ticker") \
    .show(50)

# ============================================================
# SAMPLE
# ============================================================

print("\n===== SAMPLE =====")

df.orderBy(
    "Date",
    "Ticker"
).show(
    30,
    truncate=False
)

# ============================================================
# SAVE FINAL TRAINING DATA
# ============================================================

df.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)

print("\n===== SUCCESS =====")

print("GNN training data prepared!")
print("Location:", output_path)

spark.stop()