import os
import sys

os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["hadoop_home_dir"] = "C:\\hadoop"
os.environ["PATH"] = "C:\\hadoop\\bin;" + os.environ["PATH"]

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg,
    stddev,
    col
)

spark = SparkSession.builder \
    .appName("TrainingNodeFeatures") \
    .master("local[*]") \
    .getOrCreate()

input_path = "data/processed/clean_stock_data"
output_path = "data/graph/training_node_features"

# ============================================================
# LOAD DATA
# ============================================================

df = spark.read.csv(
    input_path,
    header=True,
    inferSchema=True,
    sep=","
)

# ============================================================
# TRAINING PERIOD ONLY
# ============================================================

df = df.filter(
    col("Date") < "2018-04-17"
)

# ============================================================
# KEEP ONLY COMPANIES IN TRAINING GRAPH
# ============================================================

graph_companies = [
    "AAL", "AEE", "AEP", "AMP", "APA", "ARE",
    "AVB", "BAC", "BEN", "BK", "C", "CINF",
    "COP", "CPT", "CVX", "D", "DAL", "DLR",
    "DOC", "DTE", "DVN", "ED", "EOG", "EQR",
    "ESS", "EXR"
]

df = df.filter(
    col("Ticker").isin(graph_companies)
)

# ============================================================
# CREATE FEATURES
# ============================================================

features = df.groupBy("Ticker").agg(
    avg("Adj Close").alias("Average_Price"),
    stddev("Adj Close").alias("Price_Volatility"),
    avg("Volume").alias("Average_Volume")
)

print("\n===== TRAINING NODE FEATURES =====")

print(
    "Companies:",
    features.select("Ticker").distinct().count()
)

features.orderBy("Ticker").show(
    30,
    truncate=False
)

# ============================================================
# SAVE
# ============================================================

features.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)

print("\n===== SUCCESS =====")
print("Saved to:", output_path)

spark.stop()