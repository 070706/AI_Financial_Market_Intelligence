import os
import sys

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("PrepareFinancialGraph") \
    .master("local[*]") \
    .getOrCreate()

correlation_path = "data/processed/stock_correlations"

companies_output = "data/graph/companies"
correlations_output = "data/graph/correlations"

df = spark.read.csv(
    correlation_path,
    header=True,
    inferSchema=True
)

print("===== CORRELATIONS =====")
df.printSchema()

print("\nCorrelation edges:", df.count())

# -----------------------------
# CREATE COMPANY NODES
# -----------------------------

companies1 = df.select(
    col("Ticker1").alias("Ticker")
)

companies2 = df.select(
    col("Ticker2").alias("Ticker")
)

companies = companies1.union(companies2).distinct()

print("\n===== COMPANIES =====")
print("Number of companies:", companies.count())

companies.show(20)

# -----------------------------
# CREATE GRAPH EDGES
# -----------------------------

correlations = df.select(
    "Ticker1",
    "Ticker2",
    "Correlation"
)

print("\n===== CORRELATIONS =====")
correlations.show(20, truncate=False)

# -----------------------------
# SAVE
# -----------------------------

companies.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(companies_output)

correlations.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(correlations_output)

print("\nGraph data saved successfully.")

print("Companies:")
print(companies_output)

print("Correlations:")
print(correlations_output)

spark.stop()