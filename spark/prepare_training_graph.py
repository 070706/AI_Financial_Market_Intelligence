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
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("PrepareTrainingGraph") \
    .master("local[*]") \
    .getOrCreate()

input_path = "data/processed/training_stock_correlations"

companies_output = "data/graph/training_companies"
correlations_output = "data/graph/training_correlations"

print("\n===== LOADING TRAINING CORRELATIONS =====")

df = spark.read.csv(
    input_path,
    header=True,
    inferSchema=True
)

print("Correlation edges:", df.count())

# ============================================================
# CREATE COMPANY LIST
# ============================================================

companies_a = df.select(
    col("Ticker_A").alias("Ticker")
)

companies_b = df.select(
    col("Ticker_B").alias("Ticker")
)

companies = companies_a.union(companies_b).distinct()

print(
    "Companies in training graph:",
    companies.count()
)

print("\n===== COMPANIES =====")

companies.orderBy("Ticker").show(
    100,
    truncate=False
)

# ============================================================
# PREPARE CORRELATIONS
# ============================================================

correlations = df.select(
    "Ticker_A",
    "Ticker_B",
    "Correlation"
)

# ============================================================
# SAVE
# ============================================================

companies.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(companies_output)

correlations.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(correlations_output)

print("\n===== SUCCESS =====")

print("Companies saved to:")
print(companies_output)

print("Correlations saved to:")
print(correlations_output)

spark.stop()