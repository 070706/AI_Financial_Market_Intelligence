import os
import sys

# Windows Hadoop configuration
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["hadoop.home.dir"] = r"C:\hadoop"
os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ["PATH"]

# Spark networking configuration
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg,
    stddev,
    count,
    col,
    regexp_replace,
    when
)

spark = SparkSession.builder \
    .appName("FinancialNodeFeatures") \
    .master("local[*]") \
    .getOrCreate()

input_path = "data/processed/clean_stock_data"
output_path = "data/graph/node_features"

df = spark.read.csv(
    input_path,
    header=True,
    inferSchema=True
)

print("===== INPUT DATA =====")
print("Rows:", df.count())

# Convert numeric columns safely
numeric_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume"
]

for column_name in numeric_columns:
    df = df.withColumn(
        column_name,
        regexp_replace(col(column_name).cast("string"), ",", "").cast("double")
    )

# Remove rows where important numeric values are invalid
df = df.dropna(
    subset=["Ticker", "Date", "Adj Close", "Volume"]
)

print("\n===== CLEANED DATA =====")
print("Valid rows:", df.count())

# Create company-level features
features = df.groupBy("Ticker").agg(
    avg("Adj Close").alias("Average_Price"),
    stddev("Adj Close").alias("Price_Volatility"),
    avg("Volume").alias("Average_Volume"),
    count("*").alias("Trading_Days")
)

features = features.dropna()

print("\n===== NODE FEATURES =====")
features.show(20, truncate=False)

print("\nNumber of companies:", features.count())

features.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)

print("\nNode features saved to:")
print(output_path)

spark.stop()