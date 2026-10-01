import os
import sys

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    avg,
    sqrt,
    pow,
    sum as spark_sum
)

spark = SparkSession.builder \
    .appName("StockPairwiseCorrelations") \
    .master("local[*]") \
    .getOrCreate()

input_path = "data/processed/stock_returns"
output_path = "data/processed/stock_correlations"

# Read returns
df = spark.read.csv(
    input_path,
    header=True,
    inferSchema=True
)

print("===== RETURNS DATA =====")
df.printSchema()

print("\nReturn rows:", df.count())

# Keep only required columns
df = df.select(
    "Ticker",
    "Date",
    "Daily_Return"
)

# Create two copies for pairwise comparison
a = df.alias("a")
b = df.alias("b")

# Match stocks on the same trading date
pairs = a.join(
    b,
    (col("a.Date") == col("b.Date")) &
    (col("a.Ticker") < col("b.Ticker"))
)

print("\n===== CALCULATING PAIRS =====")

# Calculate correlation components
pair_stats = pairs.groupBy(
    col("a.Ticker").alias("Ticker1"),
    col("b.Ticker").alias("Ticker2")
).agg(
    avg(
        col("a.Daily_Return")
    ).alias("Mean1"),

    avg(
        col("b.Daily_Return")
    ).alias("Mean2"),

    avg(
        col("a.Daily_Return") * col("b.Daily_Return")
    ).alias("MeanProduct"),

    avg(
        pow(col("a.Daily_Return"), 2)
    ).alias("MeanSquare1"),

    avg(
        pow(col("b.Daily_Return"), 2)
    ).alias("MeanSquare2"),

    spark_sum(
        col("a.Daily_Return") * col("b.Daily_Return")
    ).alias("ProductSum"),

    spark_sum(
        col("a.Daily_Return")
    ).alias("Sum1"),

    spark_sum(
        col("b.Daily_Return")
    ).alias("Sum2")
)

# Calculate Pearson correlation
pair_stats = pair_stats.withColumn(
    "Covariance",
    col("MeanProduct") -
    (col("Mean1") * col("Mean2"))
)

pair_stats = pair_stats.withColumn(
    "Variance1",
    col("MeanSquare1") -
    pow(col("Mean1"), 2)
)

pair_stats = pair_stats.withColumn(
    "Variance2",
    col("MeanSquare2") -
    pow(col("Mean2"), 2)
)

pair_stats = pair_stats.withColumn(
    "Correlation",
    col("Covariance") /
    sqrt(
        col("Variance1") *
        col("Variance2")
    )
)

# Keep meaningful correlations
correlations = pair_stats \
    .filter(col("Correlation").isNotNull()) \
    .filter(col("Correlation") >= 0.70) \
    .select(
        "Ticker1",
        "Ticker2",
        "Correlation"
    )

print("\n===== CORRELATION SAMPLE =====")

correlations.show(20, truncate=False)

print(
    "\nNumber of correlation edges:",
    correlations.count()
)

# Save
correlations.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)

print("\nCorrelations saved to:")
print(output_path)

spark.stop()