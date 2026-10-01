import os
import sys

# ============================================================
# WINDOWS HADOOP
# ============================================================

os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["hadoop_home_dir"] = "C:\\hadoop"

os.environ["PATH"] = (
    "C:\\hadoop\\bin;"
    + os.environ["PATH"]
)

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ============================================================
# SPARK
# ============================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    avg,
    sqrt
)
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("TrainingOnlyCorrelations") \
    .master("local[*]") \
    .getOrCreate()

# ============================================================
# PATHS
# ============================================================

input_path = "data/processed/stock_returns"

output_path = (
    "data/processed/"
    "training_stock_correlations"
)

# ============================================================
# LOAD RETURNS
# ============================================================

print("\n===== LOADING RETURNS =====")

df = spark.read.csv(
    input_path,
    header=True,
    inferSchema=True
)

print(
    "Original rows:",
    df.count()
)

# ============================================================
# TRAINING PERIOD ONLY
# ============================================================

print("\n===== FILTERING TRAINING PERIOD =====")

df = df.filter(
    col("Date") < "2018-04-17"
)

print(
    "Training rows:",
    df.count()
)

# ============================================================
# SELECT REQUIRED DATA
# ============================================================

df = df.select(
    "Ticker",
    "Date",
    "Daily_Return"
)

df = df.dropna(
    subset=[
        "Ticker",
        "Date",
        "Daily_Return"
    ]
)

# ============================================================
# PREPARE TWO COPIES
# ============================================================

a = df.alias("a")

b = df.alias("b")

# ============================================================
# JOIN SAME DATES
# ============================================================

print("\n===== BUILDING STOCK PAIRS =====")

pairs = a.join(
    b,
    (col("a.Date") == col("b.Date"))
    &
    (col("a.Ticker") < col("b.Ticker"))
)

pairs = pairs.select(
    col("a.Ticker").alias("Ticker_A"),
    col("b.Ticker").alias("Ticker_B"),
    col("a.Daily_Return").alias("Return_A"),
    col("b.Daily_Return").alias("Return_B")
)

# ============================================================
# CALCULATE CORRELATION COMPONENTS
# ============================================================

print("\n===== CALCULATING CORRELATIONS =====")

stats = pairs.groupBy(
    "Ticker_A",
    "Ticker_B"
).agg(

    avg(
        col("Return_A")
    ).alias("Mean_A"),

    avg(
        col("Return_B")
    ).alias("Mean_B"),

    avg(
        col("Return_A") *
        col("Return_B")
    ).alias("Mean_AB"),

    avg(
        col("Return_A") *
        col("Return_A")
    ).alias("Mean_AA"),

    avg(
        col("Return_B") *
        col("Return_B")
    ).alias("Mean_BB")
)

# ============================================================
# CORRELATION FORMULA
# ============================================================

stats = stats.withColumn(
    "Numerator",
    col("Mean_AB")
    -
    (
        col("Mean_A") *
        col("Mean_B")
    )
)

stats = stats.withColumn(
    "Denominator",
    sqrt(
        (
            col("Mean_AA")
            -
            col("Mean_A") *
            col("Mean_A")
        )
        *
        (
            col("Mean_BB")
            -
            col("Mean_B") *
            col("Mean_B")
        )
    )
)

stats = stats.withColumn(
    "Correlation",
    col("Numerator")
    /
    col("Denominator")
)

# ============================================================
# KEEP STRONG POSITIVE RELATIONSHIPS
# ============================================================

correlations = stats.filter(
    col("Correlation") >= 0.70
)

correlations = correlations.select(
    "Ticker_A",
    "Ticker_B",
    "Correlation"
)

# ============================================================
# REMOVE INVALID VALUES
# ============================================================

correlations = correlations.dropna(
    subset=[
        "Ticker_A",
        "Ticker_B",
        "Correlation"
    ]
)

# ============================================================
# SHOW RESULTS
# ============================================================

print("\n===== TRAINING CORRELATIONS =====")

print(
    "Number of correlation edges:",
    correlations.count()
)

correlations.orderBy(
    col("Correlation").desc()
).show(
    20,
    truncate=False
)

# ============================================================
# SAVE
# ============================================================

correlations.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)

print("\n===== SUCCESS =====")

print(
    "Training-only correlations saved to:"
)

print(output_path)

spark.stop()