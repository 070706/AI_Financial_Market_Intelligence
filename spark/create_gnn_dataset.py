import os
import sys

# ============================================================
# WINDOWS HADOOP CONFIGURATION
# ============================================================

os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["hadoop.home.dir"] = r"C:\hadoop"
os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ["PATH"]


# ============================================================
# SPARK NETWORKING CONFIGURATION
# ============================================================

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lag,
    when,
    regexp_replace,
    to_date,
    expr
)
from pyspark.sql.window import Window


# ============================================================
# CREATE SPARK SESSION
# ============================================================

spark = SparkSession.builder \
    .appName("GNNTrainingDataset") \
    .master("local[*]") \
    .getOrCreate()


# ============================================================
# PATHS
# ============================================================

input_path = "data/processed/clean_stock_data"
output_path = "data/processed/gnn_training_dataset"


# ============================================================
# READ DATA AS STRING
# ============================================================

df = spark.read.csv(
    input_path,
    header=True,
    inferSchema=False
)

print()
print("===== INPUT DATA =====")
print("Rows:", df.count())


# ============================================================
# CONVERT DATE
# ============================================================

df = df.withColumn(
    "Date",
    to_date(
        col("Date"),
        "yyyy-MM-dd"
    )
)


# ============================================================
# SAFELY CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume"
]

for column_name in numeric_columns:

    cleaned_column = regexp_replace(
        col(column_name),
        ",",
        ""
    )

    df = df.withColumn(
        column_name,
        expr(
            f"try_cast(`{column_name}` as double)"
        )
    )


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

df = df.dropna(
    subset=[
        "Ticker",
        "Date",
        "Adj Close",
        "Volume"
    ]
)

df = df.filter(
    col("Adj Close") > 0
)


print()
print("===== CLEAN DATA =====")
print("Rows:", df.count())


# ============================================================
# REMOVE DUPLICATE TRADING-DAY RECORDS
#
# One company should have only one row per date.
# This is IMPORTANT before calculating returns.
# ============================================================

before_duplicates = df.count()

df = df.dropDuplicates(
    [
        "Ticker",
        "Date"
    ]
)

after_duplicates = df.count()

print()
print("===== DUPLICATE REMOVAL =====")
print("Rows before:", before_duplicates)
print("Rows after:", after_duplicates)
print("Duplicates removed:", before_duplicates - after_duplicates)


# ============================================================
# CREATE TIME-SERIES WINDOW
# ============================================================

window = Window \
    .partitionBy("Ticker") \
    .orderBy("Date")


# ============================================================
# PREVIOUS DAY PRICE
# ============================================================

df = df.withColumn(
    "Previous_Adj_Close",
    lag(
        "Adj Close",
        1
    ).over(window)
)


# ============================================================
# CURRENT RETURN
# ============================================================

df = df.withColumn(
    "Current_Return",
    (
        col("Adj Close")
        - col("Previous_Adj_Close")
    )
    / col("Previous_Adj_Close")
)


# ============================================================
# NEXT DAY PRICE
# ============================================================

df = df.withColumn(
    "Next_Adj_Close",
    lag(
        "Adj Close",
        -1
    ).over(window)
)


# ============================================================
# NEXT DAY RETURN
# ============================================================

df = df.withColumn(
    "Next_Return",
    (
        col("Next_Adj_Close")
        - col("Adj Close")
    )
    / col("Adj Close")
)


# ============================================================
# TARGET
#
# 0 = next return <= 0
# 1 = next return > 0
# ============================================================

df = df.withColumn(
    "Target",
    when(
        col("Next_Return") > 0,
        1
    ).otherwise(0)
)


# ============================================================
# REMOVE FIRST/LAST ROWS OF EACH STOCK
# ============================================================

df = df.dropna(
    subset=[
        "Current_Return",
        "Next_Return"
    ]
)


# ============================================================
# FINAL DATASET
# ============================================================

dataset = df.select(
    "Ticker",
    "Date",
    "Adj Close",
    "Volume",
    "Current_Return",
    "Next_Return",
    "Target"
)


# ============================================================
# SAMPLE
# ============================================================

print()
print("===== GNN TRAINING DATASET SAMPLE =====")

dataset.orderBy(
    "Ticker",
    "Date"
).show(
    20,
    truncate=False
)


# ============================================================
# DATASET INFORMATION
# ============================================================

print()
print("===== DATASET INFORMATION =====")

print(
    "Rows:",
    dataset.count()
)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print()
print("===== TARGET DISTRIBUTION =====")

dataset.groupBy(
    "Target"
).count().orderBy(
    "Target"
).show()


# ============================================================
# SAVE DATASET
# ============================================================

dataset.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)


# ============================================================
# SUCCESS
# ============================================================

print()
print("GNN TRAINING DATASET SAVED SUCCESSFULLY!")

print(
    "Location:",
    output_path
)


spark.stop()