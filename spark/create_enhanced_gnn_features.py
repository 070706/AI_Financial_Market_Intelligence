import os
import sys

# ============================================================
# SPARK / WINDOWS SETTINGS
# ============================================================

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["hadoop_home_dir"] = "C:\\hadoop"

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lag,
    avg,
    stddev,
    when,
    expr
)
from pyspark.sql.window import Window


# ============================================================
# SPARK
# ============================================================

spark = SparkSession.builder \
    .appName("EnhancedGNNFeatures") \
    .master("local[*]") \
    .getOrCreate()


# ============================================================
# PATHS
# ============================================================

input_path = "data/processed/gnn_training_dataset"

output_path = "data/processed/enhanced_gnn_features"


# ============================================================
# READ DATA
# ============================================================

print("===== READING GNN DATASET =====")

df = spark.read.csv(
    input_path,
    header=True,
    inferSchema=False
)

print("Input rows:", df.count())


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

df = df.withColumn(
    "Volume",
    expr("try_cast(`Volume` as double)")
)

df = df.withColumn(
    "Current_Return",
    expr("try_cast(`Current_Return` as double)")
)

df = df.withColumn(
    "Target",
    expr("try_cast(`Target` as int)")
)


# ============================================================
# CLEAN INVALID ROWS
# ============================================================

df = df.dropna(
    subset=[
        "Ticker",
        "Date",
        "Volume",
        "Current_Return",
        "Target"
    ]
)

print("Clean rows:", df.count())


# ============================================================
# COMPANY WINDOW
# ============================================================

window = Window \
    .partitionBy("Ticker") \
    .orderBy("Date")


# ============================================================
# 1. PREVIOUS RETURN
# ============================================================

df = df.withColumn(
    "Previous_Return",
    lag("Current_Return", 1).over(window)
)


# ============================================================
# 2. MOMENTUM
# ============================================================

df = df.withColumn(
    "Momentum",
    col("Current_Return") +
    when(
        col("Previous_Return").isNotNull(),
        col("Previous_Return")
    ).otherwise(0.0)
)


# ============================================================
# 3. ROLLING VOLATILITY
# ============================================================

rolling_window = Window \
    .partitionBy("Ticker") \
    .orderBy("Date") \
    .rowsBetween(-19, 0)

df = df.withColumn(
    "Volatility",
    stddev("Current_Return").over(
        rolling_window
    )
)


# ============================================================
# 4. MOVING AVERAGE OF RETURN
# ============================================================

df = df.withColumn(
    "Moving_Average",
    avg("Current_Return").over(
        rolling_window
    )
)


# ============================================================
# 5. PREVIOUS VOLUME
# ============================================================

df = df.withColumn(
    "Previous_Volume",
    lag("Volume", 1).over(window)
)


# ============================================================
# 6. VOLUME CHANGE
# ============================================================

df = df.withColumn(
    "Volume_Change",
    when(
        col("Previous_Volume") > 0,
        (
            col("Volume") -
            col("Previous_Volume")
        ) / col("Previous_Volume")
    ).otherwise(0.0)
)


# ============================================================
# REMOVE INVALID FEATURE ROWS
# ============================================================

df = df.dropna(
    subset=[
        "Volatility",
        "Moving_Average"
    ]
)


# ============================================================
# SELECT FINAL FEATURES
# ============================================================

df = df.select(
    "Ticker",
    "Date",
    "Volume",
    "Current_Return",
    "Volatility",
    "Momentum",
    "Moving_Average",
    "Volume_Change",
    "Target"
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print()
print("===== ENHANCED FEATURES =====")

print("Rows:", df.count())

print()
print("Schema:")

df.printSchema()

print()
print("===== SAMPLE =====")

df.show(10)


# ============================================================
# SAVE
# ============================================================

df.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)


print()
print("===== SUCCESS =====")

print("Enhanced GNN features saved:")
print(output_path)


spark.stop()