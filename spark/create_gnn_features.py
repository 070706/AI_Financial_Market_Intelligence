import os
import sys

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("GNNFeaturePreparation") \
    .master("local[*]") \
    .getOrCreate()

input_path = "data/processed/gnn_training_dataset"
output_path = "data/processed/gnn_features"

df = spark.read.csv(
    input_path,
    header=True,
    inferSchema=True
)

print("\n===== INPUT DATA =====")
print("Rows:", df.count())

print("\n===== INPUT COLUMNS =====")
df.printSchema()

# Keep only information available when making the prediction.
# Next_Return is future information and must NOT be an input feature.

features = df.select(
    "Ticker",
    "Date",
    "Volume",
    "Current_Return",
    "Target"
)

features = features.dropna(
    subset=[
        "Ticker",
        "Date",
        "Volume",
        "Current_Return",
        "Target"
    ]
)

print("\n===== GNN FEATURES =====")
features.printSchema()

print("\nRows:", features.count())

print("\n===== SAMPLE =====")
features.show(20, truncate=False)

print("\n===== TARGET DISTRIBUTION =====")
features.groupBy("Target") \
    .count() \
    .orderBy("Target") \
    .show()

features.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)

print("\nGNN FEATURES SAVED SUCCESSFULLY!")
print("Location:", output_path)

spark.stop()