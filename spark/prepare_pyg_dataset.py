import os
import sys
import torch

# ============================================================
# WINDOWS HADOOP CONFIGURATION
# ============================================================

os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["hadoop_home_dir"] = "C:\\hadoop"

# Make Hadoop DLL and winutils visible to Spark's Java process
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
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("PreparePyGDataset") \
    .master("local[*]") \
    .getOrCreate()

# ============================================================
# PATHS
# ============================================================

input_path = "data/processed/gnn_training_final"
graph_path = "gnn/final_financial_graph.pt"
output_path = "gnn/pyg_training_dataset.pt"

# ============================================================
# LOAD FINAL GRAPH
# ============================================================

print("\n===== LOADING GRAPH =====")

graph = torch.load(
    graph_path,
    weights_only=False
)

data = graph["data"]
companies = graph["companies"]
company_to_id = graph["company_to_id"]

print("Graph nodes:", len(companies))
print("Graph edges:", data.num_edges)

# ============================================================
# LOAD FINAL TRAINING DATA
# ============================================================

print("\n===== LOADING TRAINING DATA =====")

df = spark.read.csv(
    input_path,
    header=True,
    inferSchema=True
)

print("Rows:", df.count())

# ============================================================
# CHECK COLUMNS
# ============================================================

print("\n===== COLUMNS =====")
print(df.columns)

required_columns = [
    "Ticker",
    "Date",
    "Volume",
    "Current_Return",
    "Target"
]

for column_name in required_columns:

    if column_name not in df.columns:

        raise ValueError(
            "Missing required column: "
            + column_name
        )

print("All required columns are present.")

# ============================================================
# REMOVE INVALID ROWS
# ============================================================

df = df.dropna(
    subset=required_columns
)

print("\n===== AFTER CLEANING =====")
print("Rows:", df.count())

# ============================================================
# CREATE TICKER -> NODE ID MAPPING
# ============================================================

print("\n===== PREPARING NODE IDs =====")

mapping = spark.createDataFrame(
    [
        (
            ticker,
            company_to_id[ticker]
        )
        for ticker in companies
    ],
    [
        "Ticker",
        "Node_ID"
    ]
)

df = df.join(
    mapping,
    on="Ticker",
    how="inner"
)

print(
    "Rows after graph mapping:",
    df.count()
)

print(
    "Companies after graph mapping:",
    df.select("Ticker").distinct().count()
)

# ============================================================
# DATE RANGE
# ============================================================

print("\n===== DATE RANGE =====")

df.selectExpr(
    "min(Date) AS Minimum_Date",
    "max(Date) AS Maximum_Date"
).show()

# ============================================================
# GET ALL TRADING DATES
# ============================================================

print("\n===== PREPARING CHRONOLOGICAL SPLIT =====")

dates = [
    row["Date"]
    for row in
    df.select("Date")
      .distinct()
      .orderBy("Date")
      .collect()
]

total_dates = len(dates)

print("Total unique dates:", total_dates)

# ============================================================
# 70% TRAIN
# 15% VALIDATION
# 15% TEST
# ============================================================

train_end_index = int(
    total_dates * 0.70
)

validation_end_index = int(
    total_dates * 0.85
)

train_end_date = dates[
    train_end_index
]

validation_end_date = dates[
    validation_end_index
]

print(
    "Training ends before:",
    train_end_date
)

print(
    "Validation ends before:",
    validation_end_date
)

# ============================================================
# CREATE TRAINING DATA
# ============================================================

train_df = df.filter(
    col("Date") < train_end_date
)

# ============================================================
# CREATE VALIDATION DATA
# ============================================================

validation_df = df.filter(
    (col("Date") >= train_end_date)
    &
    (col("Date") < validation_end_date)
)

# ============================================================
# CREATE TEST DATA
# ============================================================

test_df = df.filter(
    col("Date") >= validation_end_date
)

# ============================================================
# SPLIT SIZES
# ============================================================

print("\n===== SPLIT SIZES =====")

train_count = train_df.count()
validation_count = validation_df.count()
test_count = test_df.count()

print(
    "Training rows:",
    train_count
)

print(
    "Validation rows:",
    validation_count
)

print(
    "Test rows:",
    test_count
)

# ============================================================
# CHECK TARGET DISTRIBUTION
# ============================================================

print("\n===== TRAIN TARGET DISTRIBUTION =====")

train_df.groupBy("Target") \
    .count() \
    .orderBy("Target") \
    .show()

print("\n===== VALIDATION TARGET DISTRIBUTION =====")

validation_df.groupBy("Target") \
    .count() \
    .orderBy("Target") \
    .show()

print("\n===== TEST TARGET DISTRIBUTION =====")

test_df.groupBy("Target") \
    .count() \
    .orderBy("Target") \
    .show()

# ============================================================
# CONVERT SPARK DATA TO PYTHON RECORDS
# ============================================================

def convert_to_records(spark_df):

    rows = spark_df.select(
        "Date",
        "Node_ID",
        "Volume",
        "Current_Return",
        "Target"
    ).orderBy(
        "Date",
        "Node_ID"
    ).collect()

    records = []

    for row in rows:

        records.append(
            {
                "date": str(row["Date"]),

                "node_id": int(
                    row["Node_ID"]
                ),

                "volume": float(
                    row["Volume"]
                ),

                "current_return": float(
                    row["Current_Return"]
                ),

                "target": int(
                    row["Target"]
                )
            }
        )

    return records


# ============================================================
# CONVERT TRAINING DATA
# ============================================================

print("\n===== CONVERTING TRAINING DATA =====")

train_records = convert_to_records(
    train_df
)

print(
    "Training records:",
    len(train_records)
)

# ============================================================
# CONVERT VALIDATION DATA
# ============================================================

print("\n===== CONVERTING VALIDATION DATA =====")

validation_records = convert_to_records(
    validation_df
)

print(
    "Validation records:",
    len(validation_records)
)

# ============================================================
# CONVERT TEST DATA
# ============================================================

print("\n===== CONVERTING TEST DATA =====")

test_records = convert_to_records(
    test_df
)

print(
    "Test records:",
    len(test_records)
)

# ============================================================
# BUILD FINAL DATASET
# ============================================================

dataset = {

    "graph": data,

    "companies": companies,

    "company_to_id": company_to_id,

    "train": train_records,

    "validation": validation_records,

    "test": test_records
}

# ============================================================
# SAVE PYTORCH DATASET
# ============================================================

torch.save(
    dataset,
    output_path
)

# ============================================================
# FINAL SUCCESS MESSAGE
# ============================================================

print("\n===== SUCCESS =====")

print(
    "PyG training dataset saved successfully!"
)

print(
    "Location:",
    output_path
)

print()
print(
    "Train records:",
    len(train_records)
)

print(
    "Validation records:",
    len(validation_records)
)

print(
    "Test records:",
    len(test_records)
)

print(
    "Graph nodes:",
    len(companies)
)

print(
    "Graph edges:",
    data.num_edges
)

# ============================================================
# STOP SPARK
# ============================================================

spark.stop()