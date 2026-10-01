import os
import sys

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession

from pyspark.sql.functions import col, to_date

spark = SparkSession.builder \
    .appName("FinancialMarketPreprocessing") \
    .master("local[*]") \
    .getOrCreate()

input_path = "data/raw/SP500_Historical_Data.csv"
output_path = "data/processed/clean_stock_data"

df = spark.read.csv(
    input_path,
    header=True,
    inferSchema=True,
    sep="\t"
)

print("\n===== ORIGINAL DATA =====")
df.printSchema()

print("\nRows:", df.count())

# Convert Date
df = df.withColumn(
    "Date",
    to_date(col("Date"), "dd-MM-yyyy")
)

# Remove rows with missing important values
df = df.dropna(
    subset=["Ticker", "Date", "Adj Close", "Volume"]
)

# Keep required columns
df = df.select(
    "Ticker",
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume"
)

# Remove invalid prices
df = df.filter(col("Adj Close") > 0)

print("\n===== CLEAN DATA =====")
df.printSchema()

print("\nClean rows:", df.count())

print("\n===== SAMPLE =====")
df.show(10)

# Save processed data
df.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)

print("\nProcessed data saved to:")
print(output_path)

spark.stop()