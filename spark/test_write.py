import os
import sys

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SparkWindowsWriteTest") \
    .master("local[*]") \
    .getOrCreate()

data = [
    ("TCS", 100.0),
    ("INFY", 200.0),
    ("RELIANCE", 300.0)
]

df = spark.createDataFrame(
    data,
    ["Ticker", "Price"]
)

df.show()

df.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("data/processed/test_write")

print("SPARK WRITE TEST SUCCESSFUL")

spark.stop()