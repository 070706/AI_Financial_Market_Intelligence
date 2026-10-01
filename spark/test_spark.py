import os
import sys

# Tell Spark to use the Python from the current virtual environment
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession

print("Python being used:")
print(sys.executable)

print("\nStarting Spark...")

spark = SparkSession.builder \
    .appName("FinancialMarketIntelligence") \
    .master("local[2]") \
    .config("spark.ui.enabled", "false") \
    .config("spark.pyspark.python", sys.executable) \
    .config("spark.pyspark.driver.python", sys.executable) \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("Spark started successfully!")

data = [
    ("TCS", 3500),
    ("INFY", 1800),
    ("HDFCBANK", 1700)
]

columns = ["ticker", "price"]

df = spark.createDataFrame(data, columns)

print("\nFinancial Market Data:")

df.show()

print("\nSpark test completed successfully!")

spark.stop()

print("Spark stopped.")