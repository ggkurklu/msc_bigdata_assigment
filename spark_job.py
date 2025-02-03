import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, ArrayType, MapType
from pyspark.sql.functions import col, lower, trim

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Electronics") \
    .getOrCreate()

# Define a schema based on your JSON structure
schema = StructType([
    StructField("main_category", StringType(), True),
    StructField("title", StringType(), True),
    StructField("average_rating", DoubleType(), True),
    StructField("rating_number", IntegerType(), True),
    StructField("features", ArrayType(StringType()), True),
    StructField("description", ArrayType(StringType()), True),
    StructField("price", DoubleType(), True),
    StructField("images", ArrayType(
        StructType([
            StructField("thumb", StringType(), True),
            StructField("large", StringType(), True),
            StructField("variant", StringType(), True),
            StructField("hi_res", StringType(), True)
        ])
    ), True),
    StructField("videos", ArrayType(StringType()), True),
    StructField("store", StringType(), True),
    StructField("categories", ArrayType(StringType()), True),
    StructField("details", MapType(StringType(), StringType()), True),
    StructField("parent_asin", StringType(), True),
    StructField("bought_together", StringType(), True)
])

start_time = time.time()

# Read the JSONL file
df = spark.read.schema(schema).json('meta_Electronics.jsonl')

# Clean the DataFrame
df_clean = df.withColumn("title", trim(col("title"))) \
             .withColumn("main_category", lower(trim(col("main_category"))))

# Remove duplicate rows based on parent_asin
df_clean = df_clean.dropDuplicates(["parent_asin"])

# Example grouping operation (as per your original code)
df_reduced = df_clean.groupBy("main_category").agg({"average_rating": "avg"})

# Write the cleaned and reduced data (optional)
df_clean.write.mode("overwrite").json("cleaned_meta_Electronics.json")
df_reduced.write.mode("overwrite").json("reduced_meta_Electronics.json")

# Register the DataFrame as a temporary view to run SQL queries
df_clean.createOrReplaceTempView("electronics")

# Run a SQL query to select rows where main_category is not NULL
result_df = spark.sql("SELECT * FROM electronics WHERE main_category IS NOT NULL")
result_df.show(10)

# Get the number of rows and columns (for df_clean)
num_rows = df_clean.count()
num_columns = len(df_clean.columns)

# End the timer and print execution details
end_time = time.time()
execution_time = end_time - start_time

print("Shape: {} rows and {} columns".format(num_rows, num_columns))
print("Execution Time using PySpark:", execution_time, "seconds")

# Stop the SparkSession when done
spark.stop()
