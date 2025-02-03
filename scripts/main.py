import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lower, trim, concat_ws, udf, coalesce, lit, array
)
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, ArrayType, MapType


# Initialize SparkSession
spark = SparkSession.builder.appName("AdvancedDataProcessing").getOrCreate()

# ================================
# Define Schema 
# ================================

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

# ================================
# Import Data from hsfs of s3
# ================================
df = spark.read.schema(schema).json("hdfs:///user/hadoop/raw_data/")
df = df.cache()

# -----------------------------------------------------------------------------
# ensure there are no null values
# -----------------------------------------------------------------------------

# put unknown for null values
df = df.withColumn("title", coalesce(trim(col("title")), lit("unknown"))) \
       .withColumn("main_category", coalesce(lower(trim(col("main_category"))), lit("unknown")))

# replace null with double
df = df.withColumn("average_rating", coalesce(col("average_rating"), lit(0.0))) \
       .withColumn("rating_number", coalesce(col("rating_number"), lit(0))) \
       .withColumn("price", coalesce(col("price"), lit(0.0)))

# use array for empty values and nulls
df = df.withColumn("images", coalesce(col("images"), array().cast("array<struct<thumb:string,large:string,variant:string,hi_res:string>>"))) \
       .withColumn("videos", coalesce(col("videos"), array().cast("array<string>"))) \
       .withColumn("description", coalesce(col("description"), array().cast("array<string>")))

# -----------------------------------------------------------------------------
# additional cleaning method
# -----------------------------------------------------------------------------
def clean_text(text):
    if text:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()
    return "unknown"

clean_text_udf = udf(clean_text, StringType())

# clean title 
df = df.withColumn("clean_title", clean_text_udf(col("title")))

# clean descriptions
df = df.withColumn("description_joined", concat_ws(" ", col("description")))
df = df.withColumn("clean_description", clean_text_udf(col("description_joined")))

# show some values from the table
df.select(
    "average_rating",
    "rating_number",
    "price",
    "images",
    "videos",
    "main_category",
    "title",
    "clean_title",
).show(5, truncate=False)

# ======================================
# Advanced Data Transformations & EDA
# ======================================

# agg main ctegory
agg_df = df.groupBy("main_category").count().orderBy(col("count").desc())
print("Record count by main_category:")
agg_df.show(truncate=False)

# filter main category
filtered_df = df.filter(col("main_category") != "unknown")
print("Filtered DataFrame (main_category != 'unknown'):")
filtered_df.show(10, truncate=False)

# price column summary
if "price" in df.columns:
    print("Summary statistics for 'price':")
    df.select("price").describe().show()

# rating summary
if "rating_number" in df.columns:
    print("Summary statistics for 'rating_number':")
    df.select("rating_number").describe().show()

# distinct titles
distinct_titles = df.select("clean_title").distinct().count()
print(f"Distinct clean titles: {distinct_titles}")


# write to hadoop HDFs
df.write.mode("overwrite").json("hdfs:///user/hadoop/output_data/")
df.write.mode("overwrite").json("s3a://msc-data-analytics/builds/01-2025/output/")


# stop app
spark.stop()
