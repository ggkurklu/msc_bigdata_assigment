import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lower, trim, regexp_replace, concat_ws, udf, coalesce, lit, when, array
)
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, ArrayType, MapType
import pandas as pd


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
df = spark.read.json("hdfs:///user/hadoop/input_data/")
# df = spark.read.schema(schema).json('meta_Electronics.jsonl')

# -----------------------------------------------------------------------------
# Ensure Target Columns are Not Null by Replacing Nulls with Default Values
# -----------------------------------------------------------------------------

# For string columns, we replace null with "unknown"
df = df.withColumn("title", coalesce(trim(col("title")), lit("unknown"))) \
       .withColumn("main_category", coalesce(lower(trim(col("main_category"))), lit("unknown")))

# For numeric columns, we replace null with 0 or 0.0
df = df.withColumn("average_rating", coalesce(col("average_rating"), lit(0.0))) \
       .withColumn("rating_number", coalesce(col("rating_number"), lit(0))) \
       .withColumn("price", coalesce(col("price"), lit(0.0)))

# For array columns, replace null with an empty array
# Note: We use array() to create an empty array literal. Ensure the proper type is inferred.
df = df.withColumn("images", coalesce(col("images"), array().cast("array<struct<thumb:string,large:string,variant:string,hi_res:string>>"))) \
       .withColumn("videos", coalesce(col("videos"), array().cast("array<string>"))) \
       .withColumn("description", coalesce(col("description"), array().cast("array<string>")))

# -----------------------------------------------------------------------------
# Additional Text Cleaning (for title and description)
# -----------------------------------------------------------------------------
def clean_text(text):
    if text:
        # Lowercase the text
        text = text.lower()
        # Remove non-alphanumeric characters (except whitespace) using Python's re module
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()
    return "unknown"

clean_text_udf = udf(clean_text, StringType())

# Clean the title
df = df.withColumn("clean_title", clean_text_udf(col("title")))

# For the description, join the array elements into a single string, then clean.
df = df.withColumn("description_joined", concat_ws(" ", col("description")))
df = df.withColumn("clean_description", clean_text_udf(col("description_joined")))

# Show a few rows to verify that no target columns are null and cleaning was applied.
df.select(
    "average_rating",
    "rating_number",
    "price",
    "images",
    "videos",
    "main_category",
    "title",
    "clean_title",
    "description_joined",
    "clean_description"
).show(5, truncate=False)

# ======================================
# 2. Advanced Data Transformations & EDA
# ======================================

# Example Aggregation: Group by main_category and count the number of records per category.
agg_df = df.groupBy("main_category").count().orderBy(col("count").desc())
print("Record count by main_category:")
agg_df.show(truncate=False)

# Example Filtering: Filter out rows where main_category is "unknown"
filtered_df = df.filter(col("main_category") != "unknown")
print("Filtered DataFrame (main_category != 'unknown'):")
filtered_df.show(10, truncate=False)

# Example EDA: Show summary statistics for numerical columns (if present)
# For instance, if you have a 'price' column:
if "price" in df.columns:
    print("Summary statistics for 'price':")
    df.select("price").describe().show()

# Likewise, if you have a 'rating_number' column:
if "rating_number" in df.columns:
    print("Summary statistics for 'rating_number':")
    df.select("rating_number").describe().show()

# Additional EDA: Count distinct clean titles (could be useful for uniqueness analysis)
distinct_titles = df.select("clean_title").distinct().count()
print(f"Distinct clean titles: {distinct_titles}")



# how many videos and images should listing have 

#brands = ['Pelican','Nanuk'] # expensive
#brands = ['Lowepro','Eylar'] # medium
brands = ['Fintie','Leayjeen'] # cheap


df_filtered = df.filter(col("brand").isin(brands) & (col("detailed_category") == det_cat))

# Group by 'images_count' and 'video_count' and count the occurrences
grouped_df = filtered_df.groupBy("images_count", "video_count").count()

# Convert the PySpark DataFrame to a Pandas DataFrame
pandas_df = grouped_df.toPandas()

# Create a heatmap using Seaborn
plt.figure(figsize=(10, 10))
heatmap_data = pandas_df.pivot("images_count", "video_count", "count")
sns.heatmap(heatmap_data, annot=True, cmap=sns.cubehelix_palette(as_cmap=True), cbar=True, fmt=",g")

# Customize the plot
plt.xlabel("Video Count")
plt.ylabel("Images Count")
plt.title("How many images and videos should the product have?", fontsize=15)

# Show the plot
plt.show()

# When finished, you might want to write the final DataFrame back to HDFS
# For example, writing in Parquet format for efficiency:
df.write.mode("overwrite").parquet("hdfs:///user/hadoop/cleaned_transformed_data/")
# sample_df = df.sample(fraction=0.1)
# sample_df.write.mode("overwrite").json("./sample_output.json")

# Stop the Spark session when done
spark.stop()
