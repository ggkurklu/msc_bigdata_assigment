import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, ArrayType, MapType
from pyspark.sql.functions import col, lower, trim, regexp_replace, concat_ws, size, coalesce, lit
from pyspark.sql.functions import udf
from pyspark.ml.feature import Tokenizer, NGram, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

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

# ========================================
# 1. Data Cleaning and Enrichment Functions
# ========================================

# Basic cleaning of columns already in your code
df_clean = df.withColumn("title", coalesce(trim(col("title")), lit("unknown"))) \
       .withColumn("main_category", coalesce(lower(trim(col("main_category"))), lit("unknown")))

# Remove duplicate rows based on parent_asin
df_clean = df_clean.dropDuplicates(["parent_asin"])

# -- Additional Text Cleaning --
# Define a UDF to clean text (lowercase, remove punctuation, and trim)
def clean_text(text):
    if text:
        # Lowercase and remove non-alphanumeric characters (except whitespace)
        text = text.lower()
        text = regexp_replace(text, "[^a-z0-9\\s]", "")
        return text.strip()
    return None

clean_text_udf = udf(clean_text, StringType())

# Apply cleaning to the title and description.
# For description (which is an array), we join the elements into a single string.
df_clean = df_clean.withColumn("clean_title", clean_text_udf(col("title")))
df_clean = df_clean.withColumn("clean_description", 
                               clean_text_udf(concat_ws(" ", col("description"))))

# ========================================
# 2. Auto-Completion Setup (Tokenization and N-Grams)
# ========================================

# Tokenize the cleaned title
tokenizer = Tokenizer(inputCol="clean_title", outputCol="title_tokens")
# Generate bigrams (n=2) for potential auto-completion suggestions
ngram = NGram(n=2, inputCol="title_tokens", outputCol="title_bigrams")

# ========================================
# 3. Trust Score Calculation
# ========================================
# Define a UDF to compute a trust score from average_rating and rating_number.
def compute_trust(average_rating, rating_number):
    # If the rating number is low, we apply a penalty to the average rating.
    if rating_number is None or rating_number < 10:
        return average_rating * 0.8
    return average_rating

compute_trust_udf = udf(compute_trust, DoubleType())
df_clean = df_clean.withColumn("trust_score", compute_trust_udf(col("average_rating"), col("rating_number")))

# ========================================
# 4. Predictive Modeling Setup
# ========================================
# For demonstration, we simulate a 'conversion_rate' column.
# In your real use case, this should be derived from your business logic or data.
from pyspark.sql.functions import rand
df_clean = df_clean.withColumn("conversion_rate", (col("average_rating") * rand()).cast(DoubleType()))

# Feature Engineering:
# As an example, we use trust_score, price, and the length of the description as features.
df_clean = df_clean.withColumn("description_length", size(col("description")))

assembler = VectorAssembler(
    inputCols=["trust_score", "price", "description_length"],
    outputCol="features_vector"
)

# Define a Linear Regression model to predict the conversion_rate
lr = LinearRegression(featuresCol="features_vector", labelCol="conversion_rate")

# Build the pipeline:
# The pipeline consists of the tokenization & ngram stages (which can be later used for auto-completion),
# the feature assembler, and the regression model.
pipeline = Pipeline(stages=[tokenizer, ngram, assembler, lr])

# Fit the model on our cleaned data
model = pipeline.fit(df_clean)

# Transform the data to get predictions
predictions = model.transform(df_clean)
predictions.select("parent_asin", "clean_title", "conversion_rate", "prediction").show(10, truncate=False)

# ========================================
# Your original grouping and writing steps (optional)
# ========================================

# Example grouping operation (grouping by main_category and computing avg of average_rating)
df_reduced = df_clean.groupBy("main_category").agg({"average_rating": "avg"})

# Write the cleaned and reduced data (optional)
# df_clean.write.mode("overwrite").json("cleaned_meta_Electronics.json")
# df_reduced.write.mode("overwrite").json("reduced_meta_Electronics.json")

# Register the DataFrame as a temporary view to run SQL queries
df_clean.createOrReplaceTempView("electronics")

# Run a SQL query to select rows where main_category is not NULL
result_df = spark.sql("SELECT * FROM electronics WHERE main_category IS NOT NULL")
result_df.show(10)

# Get the number of rows and columns (for df_clean)
num_rows = df_clean.count()
num_columns = len(df_clean.columns)

end_time = time.time()
execution_time = end_time - start_time

print("Shape: {} rows and {} columns".format(num_rows, num_columns))
print("Execution Time using PySpark:", execution_time, "seconds")

# Stop the SparkSession when done
spark.stop()
