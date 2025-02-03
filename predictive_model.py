# prediction_and_visualization.py

from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, rand, size, udf
from pyspark.ml.feature import Tokenizer, NGram, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import re

# ----------------------------------------------------
# Helper Functions
# ----------------------------------------------------
# Compute a trust score based on average_rating and rating_number
def compute_trust(average_rating, rating_number):
    # If there are fewer than 10 ratings, penalize the average_rating
    if rating_number is None or rating_number < 10:
        return average_rating * 0.8
    return average_rating

compute_trust_udf = udf(compute_trust, DoubleType())

# (Optional) If you need to re-clean text, you can use this UDF.
def clean_text(text):
    if text:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()
    return "unknown"

# ----------------------------------------------------
# Create a Spark Session
# ----------------------------------------------------
spark = SparkSession.builder \
    .appName("PredictionVisualization") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# ----------------------------------------------------
# Load Cleaned Data from HDFS
# ----------------------------------------------------
# Change the path below to the HDFS location of your cleaned data file.
cleaned_data_path = "hdfs:///user/hadoop/cleaned_transformed_data/"
df_clean = spark.read.json(cleaned_data_path)

# If needed, recompute or add any missing fields. For example, ensure trust_score exists.
df_clean = df_clean.withColumn("trust_score", compute_trust_udf(col("average_rating"), col("rating_number")))

# Simulate a conversion_rate column (replace this with your real business logic)
df_clean = df_clean.withColumn("conversion_rate", (col("average_rating") * rand()).cast(DoubleType()))

# Create a feature that captures the length of the description array.
df_clean = df_clean.withColumn("description_length", size(col("description")))

# ----------------------------------------------------
# Predictive Modeling Setup
# ----------------------------------------------------
# We will use the clean_title column for tokenization and n-gram generation.
# (Auto-completion steps: tokenizer and ngram are included so that the same pipeline can be extended for auto-completion if needed.)
tokenizer = Tokenizer(inputCol="clean_title", outputCol="title_tokens")
ngram = NGram(n=2, inputCol="title_tokens", outputCol="title_bigrams")

# Assemble features: we use trust_score, price, and description_length as features.
assembler = VectorAssembler(
    inputCols=["trust_score", "price", "description_length"],
    outputCol="features_vector"
)

# Define a simple Linear Regression model to predict conversion_rate.
lr = LinearRegression(featuresCol="features_vector", labelCol="conversion_rate")

# Build a pipeline that includes the tokenization, n-gram creation, feature assembly, and the regression model.
pipeline = Pipeline(stages=[tokenizer, ngram, assembler, lr])

# Fit the pipeline model using the cleaned data.
model = pipeline.fit(df_clean)

# Use the model to make predictions.
predictions = model.transform(df_clean)

# Display some results.
predictions.select("parent_asin", "clean_title", "conversion_rate", "prediction").show(10, truncate=False)

# ----------------------------------------------------
# Visualization
# ----------------------------------------------------
# Convert the predictions to a Pandas DataFrame for visualization.
pdf = predictions.select("conversion_rate", "prediction").toPandas()

# Create a scatter plot comparing actual vs predicted conversion rates.
plt.figure(figsize=(10, 6))
plt.scatter(pdf['conversion_rate'], pdf['prediction'], alpha=0.5)
plt.xlabel("Actual Conversion Rate")
plt.ylabel("Predicted Conversion Rate")
plt.title("Actual vs Predicted Conversion Rate")
# Plot the perfect prediction line.
min_val = min(pdf['conversion_rate'].min(), pdf['prediction'].min())
max_val = max(pdf['conversion_rate'].max(), pdf['prediction'].max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=2, label="Ideal Prediction")
plt.legend()
plt.show()

# ----------------------------------------------------
# Model Summary and Business Insights
# ----------------------------------------------------
# Print the model coefficients and intercept for business interpretation.
lr_model = model.stages[-1]  # The last stage is the Linear Regression model
print("Model Coefficients:", lr_model.coefficients)
print("Model Intercept:", lr_model.intercept)

# Business Insights (example):
# - If the coefficient for trust_score is high, it indicates that a higher trust score (derived from average_rating and rating_number)
#   significantly drives conversion rate. Sellers should focus on building trust (e.g., getting more reviews).
# - A high coefficient for price (positive or negative) suggests that pricing has a strong impact on conversion.
# - The description_length's coefficient can tell you whether longer descriptions contribute positively to conversion.
#
# These insights can help sellers improve their listings:
#  - Emphasize improving their ratings and review counts.
#  - Consider adjusting pricing strategies.
#  - Optimize product descriptions for clarity and length.
#
# Save the visualization if needed:
plt.savefig("actual_vs_predicted_conversion_rate.png")
print("Visualization saved as actual_vs_predicted_conversion_rate.png")

# ----------------------------------------------------
# Stop the Spark Session
# ----------------------------------------------------
spark.stop()
