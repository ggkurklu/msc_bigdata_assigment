from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql.functions import col, rand, size, udf
from pyspark.ml.feature import Tokenizer, NGram, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import re

# ----------------------------------------------------
# Helper Functions
# ----------------------------------------------------
def compute_trust(average_rating, rating_number):
    if rating_number is None or rating_number < 10:
        return average_rating * 0.8
    return average_rating

compute_trust_udf = udf(compute_trust, DoubleType())

def clean_text(text):
    if text:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()
    return "unknown"

# ----------------------------------------------------
# Spark Session
# ----------------------------------------------------
spark = SparkSession.builder.appName("PredictionCluster").getOrCreate()


# ----------------------------------------------------
# Load Data
# ----------------------------------------------------
df_clean = spark.read.json("hdfs:///user/hadoop/output_data/")

df_clean = df_clean.withColumn("trust_score", compute_trust_udf(col("average_rating"), col("rating_number")))
df_clean = df_clean.withColumn("conversion_rate", (col("average_rating") * rand()).cast(DoubleType()))
df_clean = df_clean.withColumn("description_length", size(col("description")))

tokenizer = Tokenizer(inputCol="clean_title", outputCol="title_tokens")
ngram = NGram(n=2, inputCol="title_tokens", outputCol="title_bigrams")

assembler = VectorAssembler(
    inputCols=["trust_score", "price", "description_length"],
    outputCol="features_vector"
)

scaler = StandardScaler(inputCol="features_vector", outputCol="scaled_features")

# ----------------------------------------------------
# Regression Model (Prediction):
# ----------------------------------------------------
lr = LinearRegression(featuresCol="scaled_features", labelCol="conversion_rate")

# ----------------------------------------------------
# Classification Model (Category Prediction)
# ----------------------------------------------------
# Convert categories into numerical labels
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="category", outputCol="category_index")
classifier = LogisticRegression(featuresCol="scaled_features", labelCol="category_index")

# ----------------------------------------------------
# Clustering Model (Grouping Similar Products)
# ----------------------------------------------------
kmeans = KMeans(featuresCol="scaled_features", k=5, seed=42, predictionCol="cluster")

# ----------------------------------------------------
# Build Pipeline and Fit Model
# ----------------------------------------------------
pipeline = Pipeline(stages=[tokenizer, ngram, assembler, scaler, indexer, lr, classifier, kmeans])
model = pipeline.fit(df_clean)

# Make Predictions
predictions = model.transform(df_clean)

# Display Results
predictions.select("parent_asin", "clean_title", "conversion_rate", "prediction", "category_index", "cluster").show(10, truncate=False)

# ----------------------------------------------------
# Visualization
# ----------------------------------------------------
pdf = predictions.select("conversion_rate", "prediction").toPandas()
plt.figure(figsize=(10, 6))
plt.scatter(pdf['conversion_rate'], pdf['prediction'], alpha=0.5)
plt.xlabel("Actual Conversion Rate")
plt.ylabel("Predicted Conversion Rate")
plt.title("Actual vs Predicted Conversion Rate")
plt.show()

# ----------------------------------------------------
# Model Evaluation
# ----------------------------------------------------
lr_model = model.stages[-3]  # Linear Regression Model
print("Model Coefficients:", lr_model.coefficients)
print("Model Intercept:", lr_model.intercept)

classifier_model = model.stages[-2]  # Logistic Regression Model
print("Classification Accuracy:", classifier_model.summary.accuracy)

kmeans_model = model.stages[-1]  # KMeans Model
print("Cluster Centers:", kmeans_model.clusterCenters())

spark.stop()