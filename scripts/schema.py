from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, ArrayType, MapType

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