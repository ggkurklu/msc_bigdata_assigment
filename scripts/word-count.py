import pyspark
import sys

# Check if exactly two arguments are provided
if len(sys.argv) != 3:
    raise Exception("Exactly 2 arguments are required: <inputuri> <outputuri>")

# Input and output URIs from command-line arguments
inputUri = sys.argv[1]
outputUri = sys.argv[2]

# Initialize SparkContext
sc = pyspark.SparkContext()

# Read input file
lines = sc.textFile(inputUri)

# Perform word count
words = lines.flatMap(lambda line: line.split())
wordCounts = words.map(lambda word: (word, 1)).reduceByKey(lambda count1, count2: count1 + count2)

# Save word frequencies to output
wordCounts.saveAsTextFile(outputUri)

# Calculate total word count
totalWords = wordCounts.map(lambda word_count: word_count[1]).sum()

# Save total word count to a separate output directory
sc.parallelize([f"Total number of words: {totalWords}"]).saveAsTextFile(outputUri + "_total")
