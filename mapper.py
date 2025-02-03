#mapper.py

import sys
import re

# Define a regex pattern for cleaning words (only keep alphabets and numbers)
word_pattern = re.compile(r'\b[a-zA-Z0-9]+\b')

def clean_text(text):
    # Lowercase and filter words based on the pattern
    return word_pattern.findall(text.lower())

def mapper():
    for line in sys.stdin:
        # Skip empty lines
        if not line.strip():
            continue

        # Assuming the line contains a description/title column, extract the text
        # Example line format: title,description,rating_number
        fields = line.strip().split(",")

        # Process only the description (assuming it's the second field)
        if len(fields) >= 2:
            description = fields[1]
            words = clean_text(description)

            for word in words:
                print(f"{word}\t1")

if __name__ == "__main__":
    mapper()
