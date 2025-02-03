#!/usr/bin/env python3

import sys

current_word = None
current_count = 0

def reducer():
    global current_word, current_count

    for line in sys.stdin:
        word, count = line.strip().split("\t")

        try:
            count = int(count)
        except ValueError:
            continue

        # If this is a new word, print the previous word count
        if current_word and current_word != word:
            print(f"{current_word}\t{current_count}")
            current_count = 0

        # Accumulate the count for the current word
        current_word = word
        current_count += count

    # Print the last word count
    if current_word:
        print(f"{current_word}\t{current_count}")

if __name__ == "__main__":
    reducer()
