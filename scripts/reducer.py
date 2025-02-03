import sys

current_title = None
current_count = 0

for line in sys.stdin:
    line = line.strip()
    title, count = line.split("\t")
    count = int(count)

    if current_title == title:
        current_count += count
    else:
        if current_title:
            print(f"{current_title}\t{current_count}")
        current_title = title
        current_count = count

if current_title == title:
    print(f"{current_title}\t{current_count}")
