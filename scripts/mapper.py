import sys
import json

for line in sys.stdin:
    line = line.strip()
    try:
        data = json.loads(line)
        title = data.get("title", "UNKNOWN")
        print(f"{title}\t1")
    except json.JSONDecodeError:
        continue
