import json, pathlib

wall_times = []

# Read JSONL format - each line is a separate JSON object
with open("trace.trace", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            if "wallTime" in event:
                wall_times.append(event["wallTime"])
        except json.JSONDecodeError:
            # Skip malformed lines
            continue

assert wall_times, "No wallTime fields found"

duration_sec = (max(wall_times) - min(wall_times)) / 1000
print(f"Human time for this task: {duration_sec:.1f} seconds")
