import sys
import pathlib
import tomllib
import csv
import os # Import os for directory creation


INPUT_FILE = pathlib.Path("data/raw/tesla_fsd_tracker.toml")
OUTPUT_DIR = pathlib.Path("horizons")
OUTPUT_FILE = OUTPUT_DIR / "tesla_driving.csv"
MILES_PER_MINUTE_RATE = 0.5 # Keep this, might be useful later, though not directly in CSV

def load_and_process_tesla_data(input_path: pathlib.Path, rate: float) -> dict:
    """Reads miles and dates data from a TOML file, converts miles to minutes."""
    try:
        with open(input_path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}", file=sys.stderr)
        sys.exit(1)
    except tomllib.TOMLDecodeError as e:
        print(f"Error decoding TOML file {input_path}: {e}", file=sys.stderr)
        sys.exit(1)

    if "scores_miles" not in data:
        print(f"Error: Key 'scores_miles' not found in {input_path}", file=sys.stderr)
        sys.exit(1)
    if "dates" not in data:
        print(f"Error: Key 'dates' not found in {input_path}", file=sys.stderr)
        sys.exit(1)


    scores_miles = data["scores_miles"]
    dates = data["dates"]
    processed_data = {}

    for version, miles in scores_miles.items():
        if not isinstance(miles, (int, float)) or miles < 0:
            print(f"Warning: Invalid miles value '{miles}' for version '{version}'. Skipping.", file=sys.stderr)
            continue
        if version not in dates:
            print(f"Warning: Date not found for version '{version}'. Skipping.", file=sys.stderr)
            continue

        minutes = miles / rate # Calculate minutes even if not directly used in CSV yet
        processed_data[version] = {
            "minutes": int(minutes),
            "date": dates[version]
        }

    # Return source and the processed data containing dates and minutes
    output_structure = {
        "source": data.get("source", "Unknown"),
        "records": processed_data
    }
    return output_structure

if __name__ == "__main__":
    processed_data = load_and_process_tesla_data(INPUT_FILE, MILES_PER_MINUTE_RATE)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write to CSV
    try:
        with open(OUTPUT_FILE, 'w', newline='') as csvfile:
            fieldnames = ['date', 'version', 'horizon']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for version, record in processed_data["records"].items():
                writer.writerow({'date': record['date'], 'version': version, 'horizon': record['minutes']})
        print(f"Successfully wrote data to {OUTPUT_FILE}")
    except IOError as e:
        print(f"Error writing to CSV file {OUTPUT_FILE}: {e}", file=sys.stderr)
        sys.exit(1)

# Remove old printing logic
# print(f"source = "{converted_data['source']}"")
# print("[scores]")
# for version, time_str in converted_data["scores"].items():
    # Ensure keys with periods are quoted if needed for strict TOML
    # key_str = f'"{version}"' if '.' in version else version
    # print(f'{key_str} = "{time_str}"')