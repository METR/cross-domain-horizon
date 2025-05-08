import sys
import pathlib
import csv
import datetime
from collections import defaultdict

INPUT_FILE = pathlib.Path("data/raw/tesla_fsd.tsv")
INTERIM_DIR = pathlib.Path("data/interim")
INTERIM_FILE = INTERIM_DIR / "tesla_fsd.tsv"
OUTPUT_DIR = pathlib.Path("data/horizons")
OUTPUT_CITY_FILE = OUTPUT_DIR / "tesla_fsd_city.csv"
OUTPUT_HWY_FILE = OUTPUT_DIR / "tesla_fsd_hwy.csv"

# Speed constants for converting miles to minutes
CITY_MPH = 30
HWY_MPH = 60

def extract_minor_version(version):
    """Extract minor version (e.g., '13.2.x') from full version string."""
    parts = version.split('.')
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}.x"
    return version

def parse_date(date_str):
    """Parse date string in format M/D/YYYY."""
    try:
        return datetime.datetime.strptime(date_str, "%m/%d/%Y").strftime("%Y-%m-%d")
    except ValueError:
        print(f"Error parsing date: {date_str}", file=sys.stderr)
        return None

def load_and_process_tesla_data(input_path):
    """Read TSV and aggregate data by minor version."""
    try:
        with open(input_path, 'r', newline='') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t')
            
            # Data structures to store aggregated data
            minor_versions = defaultdict(lambda: {
                'city_miles': 0,
                'hwy_miles': 0,
                'city_miles_to_de': 0,
                'hwy_miles_to_de': 0,
                'earliest_date': None,
                'versions': []
            })
            
            for row in reader:
                # Skip rows with missing data
                if not row['Version'] or not row['MinDate']:
                    continue
                
                minor_version = extract_minor_version(row['Version'])
                
                # Parse numeric values
                try:
                    city_miles = float(row['City Miles'].replace(',', ''))
                    hwy_miles = float(row['Hwy Miles'].replace(',', ''))
                    city_miles_to_de = float(row['City Miles to DE'].replace(',', ''))
                    hwy_miles_to_de = float(row['Hwy Miles to DE'].replace(',', ''))
                except (ValueError, KeyError) as e:
                    print(f"Error parsing numeric data for version {row['Version']}: {e}", file=sys.stderr)
                    continue
                
                date = parse_date(row['MinDate'])
                if not date:
                    continue
                
                # Update minor version data
                data = minor_versions[minor_version]
                data['city_miles'] += city_miles
                data['hwy_miles'] += hwy_miles
                
                # For miles to DE, we'll use weighted averages later
                data['city_miles_to_de'] += city_miles_to_de * city_miles
                data['hwy_miles_to_de'] += hwy_miles_to_de * hwy_miles
                
                # Track earliest date
                if data['earliest_date'] is None or date < data['earliest_date']:
                    data['earliest_date'] = date
                
                data['versions'].append(row['Version'])
    
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing TSV file {input_path}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Calculate weighted averages for miles to DE
    processed_data = {}
    for minor_version, data in minor_versions.items():
        if data['city_miles'] > 0 and data['hwy_miles'] > 0:
            city_miles_to_de = data['city_miles_to_de'] / data['city_miles']
            hwy_miles_to_de = data['hwy_miles_to_de'] / data['hwy_miles']
            
            # Convert miles to minutes based on speed
            city_minutes = city_miles_to_de / CITY_MPH * 60
            hwy_minutes = hwy_miles_to_de / HWY_MPH * 60
            
            processed_data[minor_version] = {
                'date': data['earliest_date'],
                'city_miles': data['city_miles'],
                'hwy_miles': data['hwy_miles'],
                'city_miles_to_de': city_miles_to_de,
                'hwy_miles_to_de': hwy_miles_to_de,
                'city_minutes': int(city_minutes),
                'hwy_minutes': int(hwy_minutes)
            }
    
    return processed_data

def save_interim_data(processed_data, output_path):
    """Save interim data to TSV file."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w', newline='') as tsvfile:
            fieldnames = ['minor_version', 'date', 'city_miles', 'hwy_miles', 
                         'city_miles_to_de', 'hwy_miles_to_de']
            writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter='\t')
            
            writer.writeheader()
            # Sort by date before writing
            sorted_versions = sorted(processed_data.items(), key=lambda x: x[1]['date'])
            for version, data in sorted_versions:
                writer.writerow({
                    'minor_version': version,
                    'date': data['date'],
                    'city_miles': data['city_miles'],
                    'hwy_miles': data['hwy_miles'],
                    'city_miles_to_de': data['city_miles_to_de'],
                    'hwy_miles_to_de': data['hwy_miles_to_de']
                })
        print(f"Successfully wrote interim data to {output_path}")
    except IOError as e:
        print(f"Error writing to TSV file {output_path}: {e}", file=sys.stderr)
        sys.exit(1)

def save_horizon_data(processed_data, city_output_path, hwy_output_path):
    """Save horizon data to CSV files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sort by date
    sorted_versions = sorted(processed_data.items(), key=lambda x: x[1]['date'])
    
    try:
        # Save city data
        with open(city_output_path, 'w', newline='') as csvfile:
            fieldnames = ['date', 'model', 'horizon']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for version, data in sorted_versions:
                writer.writerow({
                    'date': data['date'],
                    'model': version,
                    'horizon': data['city_minutes']
                })
        print(f"Successfully wrote city horizon data to {city_output_path}")
        
        # Save highway data
        with open(hwy_output_path, 'w', newline='') as csvfile:
            fieldnames = ['date', 'model', 'horizon']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for version, data in sorted_versions:
                writer.writerow({
                    'date': data['date'],
                    'model': version,
                    'horizon': data['hwy_minutes']
                })
        print(f"Successfully wrote highway horizon data to {hwy_output_path}")
    except IOError as e:
        print(f"Error writing to CSV files: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    processed_data = load_and_process_tesla_data(INPUT_FILE)
    save_interim_data(processed_data, INTERIM_FILE)
    save_horizon_data(processed_data, OUTPUT_CITY_FILE, OUTPUT_HWY_FILE)
