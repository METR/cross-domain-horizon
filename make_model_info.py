import yaml
from pathlib import Path
import datetime

# Define paths relative to the script location or workspace root
# Assuming the script runs from the workspace root
input_file = Path("data/raw/release_dates.yaml")
output_file = Path("data/interim/model_info.yaml")

# Hardcoded company mapping based on model name prefixes
# Add more mappings as needed
company_mapping = {
    "OpenAI": ["GPT", "gpt", "o1", "davinci"],
    "Anthropic": ["Claude"],
}

def get_company(model_name, mapping):
    """Determines the company based on the model name using prefixes."""
    for company, prefixes in mapping.items():
        for prefix in prefixes:
            # Check if the model name starts with any of the prefixes
            if model_name.startswith(prefix):
                return company
    # Optionally, handle models that don't match any prefix
    print(f"Warning: Could not determine company for model: {model_name}")
    return "Unknown" # Default company if no match is found

def main():
    # Load input data from the YAML file
    try:
        with open(input_file, 'r') as f:
            input_data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing input YAML file: {e}")
        return

    # Check if the expected 'date' key exists in the loaded data
    if not input_data or 'date' not in input_data:
        print(f"Error: Input file {input_file} does not contain 'date' key or is empty.")
        return

    models_info = {}
    # Process each model entry in the 'date' dictionary
    for model_name, release_date in input_data['date'].items():
        company = get_company(model_name, company_mapping)

        # Use the date string directly from the input
        date_str = str(release_date)

        # Structure the data for the output YAML
        models_info[model_name] = {
            "company": company,
            "release_date": date_str,
            "aliases": [], # Aliases are initially empty as requested
        }

    output_data = {"models": models_info}

    # Ensure the output directory exists before writing
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write the structured data to the output YAML file
    try:
        with open(output_file, 'w') as f:
            # Use sort_keys=False to maintain the order from the input file
            yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)
        print(f"Successfully created {output_file}")
    except IOError as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    main() 