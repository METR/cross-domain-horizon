import pandas as pd
from pathlib import Path

# Define paths
source_dir = Path("data/hcast_r_s")
source_file = source_dir / "horizons_raw.csv"
output_file = source_dir / "horizons.csv"

# Ensure output directory exists
source_dir.mkdir(parents=True, exist_ok=True)

# Define the mapping for agent names to standardized model names
name_mapping = {
    "Claude 3 Opus": "anthropic_claude_3_opus",
    "Claude 3.5 Sonnet (New)": "anthropic_claude_3_5_sonnet_latest",
    "Claude 3.5 Sonnet (Old)": "anthropic_claude_3_5_sonnet_old",
    "Claude 3.7 Sonnet": "anthropic_claude_3_7_sonnet",
    "GPT-2": "openai_gpt_2",
    "GPT-4 0125": "openai_gpt_4_0125",
    "GPT-4 0314": "openai_gpt_4_0314",
    "GPT-4 1106": "openai_gpt_4_1106",
    "GPT-4 Turbo": "openai_gpt_4_turbo",
    "GPT-4o": "openai_gpt_4o_2024_05_13",  # Using release date from raw data
    "davinci-002 (GPT-3)": "openai_davinci_002",
    "gpt-3.5-turbo-instruct": "openai_gpt_3_5_turbo_instruct",
    "human": "human",
    "o1": "openai_o1",
    "o1-preview": "openai_o1_preview",
}

# Read the raw CSV
df_raw = pd.read_csv(source_file)

# Select and rename columns
df_processed = df_raw[["agent", "p50"]].copy()
df_processed.rename(columns={"agent": "model", "p50": "horizon"}, inplace=True)

# Apply the name mapping
df_processed["model"] = df_processed["model"].map(name_mapping)

# Drop human from the dataset
df_processed = df_processed[df_processed["model"] != "human"]


# Check for any models that weren't mapped (optional, but good practice)
unmapped = df_processed[df_processed["model"].isna()]
if not unmapped.empty:
    print("Warning: Unmapped models found:")
    print(unmapped)
    # Optionally raise an error or handle them
    # raise ValueError(f"Unmapped models found: {unmapped['model'].unique()}")

# Ensure horizon is numeric (it should be already, but good practice)
df_processed["horizon"] = pd.to_numeric(df_processed["horizon"])

# Write the processed data to the new CSV
df_processed.to_csv(output_file, index=False)

print(f"Successfully processed '{source_file}' and saved to '{output_file}'")
