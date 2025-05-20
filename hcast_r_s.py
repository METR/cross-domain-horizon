import pandas as pd
from pathlib import Path
from utils import make_toml
import pandas as pd
import json

# Define paths
source_dir = Path("data/hcast_r_s")
source_horizons_file = source_dir / "horizons_raw.csv"
source_runs_file = Path("data/raw/hcast_r_s_filtered_runs.jsonl")

output_dir = Path("data/horizons")
output_full_horizons_file = output_dir / "hcast_r_s_full_method.csv"

output_scores_file = Path("data/scores") / "hcast_r_s.toml"

output_benchmarks_dir = Path("data/benchmarks")
output_benchmarks_file = output_benchmarks_dir / "hcast_r_s.toml"

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)
output_benchmarks_dir.mkdir(parents=True, exist_ok=True)

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
    "o3": "openai_o3",
    "o4-mini": "openai_o4_mini",
}

# Read the raw CSV
df_raw = pd.read_csv(source_horizons_file)


print(df_raw.columns)
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
df_processed[["model", "horizon"]].to_csv(output_full_horizons_file, index=False)


def make_scores_toml(df_runs):
    df_scores = df_runs.copy()
    df_scores["success_weight"] = df_scores["equal_task_weight"] * df_scores["score_binarized"]
    df_scores["total_weight"] = df_scores["equal_task_weight"]

    df_scores = df_scores[df_scores["alias"] != "human"]

    df_scores = df_scores.groupby("alias").agg({
        "success_weight": "sum",
        "total_weight": "sum",
    })

    df_scores["average"] = df_scores["success_weight"] / df_scores["total_weight"]

    df_scores["model"] = df_scores.index.map(name_mapping)

    print(df_scores)

    scores_str = '\n'.join([f'{r["model"]} = "{r["average"]:.2%}"' for i, r in df_scores.iterrows()])


    scores_toml = f"""source = "metr"

[scores]
{scores_str}
    """
    return scores_toml

df_runs = pd.read_json(source_runs_file, lines=True)

scores_toml = make_scores_toml(df_runs)
with open(output_scores_file, "w") as f:
    f.write(scores_toml)

print(f"Successfully processed '{source_horizons_file}' and saved to '{output_full_horizons_file}'")


first_every_task = df_runs.groupby("task_id").first().reset_index()["human_minutes"]

print(f"Number of tasks: {len(first_every_task)}")

lengths = sorted(round(x, 4) for x in first_every_task.tolist())

# make toml
toml_str = make_toml(
    benchmark_name="hcast_r_s",
    n_questions=len(first_every_task),
    chance_accuracy=0.0,
    lengths=lengths,
)

# write toml
with open(output_benchmarks_file, "w") as f:
    f.write(toml_str)
