import os
import csv
import numpy as np
import toml
import pandas as pd

MODELS_TO_ADD = {
    # https://deepmind.google/technologies/gemini/pro/
    "google_gemini_2_5_pro_exp": 84.8,
}

BENCHMARK_NAME = "livecodebench_2411_2505"
APPROX_BENCHMARK_NAME = "livecodebench_2411_2505_approx"
RAW_DATA_PATH = f"data/raw/{BENCHMARK_NAME}.csv"

OUTPUT_SCORES_PATH = f"data/scores/{BENCHMARK_NAME}.toml"
OUTPUT_LENGTHS_PATH = f"data/benchmarks/{BENCHMARK_NAME}.toml"

OUTPUT_APPROX_SCORES_PATH = f"data/scores/{APPROX_BENCHMARK_NAME}.toml"
OUTPUT_APPROX_LENGTHS_PATH = f"data/benchmarks/{APPROX_BENCHMARK_NAME}.toml"

SPLIT_NAMES = ["short", "med", "long"]
# https://chatgpt.com/share/e/6836b7e6-1edc-800c-a188-ac1e1342ffea
SPLIT_SIZES = {
    "short": 74,
    "med": 88,
    "long": 126,
}

N_QUESTIONS = 288

assert sum(SPLIT_SIZES.values()) == N_QUESTIONS

# Representative solution lengths (minutes) used for horizon fitting
LENGTH_REPRESENTATIVES = {
    "short": 8.0,
    "med": 20.0,
    "long": 40.0,
}

CHANCE_ACCURACY = 0.0

def extract_scores():
    scores = {split_name: {} for split_name in SPLIT_NAMES + ["all"]}
    
    with open(RAW_DATA_PATH, 'r') as f:
        lines = f.readlines()
    
    rows = []
    for i in range(0, len(lines), 6):
        rows.append(list(map(str.strip, lines[i:i+6])))
    
    df_scores = pd.DataFrame(rows, columns=["rank", "model", "score_all", "score_short", "score_med", "score_long"])

    def normalize_model_name(name):
        name = name.replace("-", " ").lower()
        name = name.split("(")[0].strip()
        return name

    df_scores["model"] = df_scores["model"].apply(normalize_model_name)
    # Keep only the highest score for each model
    df_scores = df_scores.sort_values('score_all', ascending=False).drop_duplicates(subset=['model'], keep='first')

    splits = SPLIT_NAMES + ["all"]
    for split_name in splits:
        for _, row in df_scores.iterrows():
            scores[split_name][row["model"]] = float(row[f"score_{split_name}"])

    print(scores)
    return scores

def generate_scores_file(scores:dict, output_filename=OUTPUT_SCORES_PATH, approx=False):
    os.makedirs('data/scores', exist_ok=True)

    if approx:
        extracted_scores = {"all": scores["all"]}
    else:
        extracted_scores = scores

    data = {
        "source": "https://livecodebench.github.io/leaderboard.html",
        "splits": extracted_scores,
    }
    
    with open(output_filename, 'w') as f:
        toml.dump(data, f)
    print(f"\nGenerated TOML file with scores for {len(scores['short'])} models")


def generate_benchmark_file(output_filename=OUTPUT_LENGTHS_PATH, approx=False):
    """
    Generates a TOML benchmark definition file for LiveCodeBench.

    Assumes a distribution over difficulties...
    """
    splits = {split_name: dict(lengths=[]) for split_name in SPLIT_NAMES}

    length_geomeans_mins = {
        "short": 8.0,
        "med": 20.0,
        "long": 40.0,
    }

    for split_name in SPLIT_NAMES:
        this_geomean = length_geomeans_mins[split_name]
        this_lengths = np.geomspace(this_geomean / 2, this_geomean * 2, SPLIT_SIZES[split_name])
        splits[split_name]["lengths"] = this_lengths.tolist()

    if approx:
        all_lengths = []
        for split in SPLIT_NAMES:
            all_lengths += splits[split]["lengths"]
        splits = {"all": {"lengths": all_lengths}}

    toml_data = {
        "n_questions": N_QUESTIONS,
        "chance_accuracy": CHANCE_ACCURACY,
        "length_type": "estimate",
        "splits": splits,
    }
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    toml.dump(toml_data, open(output_filename, 'w'))
    
    print(f"Generated benchmark file: {output_filename} with {N_QUESTIONS} questions.")

if __name__ == "__main__":
    scores = extract_scores()
    scores.update(MODELS_TO_ADD)

    # Persist leaderboard scores
    generate_scores_file(scores, OUTPUT_SCORES_PATH)
    generate_scores_file(scores, OUTPUT_APPROX_SCORES_PATH, approx=True)


    # Persist estimated task length distribution
    generate_benchmark_file(OUTPUT_LENGTHS_PATH)
    generate_benchmark_file(OUTPUT_APPROX_LENGTHS_PATH, approx=True)

