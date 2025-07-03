import os
import csv
import numpy as np
import toml
import pandas as pd

MODELS_TO_ADD = {
    # https://deepmind.google/technologies/gemini/pro/
    "google_gemini_2_5_pro_exp": 84.8,
}

RAW_DATA_PATH = "data/raw/livecodebench_2411_2505.csv"

OUTPUT_SCORES_PATH = "data/scores/livecodebench_2411_2505.toml"
OUTPUT_LENGTHS_PATH = "data/benchmarks/livecodebench_2411_2505.toml"

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

    for split_name in SPLIT_NAMES + ["all"]:
        for _, row in df_scores.iterrows():
            scores[split_name][row["model"]] = float(row[f"score_{split_name}"])

    print(scores)
    return scores

def generate_toml_file(scores):
    os.makedirs('data/scores', exist_ok=True)

    extracted_scores = scores

    data = {
        "source": "https://livecodebench.github.io/leaderboard.html",
        "splits": extracted_scores,
    }
    
    with open(OUTPUT_SCORES_PATH, 'w') as f:
        toml.dump(data, f)
    print(f"\nGenerated TOML file with scores for {len(scores['short'])} models")


def generate_benchmark_file(output_filename=OUTPUT_LENGTHS_PATH):
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

def generate_horizon_file(scores, output_filename="data/horizons/livecodebench_2411_2505_full_method.csv"):
    """Generate a full-method horizon CSV analogous to hcast_r_s_full_method.csv.

    The horizon is defined as the length (in minutes) at which the model is
    expected to reach 50 % accuracy, assuming an exponential decay of the form
    accuracy = a * exp(b * length).
    """

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    rows = []
    eps = 1e-9

    # Iterate over every model for which we have the three split accuracies
    for model in scores["short"]:
        # Skip models missing any split (should not happen, but guard anyway)
        if any(model not in scores[split] for split in SPLIT_NAMES):
            continue

        # Accuracy values for each representative length
        accs = np.array([
            scores["short"][model],
            scores["med"][model],
            scores["long"][model],
        ]) / 100.0  # convert to fraction

        # Clip to avoid log(0) / division by zero errors
        accs = np.clip(accs, eps, 1 - eps)

        lengths = np.array([
            LENGTH_REPRESENTATIVES["short"],
            LENGTH_REPRESENTATIVES["med"],
            LENGTH_REPRESENTATIVES["long"],
        ])

        # Fit power-law: ln(acc) = ln(a) + c * ln(length)
        log_lengths = np.log(lengths)
        c, log_a = np.polyfit(log_lengths, np.log(accs), 1)

        if abs(c) < 1e-12:
            continue  # degenerate fit

        # Horizon (length at 50% accuracy)
        log_L50 = (np.log(0.5) - log_a) / c
        horizon = float(np.exp(log_L50))  # minutes

        # Slope per log-length doubling (positive means steeper drop)
        slope = float(-c)  # flip sign so larger = steeper degradation (align with HRS)

        overall_score = scores["all"].get(model, float(np.mean(accs) * 100))

        rows.append(dict(model=model, horizon=horizon, slope=slope, score=overall_score))

    df = pd.DataFrame(rows)
    df.to_csv(output_filename, index=False)
    print(f"Generated horizon file: {output_filename} with {len(df)} models.")

if __name__ == "__main__":
    scores = extract_scores()
    scores.update(MODELS_TO_ADD)

    # Persist leaderboard scores
    generate_toml_file(scores)

    # Persist estimated task length distribution
    generate_benchmark_file()

    # Compute and write full-method horizon CSV
    generate_horizon_file(scores)

