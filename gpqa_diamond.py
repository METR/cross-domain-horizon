# %%

import pandas as pd
from datasets import load_dataset
from pathlib import Path
import toml

df = pd.read_csv("src/epoch/data/parsed.csv")

df_gpqa_diamond = df[df['benchmark'] == 'GPQA diamond']

SCORES_DIR = Path("data/scores")
def write_scores_data(df_scores, scores_dir: Path, benchmark: str):
    scores_dir.mkdir(parents=True, exist_ok=True)
    
    # for each question, model -> score
    scores_dict = dict()
    for _, row in df_scores.iterrows():
        question_id = row['sample_id']
        model = row['model']
        score = row['score']
        if question_id not in scores_dict:
            scores_dict[question_id] = dict()
        scores_dict[question_id][model] = score


    data = dict(
        source = "epoch",
        splits = scores_dict,
    )
    with open(scores_dir / f"{benchmark}.toml", "w") as f:
        toml.dump(data, f)


def write_lengths_data(dataset_name):
        # --- Configuration ---
    DATA_DIR = Path("data/benchmarks")
    DATASET_OUTPUT_FILE = DATA_DIR / f"{dataset_name}.toml"
    CHANCE_ACCURACY = 0.25
    # ---------------------

    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load the GPQA dataset
    print("Loading GPQA dataset...")
    dataset = load_dataset("Idavidrein/gpqa", dataset_name)
    print("Dataset loaded.")
    
    # Get the number of questions
    n_questions = len(dataset["train"])
    
    # Calculate average times
    print("Calculating average times...")
    lengths = [(a+b) / 2 for a, b in zip(dataset["train"]["Self-reported time (minutes)_EV_1"],
               dataset["train"]["Self-reported time (minutes)_EV_2"])]
    
    lengths = [dict(lengths=[l]) for l in lengths]

    ids = dataset["train"]["Record ID"]

    splits = dict(zip(ids, lengths))
    
    # Write dataset metadata output file
    print(f"\nWriting dataset metadata to {DATASET_OUTPUT_FILE}...")

    data = dict(
        n_questions=n_questions,
        chance_accuracy=CHANCE_ACCURACY,
        splits=splits,
        length_type="baseline",
    )
    with open(DATASET_OUTPUT_FILE, "w") as f:
        toml.dump(data, f)

    print(f"\nSuccessfully created {DATASET_OUTPUT_FILE}")


if __name__ == "__main__":
    write_lengths_data("gpqa_diamond")
    write_scores_data(df_gpqa_diamond, SCORES_DIR, "gpqa_diamond")