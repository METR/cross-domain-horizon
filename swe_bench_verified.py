import pandas as pd
from pathlib import Path
import toml
from datasets import load_dataset
import math

# Load the Epoch data
df = pd.read_csv("src/epoch/data/parsed.csv")
df_swe_bench = df[df['benchmark'] == 'SWE-Bench verified']

SCORES_DIR = Path("data/scores")
BENCHMARKS_DIR = Path("data/benchmarks")

MODELS_TO_EXCLUDE = ["deepseek/deepseek-reasoner"]


def get_difficulty_to_time_mapping():
    """
    Map difficulty levels to task lengths in minutes using geometric mean.
    Based on the user's specification:
    - '<15 min fix' -> 7.5 minutes
    - '15 min - 1 hour' -> geometric mean of 15 and 60 = sqrt(15*60) = 30 minutes
    - '1-4 hours' -> geometric mean of 60 and 240 = sqrt(60*240) = 120 minutes
    - '>4 hours' -> assume 4-16 hours range, geometric mean = sqrt(240*960) = 480 minutes
    """
    return {
        '<15 min fix': 7.5,
        '15 min - 1 hour': math.sqrt(15 * 60),  # ~30 minutes
        '1-4 hours': math.sqrt(60 * 240),       # ~120 minutes
        '>4 hours': math.sqrt(240 * 960),       # ~480 minutes
    }


def write_scores_data(df_scores, scores_dir: Path, benchmark: str):
    scores_dir.mkdir(parents=True, exist_ok=True)
    
    # for each question, model -> score
    scores_dict = dict()
    for _, row in df_scores.iterrows():
        question_id = row['sample_id']
        model = row['model']
        score = row['score']
        if model in MODELS_TO_EXCLUDE:
            continue
        if question_id not in scores_dict:
            scores_dict[question_id] = dict()
        scores_dict[question_id][model] = score * 100

    data = dict(
        source="epoch",
        splits=scores_dict,
    )
    with open(scores_dir / f"{benchmark}.toml", "w") as f:
        toml.dump(data, f)


def write_lengths_data(benchmark_name: str):
    # Configuration
    DATASET_OUTPUT_FILE = BENCHMARKS_DIR / f"{benchmark_name}.toml"
    CHANCE_ACCURACY = 0.0  # Assuming 0% chance accuracy for SWE-bench
    
    # Ensure output directory exists
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load the SWE-bench Verified dataset to get difficulty information
    print("Loading SWE-bench Verified dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified")
    
    # Create instance_id to difficulty mapping
    instance_to_difficulty = {}
    for item in dataset['test']:
        instance_to_difficulty[item['instance_id']] = item['difficulty']
    
    # Get difficulty to time mapping
    difficulty_to_time = get_difficulty_to_time_mapping()
    
    # Get all unique question IDs from Epoch data
    question_ids = sorted(df_swe_bench['sample_id'].unique())
    n_questions = len(question_ids)
    
    print(f"Found {n_questions} unique SWE-bench problems")
    
    # Create splits dict with lengths based on difficulty
    splits = {}
    missing_difficulties = []
    
    for question_id in question_ids:
        if question_id in instance_to_difficulty:
            difficulty = instance_to_difficulty[question_id]
            if difficulty in difficulty_to_time:
                length = difficulty_to_time[difficulty]
                splits[question_id] = {"lengths": [length]}
            else:
                missing_difficulties.append((question_id, difficulty))
                # Fallback to median time
                splits[question_id] = {"lengths": [30.0]}
        else:
            missing_difficulties.append((question_id, "NOT_FOUND"))
            # Fallback to median time
            splits[question_id] = {"lengths": [30.0]}
    
    if missing_difficulties:
        print(f"Warning: {len(missing_difficulties)} problems had missing/unknown difficulties")
        for problem_id, diff in missing_difficulties[:5]:  # Show first 5
            print(f"  {problem_id}: {diff}")
    
    # Write dataset metadata output file
    print(f"Writing dataset metadata to {DATASET_OUTPUT_FILE}...")
    
    data = dict(
        n_questions=n_questions,
        chance_accuracy=CHANCE_ACCURACY,
        splits=splits,
        length_type="estimate",
    )
    with open(DATASET_OUTPUT_FILE, "w") as f:
        toml.dump(data, f)
    
    print(f"Successfully created {DATASET_OUTPUT_FILE}")
    
    # Print difficulty distribution
    difficulty_counts = {}
    for question_id in question_ids:
        if question_id in instance_to_difficulty:
            difficulty = instance_to_difficulty[question_id]
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    
    print(f"Difficulty distribution:")
    for difficulty, count in sorted(difficulty_counts.items()):
        time_minutes = difficulty_to_time.get(difficulty, 30.0)
        print(f"  {difficulty}: {count} problems ({time_minutes:.1f} minutes each)")


if __name__ == "__main__":
    write_lengths_data("swe_bench_verified")
    write_scores_data(df_swe_bench, SCORES_DIR, "swe_bench_verified")