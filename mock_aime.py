import pandas as pd
from pathlib import Path
import toml
from aime import adjusted_times_per_q, CHANCE_ACCURACY

# Load the Epoch data
df = pd.read_csv("src/epoch/data/parsed.csv")
df_mock_aime = df[df['benchmark'] == 'OTIS Mock AIME 2024-2025']

SCORES_DIR = Path("data/scores")
BENCHMARKS_DIR = Path("data/benchmarks")


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
    
    # Ensure output directory exists
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all unique question IDs and sort them
    question_ids = sorted(df_mock_aime['sample_id'].unique())
    n_questions = len(question_ids)
    
    # Create splits dict with lengths based on question number
    splits = {}
    for question_id in question_ids:
        # Extract question number from ID and map to adjusted_times_per_q index
        if 'Problem-' in question_id:
            problem_num_str = question_id.split('Problem-')[-1]
            try:
                problem_num = int(problem_num_str)
                # Map to 0-based index for adjusted_times_per_q (which has 15 elements)
                index = (problem_num - 1) % 15
                length = adjusted_times_per_q[index]
            except (ValueError, IndexError):
                # Fallback to first element if parsing fails
                length = adjusted_times_per_q[0]
        else:
            # Fallback to first element if format is unexpected
            length = adjusted_times_per_q[0]
            
        splits[question_id] = {"lengths": [length]}
    
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
    print(f"Created lengths for {n_questions} questions using adjusted_times_per_q: {adjusted_times_per_q}")


if __name__ == "__main__":
    write_lengths_data("mock_aime")
    write_scores_data(df_mock_aime, SCORES_DIR, "mock_aime")