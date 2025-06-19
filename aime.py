import pathlib
import math
import toml

# Constants
DATA_DIR = pathlib.Path("data/benchmarks")
DATASET_OUTPUT_FILE = DATA_DIR / "aime.toml"
N_EXAMS = 4
QUESTIONS_PER_EXAM = 15
TOTAL_TIME_PER_EXAM = 180  # minutes
HARDEST_TO_EASIEST_RATIO = 5
CHANCE_ACCURACY = 0.001

SUCCESS_RATES = [.98, .97, .96, .94, .91, .87, .82, .76, .67, .58, .46, .34, .24, .16, .10]

# Calculate time per question assuming geometric progression
# r^(n-1) = ratio, where n = QUESTIONS_PER_EXAM
# t1 * (r^n - 1) / (r - 1) = total_time
n = QUESTIONS_PER_EXAM
ratio = HARDEST_TO_EASIEST_RATIO
total_time = TOTAL_TIME_PER_EXAM

assert n > 1 and ratio > 1
r = math.pow(ratio, 1 / (n - 1))
t1 = total_time * (r - 1) / (math.pow(r, n) - 1)
times_per_exam = [t1 * math.pow(r, k) for k in range(n)]
assert abs(sum(times_per_exam) - total_time) < 1e-6, f"Total time {sum(times_per_exam)} != {total_time}"

# Format to one decimal place
adjusted_times_per_q = [round(t / s, 1) for t, s in zip(times_per_exam, SUCCESS_RATES)]

# Combine times for all exams
all_times = adjusted_times_per_q * N_EXAMS
total_questions = n * N_EXAMS

# Prepare data for TOML output
data = {
    "n_questions": total_questions,
    "chance_accuracy": CHANCE_ACCURACY,
    "length_type": "estimate",
}

data["splits"] = {
    "all": {
        "lengths": all_times,
    }
}

if __name__ == "__main__":
    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Write to TOML file
    with open(DATASET_OUTPUT_FILE, "w") as f:
        toml.dump(data, f)

    print(f"Successfully generated {DATASET_OUTPUT_FILE}")
    print(f"Calculated times for one exam (unadjusted sum={sum(times_per_exam):.2f} min): {adjusted_times_per_q}")





