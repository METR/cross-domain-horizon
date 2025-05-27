# %%

from datasets import load_dataset
import toml
import pathlib
import csv
import pandas as pd
import numpy as np
import scipy.stats

scores_file = pathlib.Path("data/scores/hendrycks_math.toml")
BASELINES_FILE = pathlib.Path("data/raw/baselines/hendrycks-math.csv")
BENCHMARK_FILE = pathlib.Path("data/benchmarks/hendrycks_math.toml")

source = "nlile/math_benchmark_test_saturation"
ds = load_dataset(source)["train"]

# Get the scores for each model
scores = {}
for i, row in enumerate(ds):
    model = row["model"]
    scores[model] = row["accuracy"]

scores_data = dict(
    source="https://huggingface.co/datasets/nlile/math_benchmark_test_saturation",
    splits=dict(all=scores)
)

with open(scores_file, "w") as f:
    toml.dump(scores_data, f)

# %%

# Now look at the baselines
with open(BASELINES_FILE, "r") as f:
    df = pd.read_csv(f)
    
def geomean(x):
    return np.exp(np.mean(np.log(x)))

df['log2_time_taken_seconds'] = np.log2(df['time_taken_seconds'])

by_level = df.groupby("level")
level_stats = by_level.agg({
    'correct': ['sum', 'count'],
    'log2_time_taken_seconds': ['mean', 'std'],
}).reset_index()

level_stats.columns = ['level', 'num_correct', 'num_total', 'mean_log2_time', 'std_log2_time']
level_stats['mean_log2_time'] = level_stats['mean_log2_time'].round(2)
level_stats['std_log2_time'] = level_stats['std_log2_time'].round(2)

frac_correct = level_stats['num_correct'] / level_stats['num_total']

n_questions = 500

n_questions_by_level = {
    1: 43,
    2: 90,
    3: 105,
    4: 128,
    5: 134,
}

assert sum(n_questions_by_level.values()) == n_questions

def lognormal_quantiles(df_level_stats: pd.DataFrame, num_questions: int, level: int) -> list[float]:
    """
    Given a dataframe of level stats, return the estimated distribution of question lengths.

    - Assume normal distribution with mean, std matching log2 times
    - Get quantiles 0.5/n .. 1 - 0.5/n
    """
    split = df_level_stats[df_level_stats['level'] == level]
    # Fit lognormal
    mu, sigma = split['mean_log2_time'], split['std_log2_time']
    
    # Get quantiles
    quantiles = 2 ** scipy.stats.norm.ppf(np.linspace(0.5/num_questions, 1 - 0.5/num_questions, num_questions), mu, sigma)
    return quantiles

lengths = {
    level: lognormal_quantiles(level_stats, n_questions_by_level[level], level)
    for level in range(1, 6)
}

all_lengths = [round(float(l), 3) for level in range(1, 6) for l in lengths[level]]
assert len(all_lengths) == n_questions


# We don't have MATH benchmark data per split, so the benchmark data
# should aggregate from all splits...
benchmark_data = dict(
    n_questions=n_questions,
    splits=dict(all=all_lengths)
)

with open(BENCHMARK_FILE, "w") as f:
    toml.dump(benchmark_data, f)
# %%
