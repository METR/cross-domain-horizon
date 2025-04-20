"""
Applies MLE to get a horizon length estimate given
- percentage score on benchmark
- estimate of time for each question

Algorithm (EM), not used:
- For each model, iterate until convergence:
  - Estimate Q(horizon | horizon_t)
    - Estimate / sample from distribution over which questions are correct
        - sample from distribution of bernoulli variables X
        - is there closed form? or do I need to use MCMC?
  - Argmax Q to get horizon_(t+1)

Note we can just do binary search if we know the slope which would be WAY simpler
"""

import numpy as np
import pandas as pd
import pathlib
from typing import Iterator
from dataclasses import dataclass
import tomllib
import argparse

DEFAULT_SLOPE = 0.6
DEFAULT_CHANCE_ACCURACY = 0.0

@dataclass
class BenchmarkSpec:
    n_questions: int
    lengths: list[int]
    chance_accuracy: float

def sigmoid(horizon, x, slope, chance_accuracy) -> np.ndarray:
    result = 1 / (1 + np.exp(slope * (-np.log2(horizon) + np.log2(x))))
    return chance_accuracy + (1 - chance_accuracy) * result

def expected_score(horizon: float, bspec: BenchmarkSpec):
    """
    Gets expected score that horizon `horizon` will produce
    Input:
    - horizon: float
    - lengths: list of ints
    """
    probs_for_horizon = sigmoid(horizon, bspec.lengths, DEFAULT_SLOPE, bspec.chance_accuracy)
    return np.sum(probs_for_horizon)

    
def initial_estimate(score: int, bspec: BenchmarkSpec):
    """
    Estimates the horizon as the value of h for which
    a model would get score `score` with horizon h
    """
    lengths = sorted(bspec.lengths)
    return lengths[min(len(lengths) - 1, score)]

def estimate_horizon(score: int, bspec: BenchmarkSpec, n_iterations=100, min_horizon=None, max_horizon=None):
    """
    Estimates the horizon as the value of h for which
    a model would get mean score `score` with horizon h

    Algorithm:
    - Initialize horizon
    - Binary search in log space until convergence:
      - Estimate expected score for horizon
      - Update horizon
      
    Args:
        score: Target score to achieve
        lengths: List of question lengths
        n_iterations: Maximum number of iterations for binary search
        min_horizon: Minimum possible horizon (default: 0.1 * min(lengths))
        max_horizon: Maximum possible horizon (default: 10 * max(lengths))
        
    Returns:
        Estimated horizon or nan if horizon is outside bounds
    """
    lengths = bspec.lengths
    # Set default bounds if not provided
    if min_horizon is None:
        min_horizon = 0.1 * min(lengths)
    if max_horizon is None:
        max_horizon = 10 * max(lengths)
    
    # Convert to log space for binary search
    log_min = np.log(min_horizon)
    log_max = np.log(max_horizon)
    
    # Initialize horizon with a reasonable guess
    init_horizon = initial_estimate(score, bspec=bspec)
    log_horizon = np.log(max(min_horizon, min(max_horizon, init_horizon)))
    
    # Check if target score is outside the range of what's possible with these bounds
    min_expected = expected_score(min_horizon, bspec=bspec)
    max_expected = expected_score(max_horizon, bspec=bspec)
    
    if score < min_expected or score > max_expected:
        return float('nan')
    
    for _ in range(n_iterations):
        # Convert from log space to calculate expected score
        horizon = np.exp(log_horizon)
        expected = expected_score(horizon, bspec=bspec)
        
        # If the expected score is close enough to the target, return the horizon
        if abs(expected - score) < 0.1:
            return horizon
        
        # Binary search in log space
        if expected < score:
            log_min = log_horizon
            log_horizon = (log_horizon + log_max) / 2
        else:
            log_max = log_horizon
            log_horizon = (log_min + log_horizon) / 2
            
    result = np.exp(log_horizon)
    return result

def estimate_horizons(scores: dict[str, int], bspec: BenchmarkSpec) -> dict[str, float]:
    """
    Estimates time horizon for each model given score and question lengths in minutes

    Input:
    - scores: model name -> # questions correct on the benchmark
    - lengths: 
    """

    assert all(0 <= score <= bspec.n_questions for score in scores.values())

    result = {}
    for model, score in scores.items():
        result[model] = estimate_horizon(score, bspec=bspec)
    return result
    

def process_dataset(dataset_name: str) -> None:
    """Processes a single dataset (e.g., 'gpqa', 'aime')."""
    dataset_file = pathlib.Path(f"data/benchmarks/{dataset_name}.toml")
    scores_file = pathlib.Path(f"data/scores/{dataset_name}.toml")
    output_file = pathlib.Path(f"data/horizons/{dataset_name}.csv")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(dataset_file, "rb") as f:
        data = tomllib.load(f)

    with open(scores_file, "rb") as f:
        scores_data = tomllib.load(f)

    n_questions = int(data["n_questions"])
    score_percent = scores_data["scores"]
    score_percent = {k: float(v.strip("%")) for k, v in score_percent.items()}
    scores = {k: round(score * n_questions / 100) for k, score in score_percent.items()}
    lengths = data["lengths"]
    assert len(lengths) == n_questions

    bspec = BenchmarkSpec(n_questions=n_questions, lengths=lengths, chance_accuracy=DEFAULT_CHANCE_ACCURACY)
    horizons = estimate_horizons(scores, bspec=bspec)
    df = pd.DataFrame({
        'model': list(horizons.keys()),
        'horizon': list(horizons.values()),
        'score': [score_percent[m] for m in horizons.keys()]
    })
    df = df.sort_values('score', ascending=False)
    df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"Horizons for {dataset_name} saved to {output_file}")

benchmarks = ["gpqa", "aime", "osworld"]

def main(data_path: str) -> None:
    if data_path == "all":
        for benchmark in benchmarks:
            process_dataset(benchmark)
    elif data_path in benchmarks:
        process_dataset(data_path)
    else:
        print(f"Error: Invalid data_path '{data_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate model time horizons based on benchmark scores.")
    parser.add_argument(
        "--data-path",
        type=str,
        choices=benchmarks + ["all"],
        default="all",
        help="Specify the dataset to process ('gpqa', 'aime', or 'all'). Default is 'all'."
    )
    args = parser.parse_args()
    main(data_path=args.data_path)