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
import tomllib
import argparse
import logging

from classes import BenchmarkScoresSpec, SplitScoresSpec
from mle import sigmoid, estimate_params_mle, ModelParams

DEFAULT_SLOPE = 0.6
DEFAULT_CHANCE_ACCURACY = 0.0

BENCHMARKS = ["gpqa", "gpqa_diamond","aime", "mock_aime","osworld", "video_mme", "hcast_r_s", "hendrycks_math", "livecodebench_2411_2505", "swe_bench_verified", "rlbench", "webarena"]


def expected_score(horizon: float, bspec: BenchmarkScoresSpec, slope: float = DEFAULT_SLOPE):
    """
    Gets expected score that horizon `horizon` will produce
    Input:
    - horizon: float
    - lengths: list of ints
    """
    probs_for_horizon = sigmoid(horizon, bspec.splits["all"].lengths, slope, bspec.chance_accuracy)
    return np.mean(probs_for_horizon)


def initial_estimate(score: int, bspec: BenchmarkScoresSpec):
    """
    Estimates the horizon as the value of h for which
    a model would get score `score` with horizon h
    """
    lengths = sorted(bspec.splits["all"].lengths)
    return lengths[min(len(lengths) - 1, int(score))]

def estimate_horizon_binsearch(score: int, bspec: BenchmarkScoresSpec, n_iterations=100, min_horizon=None, max_horizon=None):
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
    lengths = bspec.splits["all"].lengths
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
        return ModelParams(horizon=float('nan'), slope=DEFAULT_SLOPE)
    
    for _ in range(n_iterations):
        # Convert from log space to calculate expected score
        horizon = np.exp(log_horizon)
        expected = expected_score(horizon, bspec=bspec)
        
        # If the expected score is close enough to the target, return the horizon
        if abs(expected - score) < 0.01:
            break
        
        # Binary search in log space
        if expected < score:
            log_min = log_horizon
            log_horizon = (log_horizon + log_max) / 2
        else:
            log_max = log_horizon
            log_horizon = (log_min + log_horizon) / 2
            
    return ModelParams(horizon=float(np.exp(log_horizon)), slope=DEFAULT_SLOPE, score=score)

def estimate_horizons(scores: dict[str, int], bspec: BenchmarkScoresSpec, mle: bool) -> dict[str, float]:
    """
    Estimates time horizon for each model given score and question lengths in minutes

    Input:
    - scores: model name -> # questions correct on the benchmark
    - lengths: 
    """

    result = {}
    for model, score in scores.items():
        # if model != 'google/gemini-2.5-pro-exp-03-25': continue
        if mle:
            result[model] = estimate_params_mle(model, bspec=bspec)
        else:
            result[model] = estimate_horizon_binsearch(score, bspec=bspec)
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

    # TODO edit this for more than one split
    assert "splits" in data, f"Ill-formatted dataset file {dataset_file}"
    assert "splits" in scores_data, f"Ill-formatted scores file {scores_file}"

    chance_accuracy = data["chance_accuracy"]
    split_specs = {}

    use_mle = (len(data["splits"]) > 1)
    # eps = 0.0001
    for split_name, split_data in data["splits"].items():
        if split_name == "all" and use_mle:
            continue
        lengths = split_data["lengths"]
        scores = scores_data["splits"][split_name]
        scores = {k:float(v) / 100 for k, v in scores.items()}
        assert all(0 <= score <= 1 for score in scores.values())
        n_questions = len(lengths)
        split_specs[split_name] = SplitScoresSpec(lengths=lengths, scores=scores)


    bspec = BenchmarkScoresSpec(name=dataset_name, chance_accuracy=chance_accuracy, splits=split_specs)
    logging.info(f"Estimating horizons for {dataset_name} {'with MLE' if use_mle else 'with binsearch'}")
    horizons = estimate_horizons(scores, bspec=bspec, mle=use_mle)

    df = pd.DataFrame({
        'model': list(horizons.keys()),
        'horizon': [h.horizon for h in horizons.values()],
        'slope': [h.slope for h in horizons.values()],
        'score': [h.score for h in horizons.values()],
    })
    df = df.sort_values('horizon', ascending=False)
    df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"Horizons for {dataset_name} saved to {output_file}")



def main(data_path: str) -> None:
    if data_path == "all":
        for benchmark in BENCHMARKS:
            process_dataset(benchmark)
    elif data_path in BENCHMARKS:
        process_dataset(data_path)
    else:
        print(f"Error: Invalid data_path '{data_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate model time horizons based on benchmark scores.")
    parser.add_argument(
        "--data-path",
        type=str,
        choices=BENCHMARKS + ["all"],
        default="all",
        help="Specify the dataset to process ('gpqa', 'aime', or 'all'). Default is 'all'."
    )
    args = parser.parse_args()
    main(data_path=args.data_path)