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

DEFAULT_SLOPE = 0.6

def sigmoid(horizon, x, slope) -> np.ndarray:
    result = 1 / (1 + np.exp(slope * (-np.log2(horizon) + np.log2(x))))
    return result

def expected_score(horizon: float, lengths: list[int]):
    """
    Gets expected score that horizon `horizon` will produce
    Input:
    - horizon: float
    - lengths: list of ints
    """
    probs_for_horizon = sigmoid(horizon, lengths, DEFAULT_SLOPE)
    return np.sum(probs_for_horizon)

    
def initial_estimate(score: int, lengths: list[int]):
    """
    Estimates the horizon as the value of h for which
    a model would get score `score` with horizon h
    """
    lengths = sorted(lengths)
    return lengths[min(len(lengths) - 1, score)]

def estimate_horizon(score: int, lengths: list[int], n_iterations=100, min_horizon=None, max_horizon=None):
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
    # Set default bounds if not provided
    if min_horizon is None:
        min_horizon = 0.1 * min(lengths)
    if max_horizon is None:
        max_horizon = 10 * max(lengths)
    
    # Convert to log space for binary search
    log_min = np.log(min_horizon)
    log_max = np.log(max_horizon)
    
    # Initialize horizon with a reasonable guess
    init_horizon = initial_estimate(score, lengths)
    log_horizon = np.log(max(min_horizon, min(max_horizon, init_horizon)))
    
    # Check if target score is outside the range of what's possible with these bounds
    min_expected = expected_score(min_horizon, lengths)
    max_expected = expected_score(max_horizon, lengths)
    
    if score < min_expected or score > max_expected:
        return float('nan')
    
    for _ in range(n_iterations):
        # Convert from log space to calculate expected score
        horizon = np.exp(log_horizon)
        expected = expected_score(horizon, lengths)
        print(f"Horizon: {horizon:.2f}, Expected: {expected:.2f}, Score: {score}")
        
        # If the expected score is close enough to the target, return the horizon
        if abs(expected - score) < 0.1:
            print(f"Horizon estimate for {score} questions: {horizon:.2f} minutes")
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

def estimate_horizons(scores: dict[str, int], lengths: list[float]) -> dict[str, float]:
    """
    Estimates time horizon for each model given score and question lengths in minutes

    Input:
    - scores: model name -> # questions correct on the benchmark
    - lengths: 
    """

    n_questions = len(lengths)
    assert all(0 <= score <= n_questions for score in scores.values())

    result = {}
    for model, score in scores.items():
        result[model] = estimate_horizon(score, lengths)
    return result
    

def main(data_file: pathlib.Path, output_file: pathlib.Path) -> None:
    with open(data_file, "rb") as f:
        data = tomllib.load(f)

    n_questions = int(data["n_questions"])
    score_percent = data["scores"]
    score_percent = {k: float(v.strip("%")) for k, v in score_percent.items()}
    scores = {k: round(score * n_questions / 100) for k, score in score_percent.items()}
    lengths = data["lengths"]
    assert len(lengths) == n_questions

    horizons = estimate_horizons(scores, lengths)
    df = pd.DataFrame({
        'model': list(horizons.keys()),
        'horizon': list(horizons.values()),
        'score': [score_percent[m] for m in horizons.keys()]
    })
    df = df.sort_values('score', ascending=False)
    df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"Horizons saved to {output_file}")


if __name__ == "__main__":
    main(data_file="data/gpqa.toml", output_file="data/gpqa_horizons.csv")