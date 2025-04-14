"""
Applies MLE to get a horizon length estimate given
- percentage score on benchmark
- estimate of time for each question

Algorithm (EM):
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
    result = 1 / (1 + np.exp(-np.log2(horizon) + slope * np.log2(x)))
    return result


def bernoulli_sum_generator(total: int, probs: list[float], rng: np.random.Generator) -> Iterator[list[int]]:
    """
    Generator that uses rejection sampling to sample len(probs) Bernoulli variables
    with probabilities `probs` that sum to a total `total`.
    """
    n = len(probs)
    while True:
        X = rng.binomial(n=[1]*n, p=probs)
        if sum(X) == total:
            yield X


def log_likelihood(probs: list[float], horizon: float, lengths: list[int]):
    """
    Gets expected log likelihood that horizon `horizon` will produce samples `samples`
    Input:
    - probs: list of floats
    - horizon: float
    - lengths: list of ints
    """
    probs_for_horizon = sigmoid(horizon, lengths, DEFAULT_SLOPE)

    

    
def initial_estimate(score: int, lengths: list[int]):
    """
    Estimates the horizon as the value of h for which
    a model would get score `score` with horizon h
    """
    lengths = sorted(lengths)
    return lengths[min(len(lengths) - 1, score)]

def estimate_horizon(score: int, lengths: list[int], n_samples:int=100, rtol = 1e-2, max_iterations=100, seed=0):

    bsgen = bernoulli_sum_generator(score, lengths, rng=np.random.default_rng(seed))
    default_slope=DEFAULT_SLOPE

    first_time = True
    horizon = initial_estimate(score, lengths)
    new_horizon = 0
    while first_time or abs((new_horizon / horizon) - 1) < rtol:
        probs = sigmoid(horizon, lengths, default_slope)
        for _ in range(n_samples):
        X = next(bsgen)
        new_horizon = np.sum(X * lengths)


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
    

def main(data_file: pathlib.Path) -> None:
    with open(data_file, "rb") as f:
        data = tomllib.load(f)

    n_questions = int(data["n_questions"])
    scores = data["scores"]
    scores = {k: float(v.strip("%")) for k, v in scores.items()}
    scores = {k: round(score * n_questions / 100) for k, score in scores.items()}
    # lengths = data["lengths"]
    lengths = np.random.randint(1, 100, size=n_questions)

    horizons = estimate_horizons(scores, lengths)
    print(horizons)


if __name__ == "__main__":
    main("data/gpqa.toml")