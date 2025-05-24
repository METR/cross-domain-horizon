import numpy as np
import pandas as pd
import pathlib
from typing import Iterator
from dataclasses import dataclass
import tomllib
import argparse
from scipy.stats import beta as beta_dist
from scipy.optimize import minimize
from collections import namedtuple


@dataclass
class ModelParams:
    horizon: float
    slope: float | None

@dataclass
class BenchmarkSpec:
    n_questions: int
    lengths: list[int]
    chance_accuracy: float

def sigmoid(horizon, task_len, slope, chance_accuracy) -> np.ndarray:
    result = 1 / (1 + np.exp(slope * (-np.log2(horizon) + np.log2(task_len))))
    return chance_accuracy + (1 - chance_accuracy) * result


# Log-likelihood for one model
def beta_nlog_likelihood(h, slope, observed_scores:dict[str, float], t_tasks_per_split:dict[str, float], chance_accuracy):
    ll = 0
    for split_idx, score in observed_scores.items():
        # Expected score for this split
        predicted_scores = sigmoid(h, t_tasks_per_split[split_idx], slope, chance_accuracy)

        mu = np.mean(predicted_scores)
        var = np.mean(predicted_scores * (1 - predicted_scores)) / len(predicted_scores)
    
        # Beta parameters via moment matching
        # Add small epsilon to avoid numerical issues
        eps = 1e-10
        mu = np.clip(mu, eps, 1-eps)
        var = np.clip(var, eps, mu*(1-mu)-eps)
        
        common_factor = mu * (1 - mu) / var - 1
        alpha = mu * common_factor
        beta = (1 - mu) * common_factor
        
        ll += beta_dist.logpdf(score, alpha, beta)
    return -ll

def estimate_params_mle(observed_scores, bspec: BenchmarkSpec):
    """
    Uses MLE, specifically scipy.optimize.minimize, to find the horizon
    and slope consistent with the observed scores

    Consider:
    log parametrizing slope
    """
    t_tasks_per_split = {split_idx: len(bspec.lengths) for split_idx in observed_scores.keys()}
    chance_accuracy = bspec.chance_accuracy
    result = minimize(beta_nlog_likelihood, x0=[1.0, 1.0], args=(observed_scores, t_tasks_per_split, chance_accuracy))
    return ModelParams(horizon=result.x[0], slope=result.x[1])

