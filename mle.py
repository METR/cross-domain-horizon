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
    slope_method: str | None = None

@dataclass
class SplitSpec:
    lengths: list[int]
    scores: dict[str, float]

@dataclass
class BenchmarkSpec:
    name: str
    chance_accuracy: float
    splits: dict[str, SplitSpec]

def sigmoid(horizon, task_len, slope, chance_accuracy) -> np.ndarray:
    result = 1 / (1 + np.exp(slope * (-np.log2(horizon) + np.log2(task_len))))
    return chance_accuracy + (1 - chance_accuracy) * result


# Log-likelihood for one model
def beta_nlog_likelihood(params, model_name, observed_scores:dict[str, float], task_lengths:dict[str, float], chance_accuracy):
    h, slope = params
    ll = 0
    for split_idx, score in observed_scores.items():
        # Expected score for this split
        predicted_scores = sigmoid(h, task_lengths[split_idx], slope, chance_accuracy)
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

def estimate_params_mle(model_name: str, bspec: BenchmarkSpec):
    """
    Uses MLE, specifically scipy.optimize.minimize, to find the horizon
    and slope consistent with the observed scores

    Consider:
    log parametrizing slope
    """
    distinct_splits = {split_name: bspec.splits[split_name] for split_name in bspec.splits.keys() if split_name != "all"}

    task_lengths = {split_name: split.lengths for split_name, split in distinct_splits.items()}
    observed_scores = {split_name: split.scores[model_name] for split_name, split in distinct_splits.items()}
    chance_accuracy = bspec.chance_accuracy
 
    result = minimize(beta_nlog_likelihood, x0=[5.0, 0.5], bounds=[(None, None), (0, None)], args=(model_name, observed_scores, task_lengths, chance_accuracy))
    return ModelParams(horizon=float(result.x[0]), slope=float(result.x[1]), slope_method="mle")