import numpy as np
import pandas as pd
import pathlib
from typing import Iterator
from dataclasses import dataclass
import tomllib
import argparse
from scipy.stats import beta as beta_dist


def sigmoid(horizon, task_len, slope, chance_accuracy) -> np.ndarray:
    result = 1 / (1 + np.exp(slope * (-np.log2(horizon) + np.log2(task_len))))
    return chance_accuracy + (1 - chance_accuracy) * result



# Log-likelihood for one model
def beta_log_likelihood(h, slope, observed_scores, t_tasks_per_split, chance_accuracy):
    ll = 0
    for split_idx, score in enumerate(observed_scores):
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
    return ll