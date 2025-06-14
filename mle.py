import numpy as np
import pandas as pd
import pathlib
from typing import Iterator
from dataclasses import dataclass
import tomllib
import argparse
import scipy.stats
import math
from scipy.stats import beta as beta_dist, poisson_binom as poisson_binom_dist
from scipy.optimize import minimize
from collections import namedtuple
from classes import BenchmarkScoresSpec

@dataclass
class ModelParams:
    horizon: float
    slope: float | None
    slope_method: str | None = None
    score: float | None = None

def sigmoid(horizon, task_len, slope, chance_accuracy) -> np.ndarray:
    result = 1 / (1 + np.exp(slope * (-np.log2(horizon) + np.log2(task_len))))
    return chance_accuracy + (1 - chance_accuracy) * result


# Log-likelihood for one model
def beta_nlog_likelihood(params: tuple[float, float], observed_scores:dict[str, float], task_lengths:dict[str, float], chance_accuracy, model_name=None, verbose=False):
    h, slope = params
    lls = []
    for split_idx, score in observed_scores.items():
        # Predicted probability for each question in this split
        predicted_scores = sigmoid(h, task_lengths[split_idx], slope, chance_accuracy)

        split_size = len(predicted_scores)
        score_on_split = score * split_size
        score_floor = math.floor(score_on_split)
        if score_floor == score_on_split:
            split_ll = poisson_binom_dist.logpmf(score_floor, predicted_scores)
        else:
            score_ceil = score_floor + 1 # not exactly ceil if it's an integer
            split_lls = poisson_binom_dist.logpmf([score_floor,score_ceil], predicted_scores)
            split_ll = (score_ceil - score_on_split) * split_lls[0] + (score_on_split - score_floor) * split_lls[1]
            assert (split_lls[0] >= split_ll >= split_lls[1]) or (split_lls[1] >= split_ll >= split_lls[0]), f"model: {model_name}, split_lls: {split_lls}, split_ll: {split_ll}, split_size: {split_size}, score: {score}, score_on_split: {score_on_split}, score_floor: {score_floor}, score_ceil: {score_ceil}, h: {h}, slope: {slope}, task_lengths: {task_lengths[split_idx]}"
        lls.append(split_ll)

    ll = sum(lls)
    return -ll

def estimate_params_mle(model_name: str, bspec: BenchmarkScoresSpec):
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
 
    result = minimize(beta_nlog_likelihood, x0=[5.0, 0.5], bounds=[(0, None), (0.01, None)], args=(observed_scores, task_lengths, chance_accuracy, model_name))
    return ModelParams(horizon=float(result.x[0]), slope=float(result.x[1]), slope_method="mle", score=np.mean(list(observed_scores.values())))