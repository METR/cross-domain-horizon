import numpy as np
from dataclasses import dataclass
import math
from scipy.stats import poisson_binom as poisson_binom_dist
from scipy.optimize import minimize
from classes import BenchmarkScoresSpec

@dataclass
class ModelParams:
    horizon: float
    slope: float | None
    slope_method: str | None = None
    score: float | None = None
    # nll (or other error metric) for this parameter fit
    loss: float | None = None

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
        assert not np.isnan(predicted_scores).any(), f"model: {model_name}, h: {h}, slope: {slope}, chance_accuracy: {chance_accuracy}, task_lengths: {task_lengths[split_idx]}"

        split_size = len(predicted_scores)
        score_on_split = score * split_size
        score_floor = math.floor(score_on_split)
        if score_floor == score_on_split:
            split_ll = poisson_binom_dist.logpmf(score_floor, predicted_scores)
        else:
            score_ceil = score_floor + 1 # not exactly ceil if it's an integer
            split_lls = poisson_binom_dist.logpmf([score_floor,score_ceil], predicted_scores)
            assert score_floor < score_ceil <= len(predicted_scores), f"model: {model_name} has a score of {score} on split {split_idx} with {len(predicted_scores)} questions"
            split_ll = (score_ceil - score_on_split) * split_lls[0] + (score_on_split - score_floor) * split_lls[1]
            assert (split_lls[0] >= split_ll >= split_lls[1]) or (split_lls[1] >= split_ll >= split_lls[0]), f"model: {model_name}, split_lls: {split_lls}, split_ll: {split_ll}, split_size: {split_size}, score: {score}, score_on_split: {score_on_split}, score_floor: {score_floor}, score_ceil: {score_ceil}, h: {h}, slope: {slope}, task_lengths: {task_lengths[split_idx]}"
        lls.append(split_ll)

    ll = sum(lls)
    if verbose: print(f"{model_name}: h: {h}, slope: {slope}, ll: {ll}, lls: {lls}")
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
 
    result = minimize(beta_nlog_likelihood, x0=[5.0, 0.5], bounds=[(0.01, None), (0.01, None)], args=(observed_scores, task_lengths, chance_accuracy, model_name), method="SLSQP")
    # store optimizer's nll 
    return ModelParams(horizon=float(result.x[0]), slope=float(result.x[1]), slope_method="mle", score=np.mean(list(observed_scores.values())), loss=float(result.fun))

if __name__ == "__main__":
    from classes import SplitScoresSpec
    estimate_params_mle("google_gemini_2_5_pro_exp", BenchmarkScoresSpec(
        name="example_benchmark",
        chance_accuracy=0.01,
        splits={
            "short": SplitScoresSpec(lengths=[10, 20, 30], scores={"google_gemini_2_5_pro_exp": 0.5}),
            "med": SplitScoresSpec(lengths=[10, 20, 30], scores={"google_gemini_2_5_pro_exp": 0.5}),
            "long": SplitScoresSpec(lengths=[10, 20, 30], scores={"google_gemini_2_5_pro_exp": 0.5}),
        }
    ))