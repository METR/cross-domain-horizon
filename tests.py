import numpy as np
from classes import BenchmarkScoresSpec, SplitScoresSpec
import pathlib
import pytest
import tomllib

from mle import sigmoid, estimate_params_mle
from calculate_horizons import estimate_horizon_binsearch


BENCHMARKS_NO_SPLITS = ["gpqa", "aime", "osworld", "hcast_r_s", "hendrycks_math", "livecodebench_2411_2505_approx", "rlbench", "webarena"]

@pytest.mark.parametrize("benchmark", BENCHMARKS_NO_SPLITS)
def test_no_splits(benchmark):
    beta = 0.6
    dataset_file = pathlib.Path(f"data/benchmarks/{benchmark}.toml") # the lengths
    # scores_file = pathlib.Path(f"data/scores/{benchmark}.toml") # the scores

    with open(dataset_file, "rb") as f:
        data = tomllib.load(f)
    # with open(scores_file, "rb") as f:
    #     scores_data = tomllib.load(f)
    
    for split_name, split_data in data["splits"].items():
        max_length = max(split_data['lengths'])
        min_length = min(split_data['lengths'])
        chance_accuracy = data["chance_accuracy"]

        for horizon in [min_length, (min_length+max_length)/2, max_length]:
            predicted_scores = sigmoid(horizon, split_data['lengths'], beta, chance_accuracy)

            score = np.mean(predicted_scores)
            scorespec = SplitScoresSpec(
                lengths=split_data['lengths'],
                scores=score
            )

            bspec = BenchmarkScoresSpec(
                name=benchmark,
                chance_accuracy=chance_accuracy,
                splits={"all":scorespec}, 
            )
            relative_error = abs(estimate_horizon_binsearch(score, bspec).horizon - horizon)/horizon
            assert relative_error < 0.20, f"RELATIVE ERROR GREATER THAN 5%: {relative_error*100}%"


BENCHMARKS_WITH_SPLITS = ["livecodebench_2411_2505","mock_aime","video_mme", "gpqa_diamond", "swe_bench_verified"] # datasets with splits

@pytest.mark.parametrize("benchmark", BENCHMARKS_WITH_SPLITS)
def test_with_splits(benchmark):
    dataset_file = pathlib.Path(f"data/benchmarks/{benchmark}.toml") # the lengths
    scores_file = pathlib.Path(f"data/scores/{benchmark}.toml") # the scores

    with open(dataset_file, "rb") as f:
        data = tomllib.load(f)
    with open(scores_file, "rb") as f:
        scores_data = tomllib.load(f)
    
    split_specs = {}
    chance_accuracy = data["chance_accuracy"]

    print("benchmark: ", benchmark)
    for split_name, split_data in data["splits"].items():
        lengths = split_data["lengths"]
        scores = scores_data["splits"][split_name]
        scores = {k:float(v) / 100 for k, v in scores.items()}
        assert all(0 <= score <= 1 for score in scores.values())
        split_specs[split_name] = SplitScoresSpec(lengths=lengths, scores=scores)

    actual_bspec=BenchmarkScoresSpec(name=benchmark, chance_accuracy=chance_accuracy, splits=split_specs)        
    min_time = 100 
    max_time = 0
    for split_name, split_data in actual_bspec.splits.items():
        min_time = min(min_time, min(split_data.lengths))
        max_time = max(max_time, max(split_data.lengths))
    print("min_time: ", min_time, "max_time: ", max_time)

    max_relative_error_beta = 0
    max_relative_error_horizon= 0
    for horizon in [min_time, (min_time+max_time)/2, max_time]:
        if horizon < 0.1:
            continue
        for beta in [0.5, 0.6, 0.7]:
            split_dict = {}
            for split_name, split_data in actual_bspec.splits.items():
                split_dict[split_name] = sigmoid(horizon, split_data.lengths, beta, chance_accuracy)
                actual_bspec.splits[split_name].scores = {"fake model": np.mean(split_dict[split_name])}

            params = estimate_params_mle('fake model', actual_bspec)
            est_horizon = params.horizon
            est_beta = params.slope

            relative_error_beta = abs(est_beta - beta) / beta
            relative_error_horizon = abs(est_horizon - horizon) / horizon

            max_relative_error_beta = max(max_relative_error_beta, relative_error_beta)
            max_relative_error_horizon = max(max_relative_error_horizon, relative_error_horizon)

    assert max_relative_error_beta < 0.05, f"RELATIVE ERROR GREATER THAN 5%: {max_relative_error_beta*100:.2f}%"
    assert max_relative_error_horizon < 0.05, f"RELATIVE ERROR GREATER THAN 5%: {max_relative_error_horizon*100:.2f}%"


