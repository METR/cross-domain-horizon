from datasets import load_dataset
import toml
import pathlib



scores_file = pathlib.Path("data/scores/hendrycks_math.toml")

source = "nlile/math_benchmark_test_saturation"
ds = load_dataset(source)["train"]

# Get the scores for each model
scores = {}
for i, row in enumerate(ds):
    model = row["model"]
    scores[model] = row["accuracy"]



scores_data = dict(
    source=source,
    splits=dict(all=scores)
)

with open(scores_file, "w") as f:
    toml.dump(scores_data, f)


