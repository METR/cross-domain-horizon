# %%

import toml
import numpy as np

# Read the TOML file
with open("data/benchmarks/hcast_r_s.toml", "r") as f:
    data = toml.load(f)

# Get the lengths array
lengths = data["lengths"]

# Calculate percentiles (10th through 90th in multiples of 10)
percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
results = np.percentile(lengths, percentiles)

# Print the results
for p, value in zip(percentiles, results):
    print(f"{p}th percentile: {value:.4f}")

# %%
