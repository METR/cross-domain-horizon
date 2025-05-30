import numpy as np
from scipy.stats import lognorm
import math
import os
import toml

# OSWorld Benchmark Parameters
# Source: Figure 4 and Table 6 from the provided image context.
# Distribution: Lognormal (assumed based on violin plot shape)
# Median: 111.94s
# Difficulty Split: Easy (<60s): 28.72%, Medium (60-180s): 40.11%, Hard (>180s): 30.17%

median_time = 111.94 / 60
n_tasks = 369
human_success_rate = 0.7236
output_dir = "data/benchmarks"
output_filename = os.path.join(output_dir, "osworld.toml")

# Lognormal parameters (mu, sigma of the underlying normal distribution)
# mu = ln(median)
mu = math.log(median_time)

# sigma is chosen to best fit the Easy/Medium/Hard percentages.
# Calculation (see conversation context) suggests sigma ~ 1.04
sigma = 1.0396

# Set a random seed for reproducible results
np.random.seed(42)

# Generate evenly spaced probabilities for quantiles
# Use (i+1)/(n+1) to avoid 0 and 1, which can give -inf/inf for ppf
probabilities = [(i + 1) / (n_tasks + 1) for i in range(n_tasks)]

# Calculate task completion times using the Percent Point Function (ppf) - inverse CDF
# scipy.stats.lognorm uses s=sigma and scale=exp(mu)
sampled_times = lognorm.ppf(probabilities, s=sigma, scale=math.exp(mu))

sampled_times = sampled_times / human_success_rate

# Format the times to 3 decimal places
times = [float(round(t, 3)) for t in sampled_times]

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

data = dict(
    n_questions=n_tasks,
    chance_accuracy=0.0,
    length_type="estimate",
    splits=dict(all=dict(lengths=times)),
)

# Write output to the specified file
with open(output_filename, 'w') as f:
    toml.dump(data, f)

print(f"Successfully generated {output_filename}")
