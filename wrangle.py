import pandas as pd
import os
import glob
from scipy.stats import gmean
import numpy as np

DATA_DIR = 'data'
MIN_HORIZON_THRESHOLD_SECONDS = 10 * 60  # 10 minutes

def load_data(data_dir):
    """Loads horizon data from all benchmark subdirectories."""
    all_data = []
    benchmark_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d))]

    for bench_dir in benchmark_dirs:
        benchmark_name = os.path.basename(bench_dir)
        csv_path = os.path.join(bench_dir, 'horizons.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # Assuming columns are 'model', 'horizon' (in minutes), 'score'
                if 'model' in df.columns and 'horizon' in df.columns:
                    # Fill missing horizons with 0 before converting
                    df['horizon'] = df['horizon'].fillna(0)
                    # Convert horizon from minutes to seconds
                    df['horizon'] = df['horizon'] * 60
                    df['benchmark'] = benchmark_name
                    all_data.append(df[['model', 'benchmark', 'horizon']])
                else:
                    print(f"Warning: Skipping {csv_path}. Missing 'model' or 'horizon' column.")
            except Exception as e:
                print(f"Warning: Could not read {csv_path}. Error: {e}")
        else:
            print(f"Warning: No horizons.csv found in {bench_dir}")

    if not all_data:
        return pd.DataFrame(columns=['model', 'benchmark', 'horizon'])

    return pd.concat(all_data, ignore_index=True)

def filter_and_sort_models(df):
    """Filters models based on horizon threshold and count, and sorts by geometric mean.

    Specifically for preparing data for the bar plot.
    """
    if df.empty:
        return df, []

    # Filter 1: Model must have >= MIN_HORIZON_THRESHOLD_SECONDS on at least one benchmark
    max_horizons = df.groupby('model')['horizon'].max()
    models_passing_threshold = max_horizons[max_horizons >= MIN_HORIZON_THRESHOLD_SECONDS].index
    df_filtered_threshold = df[df['model'].isin(models_passing_threshold)].copy()

    if df_filtered_threshold.empty:
        print("No models meet the minimum horizon threshold.")
        return df_filtered_threshold, []

    # Filter 2: Model must have horizons defined for at least 2 benchmarks
    # Count non-zero horizons per model within the threshold-filtered data
    horizon_counts = df_filtered_threshold[df_filtered_threshold['horizon'] > 0].groupby('model')['benchmark'].nunique()
    models_passing_count = horizon_counts[horizon_counts >= 2].index
    filtered_df = df_filtered_threshold[df_filtered_threshold['model'].isin(models_passing_count)].copy()

    if filtered_df.empty:
        print("No models have horizons defined for at least 2 benchmarks after threshold filtering.")
        return filtered_df, []

    # Calculate geometric mean horizon for sorting (using the final filtered data)
    pivot_df = filtered_df.pivot(index='model', columns='benchmark', values='horizon')
    # Replace NaN/0 with a small value (e.g., 1 second) for gmean calculation
    pivot_df = pivot_df.fillna(1).clip(lower=1)

    # Calculate geometric mean across benchmarks for each model
    model_gmean = pivot_df.apply(gmean, axis=1)

    # Sort models by geometric mean ascending
    sorted_models = model_gmean.sort_values().index.tolist()

    return filtered_df, sorted_models 