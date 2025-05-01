import pandas as pd
import os
import glob
from scipy.stats import gmean
import numpy as np
import yaml
from datetime import datetime

DATA_DIR = 'data'
HORIZONS_DIR = os.path.join(DATA_DIR, 'horizons') # New constant for horizons dir
RELEASE_DATES_FILE = os.path.join(DATA_DIR, 'raw', 'model_info.yaml') # Path to release dates
MIN_HORIZON_THRESHOLD_SECONDS = 10 * 60  # 10 minutes

def load_data(data_dir):
    """Loads horizon data and merges release dates."""
    # --- Load Horizon Data ---
    all_data = []
    horizons_dir = os.path.join(data_dir, 'horizons') # Use the specific horizons directory
    csv_files = glob.glob(os.path.join(horizons_dir, '*.csv')) # Find all CSVs in horizons dir

    if not csv_files:
        print(f"Warning: No CSV files found in {horizons_dir}")
        # Return empty DataFrame with all expected columns
        return pd.DataFrame(columns=['model', 'benchmark', 'horizon', 'release_date'])

    for csv_path in csv_files:
        benchmark_name = os.path.basename(csv_path).replace('.csv', '') # Extract benchmark from filename
        try:
            df = pd.read_csv(csv_path)
            # Assuming columns are 'model', 'horizon' (in minutes)
            if 'model' in df.columns and 'horizon' in df.columns:
                # Fill missing horizons with 0 before converting
                df['horizon'] = df['horizon'].fillna(0)
                # Convert horizon from minutes to seconds
                df['horizon'] = df['horizon'] * 60
                df['benchmark'] = benchmark_name
                # Select only necessary columns
                all_data.append(df[['model', 'benchmark', 'horizon']])
            else:
                print(f"Warning: Skipping {csv_path}. Missing 'model' or 'horizon' column.")
        except Exception as e:
            print(f"Warning: Could not read {csv_path}. Error: {e}")


    if not all_data:
        print("Warning: No data loaded after processing CSV files.")
        # Return empty DataFrame with all expected columns
        return pd.DataFrame(columns=['model', 'benchmark', 'horizon', 'release_date'])

    horizon_df = pd.concat(all_data, ignore_index=True)

    # --- Load and Merge Release Dates ---
    try:
        with open(RELEASE_DATES_FILE, 'r') as f:
            release_data = yaml.safe_load(f)
            # Extract the 'date' dictionary
            release_dates_dict = release_data.get('model_info', {})
            if not release_dates_dict:
                 # Print error and raise if date key is missing or empty
                 error_msg = f"Error: No 'model_info' key found or empty in {RELEASE_DATES_FILE}. Cannot proceed."
                 print(error_msg)
                 raise ValueError(error_msg)
                 # horizon_df['release_date'] = pd.NaT
                 # return horizon_df

        # Flatten nested 'release_date' values if present
        release_df = pd.DataFrame(
            [(m, (d.get('release_date') if isinstance(d, dict) else d))
             for m, d in release_dates_dict.items()],
            columns=['model', 'release_date']
        )
        release_df['release_date'] = pd.to_datetime(release_df['release_date'], errors='coerce')

        # Merge with horizon data
        merged_df = pd.merge(horizon_df, release_df, on='model', how='left')

        # Check for models without release dates after merge
        missing_dates = merged_df[merged_df['release_date'].isna()]['model'].unique()
        if len(missing_dates) > 0:
            print(f"Warning: Missing release dates for models: {list(missing_dates)}")

        return merged_df
    except Exception as e:
        # Print error and raise for any other unexpected issues
        error_msg = f"An unexpected error occurred while loading/merging release dates: {e}. Cannot proceed."
        print(error_msg)
        raise e


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