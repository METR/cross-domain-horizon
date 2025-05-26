import pandas as pd
import os
import glob
from scipy.stats import gmean
import yaml

DATA_DIR = 'data'
HORIZONS_DIR = os.path.join(DATA_DIR, 'horizons') # New constant for horizons dir
RELEASE_DATES_FILE = os.path.join(DATA_DIR, 'raw', 'model_info.yaml') # Path to release dates
MIN_HORIZON_THRESHOLD_SECONDS = 10 * 60  # 10 minutes

def load_release_dates(release_dates_file):
    """Loads model release dates and handles aliases from a YAML file."""
    try:
        with open(release_dates_file, 'r') as f:
            release_data = yaml.safe_load(f)
            model_info = release_data.get('model_info', {})
            if not model_info:
                error_msg = f"Error: No 'model_info' key found or empty in {release_dates_file}. Cannot proceed."
                print(error_msg)
                raise ValueError(error_msg)

        all_releases = []
        for model, data in model_info.items():
            release_date = None
            aliases = []
            if isinstance(data, dict):
                release_date = data.get('release_date')
                aliases = data.get('aliases', [])
            else: # Assume data is the release date string itself
                release_date = data

            if release_date:
                # Add the main model
                all_releases.append({'model': model, 'release_date': release_date})
                # Add aliases with the same release date
                for alias in aliases:
                    all_releases.append({'model': alias, 'release_date': release_date})
            else:
                 print(f"Warning: Missing release_date for model: {model} in {release_dates_file}")


        if not all_releases:
            raise ValueError(f"Warning: No release dates found after processing {release_dates_file}")

        release_df = pd.DataFrame(all_releases)
        # Attempt to convert 'release_date' to datetime, handling potential 'unknown' values
        release_df['release_date'] = pd.to_datetime(release_df['release_date'], errors='coerce')

        return release_df

    except FileNotFoundError:
        error_msg = f"Error: Release dates file not found at {release_dates_file}. Cannot proceed."
        print(error_msg)
        raise
    except Exception as e:
        error_msg = f"An unexpected error occurred while loading release dates from {release_dates_file}: {e}. Cannot proceed."
        print(error_msg)
        raise e


def load_data(data_dir):
    """Loads horizon data and merges release dates."""
    # --- Load Horizon Data ---
    all_data = []
    horizons_dir = os.path.join(data_dir, 'horizons') # Use the specific horizons directory
    csv_files = glob.glob(os.path.join(horizons_dir, '*.csv')) # Find all CSVs in horizons dir

    if not csv_files:
        print(f"Warning: No CSV files found in {horizons_dir}")
        # Return empty DataFrame with all expected columns
        raise ValueError("No data loaded after processing CSV files.")

    for csv_path in csv_files:
        benchmark_name = os.path.basename(csv_path).replace('.csv', '') # Extract benchmark from filename
        print(f"Loading {csv_path}")
        try:
            df = pd.read_csv(csv_path)

            columns_to_use = ['model', 'horizon', 'slope', 'slope_method', 'release_date', 'benchmark']

            for col in columns_to_use:
                if col not in df.columns:
                    df[col] = None

            df['horizon'] = df['horizon'].fillna(0)
            # Convert horizon from minutes to seconds
            df['horizon'] = df['horizon'] * 60
            df['benchmark'] = benchmark_name

            all_data.append(df[columns_to_use])

        except Exception as e:
            print(f"Warning: Could not read {csv_path}. Error: {e}")


    if not all_data:
        print("Warning: No data loaded after processing CSV files.")
        # Return empty DataFrame with all expected columns
        raise ValueError("No data loaded after processing CSV files.")

    horizon_df = pd.concat(all_data, ignore_index=True)


    # --- Load and Merge Release Dates ---
    release_df = load_release_dates(RELEASE_DATES_FILE)

    # Merge with horizon data
    merged_df = pd.merge(horizon_df, release_df, on='model', how='left')
    merged_df['release_date'] = merged_df['release_date_x'].fillna(merged_df['release_date_y'])
    merged_df = merged_df.drop(columns=['release_date_x', 'release_date_y'])
    
    # Convert release dates to yyyy-mm-dd format (strip time component)
    merged_df['release_date'] = pd.to_datetime(merged_df['release_date']).dt.date
    
    print(f"Number of models with release dates: {merged_df['release_date'].count()} / {len(merged_df)}")

    # Check for models without release dates after merge
    missing_dates = merged_df[merged_df['release_date'].isna()]['model'].unique()
    if len(missing_dates) > 0:
        # Filter out models that genuinely had 'unknown' or unparseable dates in the YAML
        # We only want to warn about models present in horizon data but *completely missing* from release info
        models_in_release_info = set(release_df['model'])
        truly_missing = [m for m in missing_dates if m not in models_in_release_info]
        if truly_missing:
            print(f"Warning: Missing release dates for models present in horizon data but not found in {RELEASE_DATES_FILE}: {list(truly_missing)}")

    return merged_df


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