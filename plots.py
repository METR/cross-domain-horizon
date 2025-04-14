import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
from scipy.stats import gmean
import numpy as np

DATA_DIR = 'data'
OUTPUT_FILE = 'plots/all.png'
MIN_HORIZON_THRESHOLD_SECONDS = 10 * 60  # 10 minutes
Y_AXIS_MIN_SECONDS = 60  # 1 minute

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
    """Filters models based on horizon threshold and sorts by geometric mean."""
    if df.empty:
        return df, []

    # Filter models: must have >= 10 min horizon on at least one benchmark
    max_horizons = df.groupby('model')['horizon'].max()
    models_to_keep = max_horizons[max_horizons >= MIN_HORIZON_THRESHOLD_SECONDS].index
    filtered_df = df[df['model'].isin(models_to_keep)].copy()

    if filtered_df.empty:
        return filtered_df, []

    # Calculate geometric mean horizon for sorting
    # Pivot to get models as index, benchmarks as columns, horizon as values
    pivot_df = filtered_df.pivot(index='model', columns='benchmark', values='horizon')
    # Replace NaN/0 with 1 for gmean calculation (gmean requires positive values)
    pivot_df = pivot_df.fillna(1).clip(lower=1)
    
    # Calculate geometric mean across benchmarks for each model
    model_gmean = pivot_df.apply(gmean, axis=1)
    
    # Sort models by geometric mean ascending
    sorted_models = model_gmean.sort_values().index.tolist()

    return filtered_df, sorted_models

def plot_horizons(df, sorted_models):
    """Generates and saves the grouped bar plot."""
    if df.empty or not sorted_models:
        print("No data to plot after filtering.")
        return

    # Get unique benchmarks for hue order
    benchmarks = df['benchmark'].unique()
    benchmarks.sort()

    plt.figure(figsize=(15, 8)) # Increased figure size for potentially many models
    
    # Create the bar plot - SWAPPED x and hue
    ax = sns.barplot(
        data=df,
        x='model',
        y='horizon',
        hue='benchmark',
        order=sorted_models,  # Use the sorted model order for the x-axis
        hue_order=benchmarks # Use sorted benchmarks for hue
    )

    # Set y-axis to log scale starting at 1 minute
    ax.set_yscale('log')
    ax.set_ylim(bottom=Y_AXIS_MIN_SECONDS)

    # Add labels and title - UPDATED x-label
    ax.set_xlabel("Model (Sorted by GeoMean Horizon)")
    ax.set_ylabel("Horizon (seconds, log scale)")
    ax.set_title("Model Horizon across Benchmarks")

    # Improve layout and legend - UPDATED legend title and x-ticks
    plt.xticks(rotation=75, ha='right') # Rotate more if many models
    plt.legend(title='Benchmark', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Save the plot
    plt.savefig(OUTPUT_FILE)
    print(f"Plot saved to {OUTPUT_FILE}")
    plt.close() # Close the plot to free memory

def main():
    df = load_data(DATA_DIR)
    if df.empty:
        print("No data loaded. Exiting.")
        return
        
    filtered_df, sorted_models = filter_and_sort_models(df)
    plot_horizons(filtered_df, sorted_models)

if __name__ == "__main__":
    main()
